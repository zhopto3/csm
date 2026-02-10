from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
        max_batch_size:int = 64
    ):
        self._model = model

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device
        self.max_batch_size = max_batch_size

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        # eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        # audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    def _prepare_batch_prompts(
        self,
        texts: List[str],
        speakers: List[int],
        contexts: List[List[Segment]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batched prompts with padding.
        
        Returns:
            tokens: (batch_size, max_seq_len, 33)
            tokens_mask: (batch_size, max_seq_len, 33)
            lengths: (batch_size,) actual sequence lengths before padding
        """
        batch_size = len(texts)
        all_tokens = []
        all_masks = []
        
        for text, speaker, context in zip(texts, speakers, contexts):
            tokens, tokens_mask = [], []
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)

            prompt_tokens = torch.cat(tokens, dim=0)
            prompt_tokens_mask = torch.cat(tokens_mask, dim=0)
            
            all_tokens.append(prompt_tokens)
            all_masks.append(prompt_tokens_mask)
        
        # Get max length and pad
        lengths = torch.tensor([t.size(0) for t in all_tokens], device=self.device)
        max_len = lengths.max().item()
        
        padded_tokens = torch.zeros(batch_size, max_len, 33, dtype=torch.long, device=self.device)
        padded_masks = torch.zeros(batch_size, max_len, 33, dtype=torch.bool, device=self.device)
        
        for i, (tokens, masks) in enumerate(zip(all_tokens, all_masks)):
            seq_len = tokens.size(0)
            padded_tokens[i, :seq_len] = tokens
            padded_masks[i, :seq_len] = masks
        
        return padded_tokens, padded_masks, lengths

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        output_logits: bool = False
    ) -> torch.Tensor:
        result = self.generate_batch(
            texts=[text],
            speakers=[speaker],
            contexts=[context],
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            output_logits=output_logits
        )
        
        if output_logits:
            return result[0][0], result[1][0]
        return result[0]

    @torch.inference_mode()
    def generate_batch(
        self,
        texts: List[str],
        speakers: List[int] = None,
        contexts: List[List[Segment]] = None,
        max_audio_length_ms: float = 90_000,
        temperature: float = 1.0,
        topk: int = 50,
        output_logits: bool = False
        ) :
        """
        Generate audio for multiple inputs in parallel.
        
        Args:
            texts: List of text strings to generate
            speakers: List of speaker IDs
            contexts: List of context segments for each generation
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            output_logits: Whether to return log probabilities
            
        Returns:
            If output_logits=False: List of generated audio tensors
            If output_logits=True: Tuple of (audio_outputs, log_prob_results)
                where log_prob_results is a list of dicts containing:
                    - 'total_log_prob': sum of log probs across all frames
                    - 'avg_log_prob': average log prob per frame
                    - 'num_frames': number of generated frames
                    - 'log_probs': list of per-frame log probs (optional)
        """
        batch_size = len(texts)

        # Set defaults if not provided
        if speakers is None:
            speakers = [0] * batch_size  # Default to speaker 0 for all
        if contexts is None:
            contexts = [[]] * batch_size  # Empty context for all
            
        assert batch_size <= self.max_batch_size, f"Batch size {batch_size} exceeds max {self.max_batch_size}"
        assert len(speakers) == batch_size and len(contexts) == batch_size
        # resize KV caches to the actual batch size
        if not self._model.backbone.caches_are_enabled():
            self._model.setup_caches(batch_size)
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len

        # Prepare batched prompts
        prompt_tokens, prompt_tokens_mask, prompt_lengths = self._prepare_batch_prompts(
            texts, speakers, contexts
        )
        
        if prompt_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        # Track which sequences are still generating
        is_generating = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        # Store samples for each batch item
        batch_samples = [[] for _ in range(batch_size)]
        batch_log_probs = [[] for _ in range(batch_size)] if output_logits else None
        
        curr_tokens = prompt_tokens
        curr_tokens_mask = prompt_tokens_mask
        curr_pos = torch.arange(0, prompt_tokens.size(1), device=self.device).unsqueeze(0).repeat(batch_size, 1)

        # Process the prompt first
        if output_logits:
            samples, log_probs = self._model.generate_frame(
                curr_tokens, curr_tokens_mask, curr_pos, temperature, topk, return_logits=True
            )
        else:
            samples = self._model.generate_frame(
                curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
            )
        
        # Check if prompt generation already hit EOS
        for i in range(batch_size):
            if torch.all(samples[i] == 0):
                is_generating[i] = False
            else:
                batch_samples[i].append(samples[i])
                if output_logits:
                    # log_probs should be (batch_size, num_codebooks) or (batch_size, num_codebooks, 1)
                    # Sum across codebooks to get total log prob for this frame
                    frame_log_prob = log_probs[i].sum().item()
                    batch_log_probs[i].append(frame_log_prob)
        
        # Now set up for autoregressive generation (1 token at a time)
        next_frame = torch.zeros(batch_size, 1, 33, dtype=torch.long, device=self.device)
        next_frame[:, 0, :-1] = samples
        next_frame_mask = torch.zeros(batch_size, 1, 33, dtype=torch.bool, device=self.device)
        next_frame_mask[:, 0, :-1] = True
        
        curr_tokens = next_frame
        curr_tokens_mask = next_frame_mask
        curr_pos = curr_pos[:, -1:] + 1

        for step in range(max_generation_len - 1):
            if output_logits:
                samples, log_probs = self._model.generate_frame(
                    curr_tokens, curr_tokens_mask, curr_pos, temperature, topk, return_logits=True
                )
                # Store log probs for sequences still generating
                for i in range(batch_size):
                    if is_generating[i]:
                        frame_log_prob = log_probs[i].sum().item()
                        batch_log_probs[i].append(frame_log_prob)
            else:
                samples = self._model.generate_frame(
                    curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                )
            
            # Check for EOS (all zeros) and store samples
            for i in range(batch_size):
                if is_generating[i]:
                    if torch.all(samples[i] == 0):
                        is_generating[i] = False
                    else:
                        batch_samples[i].append(samples[i])
            
            # If all sequences are done, break
            if not is_generating.any():
                break

            # Prepare next step - only for sequences still generating
            # We still need to process all batch items to maintain cache consistency
            # samples is (batch_size, num_codebooks=32), we need (batch_size, 1, 33)
            next_frame = torch.zeros(batch_size, 1, 33, dtype=torch.long, device=self.device)
            next_frame[:, 0, :-1] = samples  # audio tokens in first 32 positions
            # position 32 (index -1) stays 0 for text token
            
            next_frame_mask = torch.zeros(batch_size, 1, 33, dtype=torch.bool, device=self.device)
            next_frame_mask[:, 0, :-1] = True  # mask for audio tokens
            
            curr_tokens = next_frame
            curr_tokens_mask = next_frame_mask
            curr_pos = curr_pos[:, -1:] + 1
        
        max_frames = max(len(samples) for samples in batch_samples)
        
        if max_frames == 0:
            # All generations were empty
            audio_outputs = [torch.zeros(1, device=self.device) for _ in range(batch_size)]
        else:
            # Prepare batched tensor for decoder
            # Shape: (batch_size, num_codebooks, num_frames)
            batched_samples = torch.zeros(
                batch_size, 32, max_frames, dtype=torch.long, device=self.device
            )
            sample_lengths = []
            
            for i, samples in enumerate(batch_samples):
                if len(samples) > 0:
                    stacked = torch.stack(samples)  # (num_frames, num_codebooks)
                    batched_samples[i, :, :len(samples)] = stacked.T
                    sample_lengths.append(len(samples))
                else:
                    sample_lengths.append(0)
            
            # Decode entire batch at once
            batched_audio = self._audio_tokenizer.decode(batched_samples)  # (batch_size, 1, audio_samples)
            
            # Process each audio output (trim padding and apply watermark)
            audio_outputs = []
            for i in range(batch_size):
                if sample_lengths[i] == 0:
                    audio = torch.zeros(1, device=self.device)
                else:
                    # Trim to actual length (80ms per frame = sample_rate * 0.08 samples per frame)
                    expected_samples = int(sample_lengths[i] * self.sample_rate * 0.08)
                    audio = batched_audio[i, 0, :expected_samples]
                    
                    # Apply watermark
                    audio, wm_sample_rate = watermark(
                        self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK
                    )
                    audio = torchaudio.functional.resample(
                        audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate
                    )
                
                audio_outputs.append(audio)

        if output_logits:
            # Compile log probability statistics for each sequence
            log_prob_results = []
            for i in range(batch_size):
                total_log_prob = sum(batch_log_probs[i])
                num_frames = len(batch_log_probs[i])
                log_prob_results.append({
                    'total_log_prob': total_log_prob,
                    'avg_log_prob': total_log_prob / max(num_frames, 1),
                    'num_frames': num_frames,
                    'frame_log_probs': batch_log_probs[i]  # Optional: keep per-frame for debugging
                })
            return audio_outputs, log_prob_results
        
        return audio_outputs

    @torch.inference_mode()
    def compute_audio_logprob_batch(
        self,
        target_audios: List[torch.Tensor],
        texts: List[str],
        speakers: List[int] = None,
        contexts: List[List[Segment]] = None,
    ) -> List[dict]:
        """
        Compute the log probability of target audios given text and context (batched).
        Only uses the 0th codebook.
        """
        batch_size = len(target_audios)
        
        if speakers is None:
            speakers = [0] * batch_size
        if contexts is None:
            contexts = [[]] * batch_size
        
        assert batch_size <= self.max_batch_size
        
        # Tokenize all target audios
        all_target_tokens = []
        for audio in target_audios:
            tokens, _ = self._tokenize_audio(audio.to(self.device))
            all_target_tokens.append(tokens)
        
        max_frames = max(tokens.size(0) for tokens in all_target_tokens)
        
        # Prepare the prompts
        prompt_tokens, prompt_tokens_mask, prompt_lengths = self._prepare_batch_prompts(
            texts=texts,
            speakers=speakers,
            contexts=contexts
        )
        
        # Reset caches
        if not self._model.backbone.caches_are_enabled():
            self._model.setup_caches(batch_size)
        self._model.reset_caches()
        
        curr_pos = torch.arange(0, prompt_tokens.size(1), device=self.device).unsqueeze(0).repeat(batch_size, 1)
        
        # Track log probs
        batch_frame_log_probs = [[] for _ in range(batch_size)]
        num_frames_per_item = [tokens.size(0) for tokens in all_target_tokens]
        is_active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        # Process prompt first using generate_frame to get logits
        c0_logits = self._model.generate_frame(
            prompt_tokens, prompt_tokens_mask, curr_pos, 
            temperature=1.0, topk=50, return_logits=False, return_raw_logits=True
        )#[1]  # Returns (samples, logits), we want logits
        
        # Get log probs for first frame
        log_probs = torch.log_softmax(c0_logits, dim=-1)  # (batch_size, vocab_size)
        
        for i in range(batch_size):
            if 0 < num_frames_per_item[i]:
                gt_code = all_target_tokens[i][0, 0].item()  # First frame, 0th codebook
                frame_log_prob = log_probs[i, gt_code].item()
                batch_frame_log_probs[i].append(frame_log_prob)
        
        # Prepare next tokens with ground truth
        next_frame = torch.zeros(batch_size, 1, 33, dtype=torch.long, device=self.device)
        next_frame_mask = torch.zeros(batch_size, 1, 33, dtype=torch.bool, device=self.device)
        
        for i in range(batch_size):
            if 0 < num_frames_per_item[i]:
                next_frame[i, 0, :-1] = all_target_tokens[i][0, :-1]
                next_frame_mask[i, 0, :-1] = True
        
        curr_tokens = next_frame
        curr_tokens_mask = next_frame_mask
        curr_pos = curr_pos[:, -1:] + 1
        
        # Autoregressively process remaining frames
        for frame_idx in range(1, max_frames):
            # Get logits using generate_frame
            c0_logits = self._model.generate_frame(
                curr_tokens, curr_tokens_mask, curr_pos,
                temperature=1.0, topk=50, return_logits=False, return_raw_logits=True
            )#[1]
            
            log_probs = torch.log_softmax(c0_logits, dim=-1)
            
            # Get log prob of ground truth for this frame
            for i in range(batch_size):
                if frame_idx < num_frames_per_item[i]:
                    gt_code = all_target_tokens[i][frame_idx, 0].item()  # 0th codebook
                    frame_log_prob = log_probs[i, gt_code].item()
                    batch_frame_log_probs[i].append(frame_log_prob)
                else:
                    is_active[i] = False
            
            if not is_active.any():
                break
            
            # Prepare next frame with ground truth
            next_frame = torch.zeros(batch_size, 1, 33, dtype=torch.long, device=self.device)
            next_frame_mask = torch.zeros(batch_size, 1, 33, dtype=torch.bool, device=self.device)
            
            for i in range(batch_size):
                if frame_idx < num_frames_per_item[i]:
                    next_frame[i, 0, :-1] = all_target_tokens[i][frame_idx, :-1]
                    next_frame_mask[i, 0, :-1] = True
            
            curr_tokens = next_frame
            curr_tokens_mask = next_frame_mask
            curr_pos = curr_pos[:, -1:] + 1
        
        # Compile results
        results = []
        for i in range(batch_size):
            total_log_prob = sum(batch_frame_log_probs[i])
            results.append({
                'total_log_prob': total_log_prob,
                'avg_log_prob': total_log_prob / max(len(batch_frame_log_probs[i]), 1),
                'num_frames': len(batch_frame_log_probs[i]),
                'frame_log_probs': batch_frame_log_probs[i]
            })
        
        return results

def load_csm_1b(device: str = "cuda", max_batch_size=64):
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)

    generator = Generator(model, max_batch_size=max_batch_size)
    return generator