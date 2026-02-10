import os
import sys
from pathlib import Path
import torch
import torchaudio
from generator import load_csm_1b, Segment


def get_word_segments(seg_dir, subdir_name, sample_id):
    """
    Load all word segments for a given sample in order.
    Returns list of (word_idx, audio_tensor, word_text) tuples.
    """
    seg_path = Path(seg_dir) / subdir_name
    if not seg_path.exists():
        return []
    
    # Find all files matching this sample_id
    word_files = sorted(seg_path.glob(f"*_{sample_id}.wav"))
    
    segments = []
    for wav_file in word_files:
        # Parse: 00_n00099.wav -> word_idx=0
        basename = wav_file.stem
        word_idx = int(basename.split('_')[0])
        
        # Load audio
        audio, sr = torchaudio.load(str(wav_file))
        audio = audio[0]  # Get first channel
        
        segments.append((word_idx, audio, wav_file.stem))
    
    return segments


def preload_all_samples(seg_dir, subdir_name, sample_ids, num_words):
    """Preload all audio for all samples to avoid repeated I/O"""
    all_samples = {}
    
    for sample_id in sample_ids:
        word_segments = get_word_segments(seg_dir, subdir_name, sample_id)
        if len(word_segments) == num_words:
            all_samples[sample_id] = word_segments
    
    return all_samples


def main():
    MANIFEST_DIR = "../asr-surprisal/src/imp_samp/text_pref"
    SEG_DIR = "/shares/chodroff.linguistics.uzh/zhopto/imp_samp/mfa_seg"
    OUTPUT_DIR = "/shares/chodroff.linguistics.uzh/zhopto/imp_samp/mfa_q_prob"
    BATCH_SIZE = 256  # Adjust based on GPU memory
    
    # Get task ID from environment if running as array job
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Loading CSM model on {device}...")
    generator = load_csm_1b(device, max_batch_size=BATCH_SIZE)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all subdirectories
    all_subdirs = sorted(Path(SEG_DIR).glob("[3-6]*"))
    all_subdirs = [s for s in all_subdirs if s.is_dir()]
    
    # If array job, process only the subdir for this task
    if task_id >= 0:
        if task_id >= len(all_subdirs):
            print(f"Task ID {task_id} out of range")
            return
        subdirs_to_process = [all_subdirs[task_id]]
    else:
        subdirs_to_process = all_subdirs
    
    # Process each subdirectory
    for subdir in subdirs_to_process:
        subdir_name = subdir.name
        
        # Check if already processed
        output_file = Path(OUTPUT_DIR) / f"{subdir_name}.csv"
        if output_file.exists():
            print(f"Skipping {subdir_name} (already processed)")
            continue
        
        print(f"\nProcessing {subdir_name}...")
        
        # Read manifest
        manifest_file = Path(MANIFEST_DIR) / f"{subdir_name}.txt"
        if not manifest_file.exists():
            print(f"  Warning: manifest not found, skipping...")
            continue
        
        with open(manifest_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        sentence = lines[-1]
        text_words = sentence.split()
        num_words = len(text_words)
        
        print(f"  Sentence: {sentence}")
        
        # Find all unique sample IDs
        sample_ids = set()
        for wav_file in subdir.glob("*.wav"):
            sample_id = '_'.join(wav_file.stem.split('_')[1:])
            sample_ids.add(sample_id)
        
        sample_ids = sorted(sample_ids)
        print(f"  Found {len(sample_ids)} samples")
        
        # Preload all audio to avoid repeated I/O
        print(f"  Preloading all audio...")
        all_samples = preload_all_samples(SEG_DIR, subdir_name, sample_ids, num_words)
        valid_sample_ids = list(all_samples.keys())
        print(f"  Valid samples: {len(valid_sample_ids)}")
        
        # Create all jobs upfront (better for batching)
        all_jobs = []
        for word_idx in range(num_words):
            for sample_id in valid_sample_ids:
                word_segments = all_samples[sample_id]
                
                # Build context (all previous words)
                context = []
                for j in range(word_idx):
                    prev_audio = word_segments[j][1]
                    prev_text = text_words[j]
                    context.append((prev_audio, prev_text))
                
                current_audio = word_segments[word_idx][1]
                all_jobs.append((sample_id, word_idx, context, current_audio))
        
        print(f"  Total jobs: {len(all_jobs)}")
        
        # Process all jobs in batches
        results_dict = {}  # (sample_id, word_idx) -> result
        
        for batch_start in range(0, len(all_jobs), BATCH_SIZE):
            batch = all_jobs[batch_start:batch_start + BATCH_SIZE]
            actual_batch_size = len(batch)
            
            # Pad batch to full BATCH_SIZE
            if actual_batch_size < BATCH_SIZE:
                batch = batch + [batch[-1]] * (BATCH_SIZE - actual_batch_size)
            
            # Prepare batch inputs
            target_audios = [job[3] for job in batch]
            texts = [text_words[job[1]] for job in batch]
            contexts = []
            for _, _, ctx, _ in batch:
                segment_list = [
                    Segment(speaker=0, text=txt, audio=aud.to(generator.device))
                    for aud, txt in ctx
                ]
                contexts.append(segment_list)
            
            # Compute log probs
            results = generator.compute_audio_logprob_batch(
                target_audios=target_audios,
                texts=texts,
                contexts=contexts
            )
            
            # Store results (only actual batch, not padding)
            for i in range(actual_batch_size):
                sample_id, word_idx = batch[i][0], batch[i][1]
                results_dict[(sample_id, word_idx)] = results[i]
            
            if (batch_start // BATCH_SIZE) % 10 == 0:
                progress = (batch_start / len(all_jobs)) * 100
                print(f"    Progress: {progress:.1f}%")
        
        # Write results to CSV
        print(f"  Writing results...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("sample_id,word_idx,word_text,total_log_prob,avg_log_prob,num_frames\n")
            
            for sample_id in valid_sample_ids:
                for word_idx in range(num_words):
                    result = results_dict[(sample_id, word_idx)]
                    f.write(
                        f"{sample_id},{word_idx},{text_words[word_idx]},"
                        f"{result['total_log_prob']},{result['avg_log_prob']},{result['num_frames']}\n"
                    )
        
        print(f"  Saved to {output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()