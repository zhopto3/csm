"""Run sampling from csm with efficient batching."""
import argparse
import os
import torch
import torchaudio
from generator import load_csm_1b


def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def main(input_file, batch_size, samples_per_text, out_dir_wav, out_dir_probs):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load generator with max batch size
    generator = load_csm_1b(device, max_batch_size=batch_size)
    sr = generator.sample_rate
    resampler = torchaudio.transforms.Resample(
    orig_freq=sr, 
    new_freq=16000
        ).to(device)
    
    item = input_file.split('/')[-1].split('.')[0]
    print(f"Processing: {item}")
    
    os.makedirs(f'{out_dir_wav}/{item}', exist_ok=True)
    
    # Read all lines first
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Total lines to process: {len(lines)}")
    print(f"Samples per text: {samples_per_text}")
    print(f"Total audio files to generate: {len(lines) * samples_per_text}")
    
    # Prepare all jobs (line_idx, sample_idx, text)
    jobs = []
    for line_idx, text in enumerate(lines):
        for sample_idx in range(samples_per_text):
            jobs.append((line_idx, sample_idx, text))
    
    print(f"Processing in batches of {batch_size}")
    
    with open(f'{out_dir_probs}/{item}.csv', 'w', encoding='utf-8') as prob_out:
        
        # Process in batches
        for batch_idx, job_batch in enumerate(chunk_list(jobs, batch_size)):
            batch_texts = [job[2] for job in job_batch]
            actual_batch_size = len(batch_texts)
            
            # Pad batch if needed
            if actual_batch_size < batch_size:
                print(f"Last batch: padding {actual_batch_size} -> {batch_size}")
                batch_texts = batch_texts + [batch_texts[0]] * (batch_size - actual_batch_size)
            
            # Generate audio
            audios, log_probs = generator.generate_batch(
                texts=batch_texts,
                max_audio_length_ms=10000,
                output_logits=True
            )
            
            # Save only the actual (non-padded) results
            for idx in range(actual_batch_size):
                line_idx, sample_idx, text = job_batch[idx]
                audio = audios[idx]
                prob = log_probs[idx]
                
                # Create sample name
                samp = f"{line_idx:02d}_n{sample_idx:05d}"
                
                # resamp and save audio
                audio_16k = resampler(audio.unsqueeze(0)).squeeze(0).cpu()
                torchaudio.save(
                    f"{out_dir_wav}/{item}/{samp}.wav",
                    audio_16k.unsqueeze(0),
                    16000
                )
                
                # Save log probs
                prob_out.write(
                    f"{samp},{prob['total_log_prob']},{prob['avg_log_prob']}\n"
                )
            
            if (batch_idx + 1) % 10 == 0:
                progress = ((batch_idx + 1) * batch_size) / len(jobs) * 100
                print(f"Processed {batch_idx + 1} batches ({progress:.1f}%)")
    
    print(f"Done---Generated {len(jobs)} audio files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_text", required=True, help="Input text file with one prompt per line")
    parser.add_argument("--batch_size", default=256, type=int, help="Model batch size")
    parser.add_argument("--samples_per_text", default=2, type=int, help="Number of audio samples to generate per text line")
    parser.add_argument("--out_wav", type=str, default='/shares/chodroff.linguistics.uzh/zhopto/imp_samp/q_samp_wav')
    parser.add_argument("--out_log_probs", type=str, default='/shares/chodroff.linguistics.uzh/zhopto/imp_samp/q_samp_prob')
    
    args = parser.parse_args()
    
    main(args.in_text, args.batch_size, args.samples_per_text, args.out_wav, args.out_log_probs)