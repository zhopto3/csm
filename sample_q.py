"""Run sampling from csm with efficient batching and resume capability."""
import argparse
import os
import torch
import torchaudio
from generator import load_csm_1b


def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def get_existing_samples(out_dir_wav, item):
    """Check which samples already exist."""
    existing = set()
    item_dir = f'{out_dir_wav}/{item}'
    
    if not os.path.exists(item_dir):
        return existing
    
    for filename in os.listdir(item_dir):
        if filename.endswith('.wav'):
            # Parse format: "00_n00000.wav" -> (0, 0)
            basename = filename.replace('.wav', '')
            parts = basename.split('_n')
            if len(parts) == 2:
                try:
                    line_idx = int(parts[0])
                    sample_idx = int(parts[1])
                    existing.add((line_idx, sample_idx))
                except ValueError:
                    continue
    
    return existing


def main(input_file, batch_size, samples_per_text, out_dir_wav, out_dir_probs):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    item = input_file.split('/')[-1].split('.')[0]
    print(f"Processing: {item}")
    
    # Read all lines first
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Check existing samples
    existing_samples = get_existing_samples(out_dir_wav, item)
    print(f"Found {len(existing_samples)} existing samples")
    
    # Prepare jobs only for missing samples
    jobs = []
    for line_idx, text in enumerate(lines):
        for sample_idx in range(samples_per_text):
            if (line_idx, sample_idx) not in existing_samples:
                jobs.append((line_idx, sample_idx, text))
    
    if not jobs:
        print(f"All {len(lines) * samples_per_text} samples already exist. Skipping.")
        return
    
    print(f"Total lines: {len(lines)}")
    print(f"Samples per text: {samples_per_text}")
    print(f"Total expected: {len(lines) * samples_per_text}")
    print(f"Missing samples to generate: {len(jobs)}")
    
    # Load generator only if needed
    generator = load_csm_1b(device, max_batch_size=batch_size)
    sr = generator.sample_rate
    resampler = torchaudio.transforms.Resample(
        orig_freq=sr, 
        new_freq=16000
    ).to(device)
    
    os.makedirs(f'{out_dir_wav}/{item}', exist_ok=True)
    os.makedirs(out_dir_probs, exist_ok=True)
    
    print(f"Processing in batches of {batch_size}")
    
    # Append to existing prob file
    with open(f'{out_dir_probs}/{item}.csv', 'a', encoding='utf-8') as prob_out:
        
        # Process in batches
        for batch_idx, job_batch in enumerate(chunk_list(jobs, batch_size)):
            batch_texts = [job[2] for job in job_batch]
            actual_batch_size = len(batch_texts)
            
            # Pad batch if needed
            if actual_batch_size < batch_size:
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
    
    print(f"Done---Generated {len(jobs)} new audio files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_text", required=True, help="Input text file with one prompt per line")
    parser.add_argument("--batch_size", default=256, type=int, help="Model batch size")
    parser.add_argument("--samples_per_text", default=2, type=int, help="Number of audio samples to generate per text line")
    parser.add_argument("--out_wav", type=str, default='/shares/chodroff.linguistics.uzh/zhopto/imp_samp/q_samp_wav')
    parser.add_argument("--out_log_probs", type=str, default='/shares/chodroff.linguistics.uzh/zhopto/imp_samp/q_samp_prob')
    
    args = parser.parse_args()
    
    main(args.in_text, args.batch_size, args.samples_per_text, args.out_wav, args.out_log_probs)