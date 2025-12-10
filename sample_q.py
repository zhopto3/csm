"""Run sampling from csm."""
import argparse
import os

import torch
import torchaudio

from generator import load_csm_1b

def main(input, batch_size, out_dir_wav, out_dir_probs):
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    generator = load_csm_1b(device, max_batch_size=batch_size)
    sr = generator.sample_rate

    item = input.split('/')[-1].split('.')[0]
    print(item)

    os.makedirs(f'{out_dir_wav}/{item}',exist_ok=True)

    with (open(input, 'r',encoding='utf-8') as pref_in,
        open(f'{out_dir_probs}/{item}.csv', 'w', encoding='utf-8') as prob_out
        ):
        for i, line in enumerate(pref_in):
            #remove new line
            line = line.strip()

            #sample:
            audios, log_probs = generator.generate(texts = [line]*batch_size,max_audio_length_ms=10000, output_logits=True)

            #Save audio
            for j, audio in enumerate(audios):
                samp = f"{i:02d}_n{j:05d}"
                torchaudio.save(f"{out_dir_wav}/{item}/{samp}.wav", audio.unsqueeze(0),sr)

            #save log_probs
            for j, prob in enumerate(log_probs):
                samp = f"{i:02d}_n{j:05d}"
                prob_out.write(f"{samp},{prob['total_log_prob']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_text", required = True)
    parser.add_argument("--batch_size", default = 2, type = int)
    parser.add_argument("--out_wav", default = '/shares/chodroff.linguistics.uzh/zhopto/imp_samp/q_samp_wav')
    parser.add_argument("--out_log_probs", default = '/shares/chodroff.linguistics.uzh/zhopto/imp_samp/q_samp_prob')

    main(parser.parse_args)