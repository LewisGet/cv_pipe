from transformers import AutoModelForAudioClassification
import torch
import torchaudio

import config

import glob
import sys
import os


def main(w2v2_model_path, audio_paths):
    model = AutoModelForAudioClassification.from_pretrained(
        w2v2_model_path,
        weights_only=False,
        num_labels=len(config.labels), label2id=config.label2id, id2label=config.id2label
    )

    classified = {i: [] for i in config.labels}
    classified['error'] = []
    classified['small'] = []
    classified['oversize'] = []

    for path in audio_paths:
        file_size = os.path.getsize(path)

        if file_size > (config.sample_width * config.sample_rate * 60 * 5):
            classified['oversize'].append(path)
            continue

        wav, sample_rate = torchaudio.load(path)

        if len(wav[0]) < 160:
            classified['small'].append(path)
            continue

        try:
            logits = model(wav).logits
        except:
            classified['error'].append(path)
            continue

        target = model.config.id2label[torch.argmax(logits).item()]
        classified[target].append(path)

        return classified


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m audio_classify.label <model_path> <glob_wav_string>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_paths = glob.glob(sys.argv[2])
    print(main(model_path, audio_paths))
