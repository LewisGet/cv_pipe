import torch
import torchaudio

import utils
import config
import glob
import os


label_data_paths = dict()

for i, v in enumerate(config.labels):
    label_path = config.create_directory(config.label_audio_classify_path, v)
    label_data_paths[v] = list(glob.glob(os.path.join(label_path, "*.wav")))


class TrainDataset:
    train_data = []
    feature_extractor = None
    id2label = config.id2label
    label2id = config.label2id

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def add(self, paths, label):
        for path in paths:
            wav, sample_rate = torchaudio.load(path)

            wav = utils.fixed_wav_length(wav, 16000)

            input_values = self.feature_extractor(
                wav[0],
                sampling_rate=sample_rate,
                max_length=16000,
                truncation=True
            )

            # output batch to one
            input_values['input_values'] = input_values['input_values'][0]
            input_values['attention_mask'] = input_values['attention_mask'][0]
            input_values['label'] = self.label2id[label]

            self.train_data.append(input_values)

    def load_label_paths(self, dict_paths):
        for label, paths in enumerate(dict_paths):
            self.add(paths, label)
