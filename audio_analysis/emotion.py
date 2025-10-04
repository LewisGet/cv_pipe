from pyannote.audio import Pipeline
import config
import utils

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

import os


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        #  Arousal, dominance, valence
        return logits

model = EmotionModel.from_pretrained(config.emotion_model_path).to(torch.device("cuda"))

def get_pad_dicts(dicts):
    sources = list()

    for clip_section in dicts:
        filename = os.path.basename(clip_section['path'])
        clip_filename = config.format_clip_name(filename, clip_section['s'], clip_section['e'])
        format_file = os.path.join(config.raw_audio_path, clip_filename)

        with torch.no_grad():
            wav, rate = torchaudio.load(format_file)
            wav = wav.to(torch.device("cuda"))

            _, pad = model(wav)
            _pad = pad.cpu().detach().tolist()

        clip_section['arousal'], clip_section['dominance'], clip_section['valence'] = _pad
        sources.append(clip_section)

        del wav, rate, pad, _pad, _
        torch.cuda.empty_cache()

    utils.save_json(sources, os.path.join(config.audio_analysis_save_path, "pad_source.json"))

    return sources
