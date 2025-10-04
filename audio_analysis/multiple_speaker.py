from pyannote.audio import Pipeline
import config
import utils
import torch
import torchaudio
import os

from pyannote.core import Annotation
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization


if config.multiple_speaker_diarization_model.startswith("pyannote"):
    pipeline = Pipeline.from_pretrained(config.multiple_speaker_diarization_model, use_auth_token=config.hugging_face_token)
else:
    pipeline = Pipeline.from_pretrained(config.multiple_speaker_diarization_model)

pipeline = pipeline.to(torch.device("cuda"))

def get_speaker_dict(path, return_without_save=False):
    filename = os.path.basename(path)
    waveform, sample_rate = torchaudio.load(path)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    output = diarization

    if type(diarization) == Annotation:
        output = diarization

    if type(diarization) == SpeakerDiarization:
        output = diarization.speaker_diarization

    split_datas = []
    for turn, _, speaker in output.itertracks(yield_label=True):
        _start_time = int(turn.start * 1000)
        _stop_time = int(turn.end * 1000)

        if _stop_time > (_start_time + config.min_audio_length_ms):
            split_datas.append({"s": _start_time, "e": _stop_time, "path": path})

    if not return_without_save:
        utils.save_json(split_datas, os.path.join(config.audio_analysis_save_path, f"analysis_{filename}_speaker.json"))

    return split_datas
