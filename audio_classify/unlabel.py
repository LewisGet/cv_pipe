import torch
import torchaudio
from pyannote.audio import Pipeline

import utils
import config

import glob
import os


pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=config.hugging_face_token)
pipeline.to(torch.device("cuda"))

for path in glob.glob(os.path.join(config.train_format_audio_path, "*.wav")):
    file_name = os.path.basename(path).split(".")[0]

    waveform, sample_rate = torchaudio.load(path)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    audio = utils.load_audio(path)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = int(turn.start * 1000)
        stop_time = int(turn.end * 1000)

        segment = audio[start_time:stop_time]

        output_file = os.path.join(config.auto_audio_classify_path, f"{turn.start:.1f}s-{turn.end:.1f}s-{speaker}-{file_name}.wav")

        if len(segment) > 50:
            segment.export(output_file, format="wav")
