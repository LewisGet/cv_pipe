from pydub import AudioSegment
import numpy as np
import glob
import random
import re
import os

import config


def load_audio(filepath):
    audio = AudioSegment.from_file(filepath)

    audio = audio.set_frame_rate(config.sample_rate)
    audio = audio.set_channels(config.channels)
    audio = audio.set_sample_width(config.sample_width)

    return audio

def random_file(path, times = 1):
    paths = list(glob.glob(path))
    indexes = [random.randint(0, len(paths) - 1) for i in range(times)]

    return [paths[index] for index in indexes]

def filename_valuable(filename):
    file_valuable = re.match(r"([0-9.]+)s-([0-9.]+)s-SPEAKER_([0-9]+)-([a-zA-Z0-9_.]+)", filename).groups()
    start_time = float(file_valuable[0])
    end_time = float(file_valuable[1])
    speaker_id = int(file_valuable[2])
    filename = file_valuable[3]

    return start_time, end_time, speaker_id, filename

"""
統計自動分類，單個檔名中有幾個說話者
"""
def statistics_speakers(filepath):
    name_speakers = {}

    for path in glob.glob(filepath):
        start_time, end_time, speaker_id, filename = filename_valuable(os.path.basename(path))

        if filename not in name_speakers.keys():
            name_speakers[filename] = 0

        if name_speakers[filename] < speaker_id:
            name_speakers[filename] = speaker_id

    return name_speakers

def array_to_audio(array):
    array = (array * 2 ** 15).astype(np.int16)
    audio_bytes = array.tobytes()

    return AudioSegment(
        audio_bytes,
        frame_rate=config.sample_rate,
        sample_width=config.sample_width,
        channels=array.shape[1]
    ).set_channels(config.channels)

def audio_to_array(audio):
    audio_bytes = audio.raw_data
    array = np.frombuffer(audio_bytes, np.int16)
    array = array.reshape((-1), audio.channels)

    array = array / 2 ** 15

    return array


