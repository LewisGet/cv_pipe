import torch
import torchaudio

from pydub import AudioSegment
import numpy as np
import pickle
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

def join_with_fixed_position(audio_paths, split_time = 60 * 1000):
    output_audio = AudioSegment.silent(duration=60 * 60 * 1000)
    data_mapping = []
    join_index = 0

    file_paths = audio_paths

    if isinstance(audio_paths, str):
        file_paths = glob.glob(audio_paths)

    for path in file_paths:
        start_time, end_time, speaker_id, filename = filename_valuable(os.path.basename(path))
        total_time = end_time - start_time
        join_audio = load_audio(path)
        join_time = join_index * split_time
        join_index += 1

        output_audio = output_audio.overlay(join_audio, position=join_time)
        data_mapping.append({
            "name": filename,
            "speaker": speaker_id,
            "total_time": total_time
        })

    return output_audio, data_mapping

"""
單個檔案中，每個說話者個隨機選取檔案
"""
def files_all_speaker_random_audio(filepath, pick_times = 1):
    mapping = statistics_speakers(filepath)
    file_names = mapping.keys()
    base_folder = os.path.dirname(filepath)

    select_paths = []

    for name in file_names:
        for speaker_id in range(mapping[name] + 1):
            select_file_name = "*s-*s-SPEAKER_" + str(speaker_id).zfill(2) + "-" + name
            select_paths.extend(random_file(os.path.join(base_folder, select_file_name), pick_times))

    return select_paths

def convert_audio_format():
    for path in glob.glob(os.path.join(config.raw_audio_path, "*.wav")):
        filename = os.path.basename(path).split(".")[0]
        load_audio(path).export(os.path.join(config.train_format_audio_path, filename + ".wav"), format="wav")

def get_mels(paths):
    vocoder = torch.hub.load('LewisGet/melgan-neurips', 'load_melgan')

    mels = []

    for path in paths:
        _file = open(path, "rb")
        wav, sample_rate = torchaudio.load(path)
        _file.close()

        mel = vocoder(wav)

        if mel.shape[-1] < config.fft_frames:
            continue

        mel_join = np.concatenate(mels, axis=1)
        mel_mean = np.mean(mel_join, axis=1, keepdims=True)
        mel_std = np.std(mel_join, axis=1, keepdims=True) + 1e-9

        mel_normalized = []

        for mel in mels:
            mel_normalized.append((mel - mel_mean) / mel_std)

        return mel_normalized, mel_mean, mel_std

def save_mels(prefix_name, mels, mean, std):
    f = open(os.path.join(config.fft_preprocess_path, prefix_name + "_mels.pkl"), "wb")
    pickle.dump(mels, f)
    f.close()
    np.save(os.path.join(config.fft_preprocess_path, prefix_name + "_mean.npy"), mean)
    np.save(os.path.join(config.fft_preprocess_path, prefix_name + "_std.npy"), std)

def load_mels(prefix_name):
    f = open(os.path.join(config.fft_preprocess_path, prefix_name + "_mels.pkl"), "rb")
    mels = pickle.load(f)
    f.close()
    mean = np.load(os.path.join(config.fft_preprocess_path, prefix_name + "_mean.npy"))
    std = np.load(os.path.join(config.fft_preprocess_path, prefix_name + "_std.npy"))

    return mels, mean, std

def fixed_wav_length(wav, size=config.fft_frames * 250):
    wav = wav[:, :size]
    if wav.shape[1] < size:
        wav = torch.nn.functional.pad(wav, (0, size - wav.shape[1]))

    return wav
