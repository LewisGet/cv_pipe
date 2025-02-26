import os


def create_directory(*path):
    _path = os.path.join(*path)

    if not os.path.exists(_path):
        os.makedirs(_path)

    return _path

sample_rate = 16000
channels = 1
sample_width = 2
audio_float_size = 8 * sample_width

raw_audio_path = create_directory(".", "raw")
train_format_audio_path = create_directory(".", "format")
fft_preprocess_path = create_directory(".", "mel")
fft_frames = 64

hugging_face_token = ""
weight_and_bias_api = ""
llm_cache = create_directory(".", "llm_cache")
fine_tune_path = create_directory(llm_cache, "fine-AWQ-INT4")

auto_audio_classify_path = create_directory(".", "unlabeled-classify")
label_audio_classify_path = create_directory(".", "classify")
label_audio_classify_model = create_directory(".", "model", "classify")

wav2vec2_model_path = create_directory(".", "model", "wav2vec2-large-robust-12-ft-emotion-msp-dim")

labels = ['lewis', 'kevin', 'kit', 'gold', 'else', 'mixed', 'noise']

id2label = {i: v for i, v in enumerate(labels)}
label2id = {v: i for i, v in enumerate(labels)}
