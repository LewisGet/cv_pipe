import os

sample_rate = 16000
channels = 1
sample_width = 2
audio_float_size = 8 * sample_width
#todo fft
hugging_face_token = ""
weight_and_bias_api = ""
llm_cache = os.path.join(".", "llm_cache")
fine_tune_path = os.path.join(llm_cache, "fine-AWQ-INT4")
