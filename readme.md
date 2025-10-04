# About this project

this project provide tools

1. locally training audio classify by labels (fine ture wav2vec2)
2. classify audio files
3. locally llm fine tune
4. audio data format
   1. file convert
   2. file split
   3. audio to mel transform
5. audio analysis
   1. multiple speaker split
   2. voice pad score

# demo

## split audio

```py
import config
import utils
import glob

# split details in config.py
# > `max_audio_length_ms`, `format_long_audio_split_name`
# > `raw_audio_path`, `train_format_audio_path`
for i in glob.glob("./public_voice/*"):
    utils.split_long_wav(i)
```

## split multiple speaker

```py
from audio_analysis import multiple_speaker
import config
import utils
import glob
import os

for i in glob.glob(os.path.join(config.train_format_audio_path, "__split*")):
    multiple_speaker.get_speaker_dict(i)
```

## split single speaker and mutilpe

```py
single, overlap = utils.classify_overlap_dicts(dicts)
```

## join all and split

```py
import config
import utils
import glob
import os

single_dicts, overlap_dicts = list(), list()

for i in glob.glob(os.path.join(config.audio_analysis_save_path, "analysis_*split*.json")):
    dicts = utils.load_json(i)
    if len(dicts) > 0:
        single, overlap = utils.classify_overlap_dicts(dicts)
        single_dicts.extend(single), overlap_dicts.extend(overlap)

utils.save_json(single_dicts, os.path.join(config.audio_analysis_save_path, "all_analysis_single.json"))
utils.save_json(overlap_dicts, os.path.join(config.audio_analysis_save_path, "all_analysis_overlap.json"))
```

## split voice clips by dicts

```py
import config
import utils
import os

dicts = utils.load_json(os.path.join(config.audio_analysis_save_path, "all_analysis_single.json"))

utils.split_audio_clips(dicts)
```

## emotion analysis

```py
from audio_analysis import emotion
import config
import utils
import os

dicts = utils.load_json(os.path.join(config.audio_analysis_save_path, "all_analysis_single.json"))

pad_dicts = emotion.get_pad_dicts(dicts)
```


# Project initialises

1. update valuable `label` in `config.py`, you can modify the files default path if you want to.
2. update `llm/llm_fine_tune_data.py` if you wanna fine tune llm parts.

# Default config

## audio parts

1. label audio file path (default) `./classify/(label_name)/*.wav`
2. raw audio path (default) `./raw`
3. pretrain format audio path (default) `./format`
4. mel caches path (default) `./mel`
5. wav2vec2 model path (default) `./model/wav2vec2-large-robust-12-ft-emotion-msp-dim`
6. audio classify model path (default) `./model/classify`
7. not classify audio path (default) `./unlabeled-classify`
8. classify audio output directory `./classify`

# Run script

## audio classify train

```bash
python -m audio_classify.label_train
```

## classify audio

```bash
python -m audio_classify.label <model_path> <glob_wav_string>
```

## llm fine tune

```bash
python -m llm.llm_fine_tune
```
