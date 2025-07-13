# About this project

this project provide tools

1. locally training audio classify by labels (fine ture wav2vec2)
2. classify audio files
3. locally llm fine tune
4. audio data format
   1. file convert
   2. file split
   3. audio to mel transform

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
