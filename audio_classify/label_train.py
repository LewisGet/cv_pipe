from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, Audio
import evaluate
import numpy as np

import config
from . import label_train_dataset

import os

os.environ["WANDB_DISABLED"] = "true"

num_labels = len(config.id2label)
model = AutoModelForAudioClassification.from_pretrained(
    config.wav2vec2_model_path,
    num_labels=num_labels, label2id=config.label2id, id2label=config.id2label,
    weights_only=False
)
feature_extractor = AutoFeatureExtractor.from_pretrained(config.wav2vec2_model_path)

train_dataset_loader = label_train_dataset.TrainDataset(feature_extractor)
train_dataset_loader.load_label_paths(label_train_dataset.label_data_paths)

dataset = Dataset.from_list(train_dataset_loader.train_data)

train_test_split = dataset.train_test_split(test_size=0.2)

dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

dataset_dict.set_format(type="torch", columns=["input_values", "label"])

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

training_args = TrainingArguments(
    output_dir = config.label_audio_classify_model,
    save_total_limit=3,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 3e-5,
    per_device_train_batch_size = 32,
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size = 32,
    num_train_epochs = 100,
    warmup_ratio = 0.1,
    logging_steps = 10,
    load_best_model_at_end = True,
    metric_for_best_model = "accuracy",
    remove_unused_columns=True,
    push_to_hub = False,
    report_to="none"
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset_dict["train"],
    eval_dataset = dataset_dict["test"],
    processing_class = feature_extractor,
    compute_metrics = compute_metrics,
)

trainer.train()
