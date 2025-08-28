# =========================
# training.py
# =========================

import time
import pandas as pd
import torch
import yaml
import json

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from peft import get_peft_model, LoraConfig
from sklearn.metrics import precision_recall_fscore_support

# Local imports
from utils import (
    tokenize_function,
    get_class_weights,
    compute_metrics,
    WeightedTrainer,
    MetricsCallback,
    plot_training_curves,
    plot_confusion_matrix,
)

# =========================
# Load config and label mapping
# =========================
with open("config.yml") as f:
    config = yaml.safe_load(f)

with open("label_mapping.json") as f:
    label_mapping = json.load(f)

label2id = {k: int(v) for k, v in label_mapping["label2id"].items()}
id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
num_labels = len(label2id)

print("Label2ID:", label2id)
print("ID2Label:", id2label)

# =========================
# Load datasets
# =========================
train_df = pd.read_csv(config["paths"]["train_csv"])
val_df = pd.read_csv(config["paths"]["val_csv"])

# Ensure labels are consistent with label_mapping
if train_df["label"].dtype == object:  # String labels → convert to numeric IDs
    print("Detected string labels. Converting to numeric IDs using label_mapping...")
    train_df["label"] = train_df["label"].map(label2id)
    val_df["label"] = val_df["label"].map(label2id)
else:  # Already numeric labels
    print("Detected numeric labels. Ensuring mapping consistency...")
    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    # Validate labels
    invalid_labels = set(train_df["label"].unique()) - set(id2label.keys())
    if invalid_labels:
        raise ValueError(f"Unexpected labels found in train set: {invalid_labels}")
    invalid_labels_val = set(val_df["label"].unique()) - set(id2label.keys())
    if invalid_labels_val:
        raise ValueError(f"Unexpected labels found in val set: {invalid_labels_val}")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# =========================
# Compute class weights
# =========================
class_weights, classes = get_class_weights(train_df)
print("Class Weights:", class_weights)

# =========================
# Load base model
# =========================
model_checkpoint = config["model"]["checkpoint"]
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# =========================
# Apply LoRA
# =========================
peft_config = LoraConfig(
    task_type=config["lora"]["task_type"],
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],
    target_modules=config["lora"]["target_modules"]
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# =========================
# Tokenization
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenized_train = train_dataset.map(
    lambda x: tokenize_function(x, tokenizer, config["model"]["max_seq_length"]),
    batched=True
)
tokenized_val = val_dataset.map(
    lambda x: tokenize_function(x, tokenizer, config["model"]["max_seq_length"]),
    batched=True
)

# =========================
# Training arguments
# =========================
training_args = TrainingArguments(
    output_dir=config["paths"]["output_dir"],
    learning_rate=config["training"]["learning_rate"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"],
    weight_decay=config["training"]["weight_decay"],
    do_train=True,
    do_eval=True,
    evaluation_strategy=config["training"]["evaluation_strategy"],
    logging_dir=config["paths"]["logging_dir"],
    logging_steps=config["training"]["logging_steps"],
    save_strategy=config["training"]["save_strategy"]
)

# =========================
# Metrics callback
# =========================
metrics_callback = MetricsCallback()

# =========================
# Trainer
# =========================
trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=None,
    callbacks=[metrics_callback]
)

# =========================
# Train
# =========================
start_time = time.time()
trainer.train()
end_train_time = time.time()
print(f"Training completed in {end_train_time - start_time:.2f} seconds")

# =========================
# Save LoRA adapter
# =========================
adapter_output_dir = "classification_lora-adapter"
model.save_pretrained(adapter_output_dir)
print(f"LoRA adapter saved to: {adapter_output_dir}")

# =========================
# Plot training curves
# =========================
plot_training_curves(metrics_callback)

# =========================
# Evaluate and plot confusion matrix
# =========================
predictions = trainer.predict(tokenized_val)
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

# Convert numeric IDs → original string labels
y_true_labels = [id2label[i] for i in y_true]
y_pred_labels = [id2label[i] for i in y_pred]

# Dynamically choose averaging
if num_labels == 2:
    avg_type = "binary"
else:
    avg_type = "weighted"

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average=avg_type
)

plot_confusion_matrix(
    y_true_labels,
    y_pred_labels,
    class_names=list(id2label.values()),
    precision=precision,
    recall=recall,
    f1=f1
)

# =========================
# Execution time
# =========================
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
