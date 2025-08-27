# =========================
# training.py
# =========================

import time
import pandas as pd
import torch
import yaml

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)

from peft import get_peft_model, LoraConfig

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
from sklearn.metrics import precision_recall_fscore_support


# =========================
# Load config
# =========================
with open("config.yml") as f:
    config = yaml.safe_load(f)

# =========================
# Start timer
# =========================
start_time = time.time()

# =========================
# Load datasets
# =========================
train_df = pd.read_csv(config["paths"]["train_csv"])
val_df = pd.read_csv(config["paths"]["val_csv"])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# =========================
# Model + Tokenizer
# =========================
model_checkpoint = config["model"]["checkpoint"]
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

num_labels = len(train_df["label"].unique())
id2label = {i: str(i) for i in range(num_labels)}
label2id = {v: k for k, v in id2label.items()}

# =========================
# Compute class weights
# =========================
class_weights, classes = get_class_weights(train_df)
print("Class Weights:", class_weights)

# =========================
# Load model
# =========================
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Apply LoRA
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
tokenized_train = train_dataset.map(
    lambda x: tokenize_function(x, tokenizer, config["model"]["max_seq_length"]),
    batched=True
)
tokenized_val = val_dataset.map(
    lambda x: tokenize_function(x, tokenizer, config["model"]["max_seq_length"]),
    batched=True
)

# =========================
# Training args
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
# Callback to store metrics
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
    data_collator=None,  # tokenizer handles padding
    callbacks=[metrics_callback]
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Save LoRA adapter
# =========================
adapter_output_dir = "classification_lora-adapter"
model.save_pretrained(adapter_output_dir)
print(f"\nLoRA adapter saved to: {adapter_output_dir}")

# =========================
# Plot curves
# =========================
plot_training_curves(metrics_callback)

# =========================
# Confusion Matrix
# =========================
predictions = trainer.predict(tokenized_val)
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
plot_confusion_matrix(y_true, y_pred, classes, precision, recall, f1)

# =========================
# Execution time
# =========================
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")
