# =========================
# Core imports
# =========================
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
import yaml

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig
from datasets import Dataset

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
classes = np.unique(train_df["label"].values)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(classes),
    y=train_df["label"].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
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
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=config["model"]["max_seq_length"]
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# Metrics
# =========================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# =========================
# Custom Trainer with Weighted Loss
# =========================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

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
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.eval_accuracy = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_loss.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_loss.append(logs["eval_loss"])
        if "eval_accuracy" in logs:
            self.eval_accuracy.append(logs["eval_accuracy"])

metrics_callback = MetricsCallback()

# =========================
# Trainer
# =========================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    callbacks=[metrics_callback]
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Plot curves
# =========================
def plot_training_curves(callback):
    epochs = list(range(1, len(callback.eval_loss) + 1))

    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(x=list(range(1, len(callback.train_loss) + 1)), y=callback.train_loss, mode="lines+markers", name="Train Loss")
    )
    fig_loss.add_trace(
        go.Scatter(x=epochs, y=callback.eval_loss, mode="lines+markers", name="Validation Loss")
    )
    fig_loss.update_layout(title="Training vs Validation Loss", xaxis_title="Epochs", yaxis_title="Loss")
    fig_loss.show()

    fig_acc = go.Figure()
    fig_acc.add_trace(
        go.Scatter(x=epochs, y=callback.eval_accuracy, mode="lines+markers", name="Validation Accuracy")
    )
    fig_acc.update_layout(title="Validation Accuracy per Epoch", xaxis_title="Epochs", yaxis_title="Accuracy")
    fig_acc.show()

plot_training_curves(metrics_callback)

# =========================
# Confusion Matrix
# =========================
predictions = trainer.predict(tokenized_val)
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

cm = confusion_matrix(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

fig_cm = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale="Blues",
    labels=dict(x="Predicted", y="True", color="Count"),
    x=list(classes),
    y=list(classes),
)
fig_cm.update_layout(title=f"Confusion Matrix<br>Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
fig_cm.show()

# =========================
# Execution time
# =========================
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")
