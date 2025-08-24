# =========================
# Core imports
# =========================
import time
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from peft import get_peft_model, LoraConfig

from datasets import Dataset

# =========================
# Start timer
# =========================
start_time = time.time()

# =========================
# Load datasets
# =========================
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# =========================
# Model and tokenizer setup
# =========================
model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

num_labels = 2
id2label = {0: 0, 1: 1}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# =========================
# Tokenization
# =========================
def tokenize_function(examples):
    return tokenizer(
        examples["text"],  # updated column name from previous code
        truncation=True,
        padding='max_length',
        max_length=512
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# =========================
# LoRA configuration
# =========================
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=['q_lin']
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# =========================
# Training arguments
# =========================
training_args = TrainingArguments(
    output_dir=model_checkpoint + "-lora-text-classification",
    learning_rate=1e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    eval_steps=500
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Print execution time
# =========================
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")
