import json 
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import(
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import transformers

# Check transformers version
print(f"Transformers version: {transformers.__version__}")

# Store config as dict first
config = {
    "model": "bert-base-uncased",
    "epochs": 3,
    "batch_size": 8,
    "max_length": 64,
    "learning_rate": 5e-5,
}

wandb.init(
    project="intent-classifier-bert",
    config=config,
)

with open("data/augmented_data.json") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
labels = [item["intent"] for item in data]

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=config["max_length"])

train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels}).map(tokenize, batched=True)
test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_labels}).map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = BertForSequenceClassification.from_pretrained(
    config["model"],
    num_labels=len(label_encoder.classes_)
)

training_args = TrainingArguments(
    output_dir="./bert_intent_model",
    eval_strategy="epoch",  # Modern parameter name
    save_strategy="epoch",
    learning_rate=config["learning_rate"],
    num_train_epochs=config["epochs"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    logging_dir="./logs",
    logging_steps=10,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    metrics = {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted", zero_division=0),
        "precision": precision_score(p.label_ids, preds, average="weighted", zero_division=0),
        "recall": recall_score(p.label_ids, preds, average="weighted", zero_division=0),
    }
    return metrics

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate on test set
results = trainer.evaluate()
print(f"\n📊 Test Results: {results}")

# Save model and encoder
model.save_pretrained("./bert_intent_model")
tokenizer.save_pretrained("./bert_intent_model")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Training complete! Model saved in ./bert_intent_model/")