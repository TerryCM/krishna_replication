from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)
from transformers import AutoTokenizer
import pandas as pd
import torch


def compute_metrics(pred):
    """
    Calculate and return performance metrics for a given set of predictions.

    Args:
        pred (TrainerPredictionOutput): A prediction output object containing label_ids and predictions.

    Returns:
        dict: A dictionary containing the following performance metrics:
            - accuracy (float): The overall accuracy of the model.
            - f1 (float): The weighted F1 score.
            - precision (float): The weighted precision.
            - recall (float): The weighted recall.
            - clf_rep (str): The full classification report as a string.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    clf_rep = classification_report(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "clf_rep": str(clf_rep),
    }


def tokenize_function_q(examples):
    """
    Tokenize the given examples containing answers using a pre-defined tokenizer.

    Args:
        examples (pd.DataFrame): A DataFrame containing a column "answer" with text samples to tokenize.

    Returns:
        transformers.tokenization_utils_base.BatchEncoding: A batch encoding object containing the tokenized
        input sequences (answers), with truncation and padding applied.
    """
    return tokenizer(examples["answer"].tolist(), truncation=True, padding=True)


tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
id2label = {0: "yes", 1: "pyes", 2: "middle", 3: "pno", 4: "no"}
label2id = {"yes": 0, "pyes": 1, "middle": 2, "pno": 3, "no": 4}
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=5, id2label=id2label, label2id=label2id
)


train_data = pd.read_excel("Dataset/Train_data.xlsx")
test_data = pd.read_excel("Dataset/Test_data.xlsx")
val_data = pd.read_excel("Dataset/Val_data.xlsx")

train_labels = train_data["t2_label"].tolist()
test_labels = test_data["t2_label"].tolist()
val_labels = val_data["t2_label"].tolist()


label_dict = {"yes": 0, "pyes": 1, "middle": 2, "pno": 3, "no": 4}
train_labels = [label_dict[item] for item in train_labels]
val_labels = [label_dict[item] for item in val_labels]
test_labels = [label_dict[item] for item in test_labels]


train_encodings = tokenize_function_q(train_data)
val_encodings = tokenize_function_q(val_data)
test_encodings = tokenize_function_q(test_data)


class SWDADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SWDADataset(train_encodings, train_labels)
val_dataset = SWDADataset(val_encodings, val_labels)
test_dataset = SWDADataset(test_encodings, test_labels)


training_args = TrainingArguments(
    output_dir="./RoBERTa_A",  # output directory
    num_train_epochs=32,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,  # batch size for evaluation
    warmup_steps=200,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    evaluation_strategy="epoch",
    save_strategy="epoch",
)


trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.evaluate()
trainer.save_model("RoBERTa_A")
tokenizer.save_pretrained("RoBERTa_A")
trainer.save_metrics("RoBERTa_A", metrics=trainer.evaluate())
y_true = test_labels
y_pred = trainer.predict(test_dataset).predictions.argmax(-1)
print(classification_report(y_true, y_pred))
