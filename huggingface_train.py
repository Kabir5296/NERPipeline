import json
import argparse
from itertools import chain
from functools import partial

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
import evaluate
from datasets import Dataset, features
import numpy as np

class CONFIG:
    model_path = "microsoft/deberta-v3-base"
    max_length = 1024
    output_dir = "Models"
    data_path = 'External Data'


data = json.load(open("External Data/moredata_dataset_fixed.json"))

# downsampling of negative examples
p=[] # positive samples (contain relevant labels)
n=[] # negative samples (presumably contain entities that are possibly wrongly classified as entity)
for d in data:
    if any(np.array(d["labels"]) != "O"): p.append(d)
    else: n.append(d)
print("original datapoints: ", len(data))

external = json.load(open("External Data/all_labels.json"))

moredata = json.load(open("External Data/mixtral-8x7b-v1.json"))

moredata_1 = json.load(open("External Data/pii_dataset_fixed.json"))

# moredata_2 = json.load(open("External Data/pii_Extended.json"))

moredata_3 = json.load(open("DATA/train.json"))

data = moredata_3+moredata_1+moredata+external+p+n[:len(n)//3]
print("combined: ", len(data))

all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v:k for k,v in label2id.items()}

target = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
    'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
    'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
]

print(id2label)

def tokenize(example, tokenizer, label2id, max_length):

    # rebuild text from tokens
    text = []
    labels = []

    for t, l, ws in zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")

    # actual tokenization
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {**tokenized, "labels": token_labels, "length": length}

tokenizer = AutoTokenizer.from_pretrained(model_path)

ds = Dataset.from_dict({
    # "full_text": [x["full_text"] for x in data],
    # "document": [str(x["document"]) for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    "provided_labels": [x["labels"] for x in data],
})
ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": max_length}, num_proc=10)
# ds = ds.class_encode_column("group")

x = ds[0]

for t,l in zip(x["tokens"], x["provided_labels"]):
    if l != "O":
        print((t,l))

print("*"*100)

for t, l in zip(tokenizer.convert_ids_to_tokens(x["input_ids"]), x["labels"]):
    if id2label[l] != "O":
        print((t,id2label[l]))
        
from seqeval.metrics import recall_score, precision_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f5_score = (1 + 5*5) * recall * precision / (5*5*precision + recall)
    
    results = {
        'recall': recall,
        'precision': precision,
        'f1': f5_score
    }
    return results['f1']

model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

# I decided to uses no eval
train_ds = ds.train_test_split(test_size=0.2, seed=42) # cannot use stratify_by_column='group'
# final_ds

# I actually chose to not use any validation set. This is only for the model I use for submission.
args = TrainingArguments(
    output_dir=output_dir, 
    fp16=True,
    learning_rate=2e-5,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    report_to="none",
    evaluation_strategy="no",
    do_eval=False,
    save_total_limit=1,
    logging_steps=500,
    lr_scheduler_type='cosine',
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=train_ds['train'],
    eval_dataset=train_ds['test'],
    data_collator=collator, 
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.model.push_to_hub("kabir5297/deberta3base_1024_v2", token = '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')
tokenizer.push_to_hub("kabir5297/deberta3base_1024_v2", token = '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')