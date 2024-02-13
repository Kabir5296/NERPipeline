import glob, os
from functools import partial
import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from data_utils import create_id_label_conversion
from datasets import Dataset, features
import numpy as np
from tqdm import tqdm
from seqeval.metrics import recall_score, precision_score
from dotenv import load_dotenv

tqdm.pandas()
load_dotenv()
class CONFIG:
    model_path = "microsoft/deberta-v3-base"
    max_length = 1024
    output_dir = "Models"
    data_path = 'External Data'
    num_proc = 10
    learning_rate = 2e-5
    num_epochs = 5
    train_batch_size = 2
    eval_batch_size = 2
    grad_accu = 4
    log_steps = 500
    scheduler = 'cosine'
    hf_repo = 'kabir5297/Deberta_Huge_data'
    token = os.getenv('HF_TOKEN')

# Load Data
json_files = glob.glob(CONFIG.data_path+'/*.json')
data = pd.DataFrame()
for file in json_files:
    data = pd.concat([data, pd.read_json(file)[['tokens','trailing_whitespace','labels']]], ignore_index=True)
    
data['check_data'] = data.labels.progress_apply(lambda x: len(pd.Series(x).unique().tolist()))
data = data[data.check_data > 1]

# Calculate id2label and label2id using unique labels from dataframe
unique_labels = pd.DataFrame.explode(data,column='labels').labels.unique()
id2label, label2id, num_labels = create_id_label_conversion(unique_labels)

# Load Tokenizer and Create HF Dataset :) [I don't like working with HF Dataset but don't want to write the entire function here again]
tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)
dataset = Dataset.from_pandas(data, preserve_index=False)

def tokenize(example, tokenizer, label2id, max_length):
    # rebuild text from tokens
    text = []
    labels = []

    for t, l, ws in zip(example["tokens"], example["labels"], example["trailing_whitespace"]):
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

# Apply Tokenization and Create Train Eval Set
dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": CONFIG.max_length}, num_proc=CONFIG.num_proc)
dataset = dataset.train_test_split(test_size=0.2, seed=42) 

# Training Preparation
model = AutoModelForTokenClassification.from_pretrained(
    CONFIG.model_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

args = TrainingArguments(
    output_dir=CONFIG.output_dir, 
    fp16=True,
    learning_rate=CONFIG.learning_rate,
    num_train_epochs=CONFIG.num_epochs,
    per_device_train_batch_size=CONFIG.train_batch_size,
    per_device_eval_batch_size=CONFIG.eval_batch_size,
    gradient_accumulation_steps=CONFIG.grad_accu,
    report_to="none",
    evaluation_strategy="steps",
    save_total_limit=3,
    logging_steps=CONFIG.log_steps,
    lr_scheduler_type=CONFIG.scheduler,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    dataloader_pin_memory=False,
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
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

# Trainer Initialize
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=collator, 
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics,
)

trainer.train()

# Push to HF
trainer.model.push_to_hub(CONFIG.hf_repo, token = CONFIG.token)
tokenizer.push_to_hub(CONFIG.hf_repo, token = CONFIG.token)