import glob, os, torch, wandb
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from data_utils import create_id_label_conversion
from datasets import Dataset, features
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from data_utils import preprocess
import evaluate

metric = evaluate.load("seqeval")
text_clean = preprocess()

tqdm.pandas()
load_dotenv()

def get_last_checkpoint(output_dir):
    checkpoints = []
    for folder in os.listdir(output_dir):
        if folder.startswith('checkpoint'):
            checkpoints.append(int(folder.split('-')[1]))
    try:
        last_checkpoint = f'checkpoint-{max(checkpoints)}'
    except:
        raise ValueError(f"The output directory '{CONFIG.output_dir}' doesn't have any checkpoints. Please set 'run_checkpoint' to False for training to start from scratch.")
    return last_checkpoint

class CONFIG:
    run_checkpoint = True
    output_dir = "Models/Cleaned_Data_DebertaPII"
    checkpoint_dir = os.path.join(output_dir, get_last_checkpoint(output_dir)) if run_checkpoint else 'null'
    model_path = 'lakshyakh93/deberta_finetuned_pii' #"microsoft/deberta-v3-base"
    max_length = 1024
    data_path = 'External Data'
    num_proc = 10
    learning_rate = 1e-6
    num_epochs = 20
    train_batch_size = 2
    eval_batch_size = 2
    grad_accu = 4
    log_steps = 500
    scheduler = 'cosine'
    hf_repo = 'kabir5297/DebertaPII'
    token = os.getenv('HF_TOKEN')

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="PII_Training"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="false"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

# Load Data
json_files = glob.glob(CONFIG.data_path+'/*.json')
data = pd.DataFrame()
for file in json_files:
    data = pd.concat([data, pd.read_json(file)[['tokens','trailing_whitespace','labels']]], ignore_index=True)

# Filter out data containing no positive labels
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
        text.append(text_clean.clean_data(text=t))
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

# Printing Training Information
print('+'*90+'\n')
print(f'Starting Training....')
print(f"Train Data Samples: {len(dataset['train'])}, Test Data Samples: {len(dataset['test'])}.")
if CONFIG.run_checkpoint:
    print(f'Training Resuming from "{CONFIG.checkpoint_dir}"')
print(f'Training will run for {CONFIG.num_epochs} epochs.')
print('\n'+'+'*90)

# Training Preparation
model = AutoModelForTokenClassification.from_pretrained(
    CONFIG.model_path if not CONFIG.run_checkpoint else CONFIG.checkpoint_dir,
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
    report_to="wandb",
    evaluation_strategy="steps",
    save_total_limit=3,
    logging_steps=CONFIG.log_steps,
    lr_scheduler_type=CONFIG.scheduler,
    # metric_for_best_model="loss",
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    dataloader_pin_memory=False,
)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    precision = all_metrics['overall_precision']
    recall = all_metrics['overall_recall']
    return {'f1': ((1 + 5*5) * recall * precision / (5*5*precision + recall))}

# Rewrite compute loss for weighted loss in Trainer class
class Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight =torch.from_numpy(np.array([0.005,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).astype(np.float32)).cuda()
        
    def compute_loss(self, model, inputs,return_outputs=False):
        outputs = model(**inputs)
        loss=F.cross_entropy(outputs.logits.flatten(0,1), inputs.labels.flatten(), weight=self.weight, ignore_index=-100,reduction='mean', label_smoothing=0.0)
        return (loss, outputs) if return_outputs else loss

# Trainer Initialize
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=collator, 
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=CONFIG.run_checkpoint)

# Push to HF
trainer.model.push_to_hub(CONFIG.hf_repo, token = CONFIG.token)
tokenizer.push_to_hub(CONFIG.hf_repo, token = CONFIG.token)
