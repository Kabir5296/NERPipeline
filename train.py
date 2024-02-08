import pandas as pd
import json, torch, gc
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader
from data_utils import create_id_label_conversion, CustomDataCollator, NERDataset
from torch.optim import AdamW, lr_scheduler
from train_utils import NERTrainer

class CONFIG:
    train_debug = False
    seeds = [0, 42, 43, 50] # Random seeds for each fold
    dataset_path = ['DATA/processed_train.json','External Data/mixtral-8x7b-v1.json', 'External Data/pii_Extended.json']
    model_path = 'dslim/bert-base-NER'
    stopword_dir = 'json_folder/stopwords.json'
    train_batch_size = 12
    valid_batch_size = 12
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.000001
    weight_decay = 0.1
    output_dir = 'Test' if train_debug else 'Train_Trial_3'
    num_epochs = 3 if train_debug else 100
    T_max = 500
    min_lr = learning_rate
    max_length = 128 if train_debug else 512
    patience = 9
    
# Load NLTK Stopwords for English
with open(CONFIG.stopword_dir,'r') as f:
    stopwords = json.load(f)['english']

# Load Data from CSV
data = pd.DataFrame()
for data_path in CONFIG.dataset_path:
    data = pd.concat([data, pd.DataFrame({'tokens':pd.read_json(data_path)['tokens'].tolist(), 'labels': pd.read_json(data_path)['labels'].tolist()})])

# Calculate id2label and label2id using unique labels from dataframe
unique_labels = pd.DataFrame.explode(data,column='labels').labels.unique()
id2label, label2id, num_labels = create_id_label_conversion(unique_labels)

for fold, seed in enumerate(CONFIG.seeds):
    print('\n'+'+'*150)
    print(f'Training Starting for Fold {fold}')
    # Split and Fold
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=seed)
    train_data.reset_index(drop=True,inplace=True)
    valid_data.reset_index(drop=True,inplace=True)

    # Define and initialize necessary modules
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)    
    data_collator_fn = CustomDataCollator(tokenizer=tokenizer, device=CONFIG.device)
    model = BertForTokenClassification.from_pretrained(pretrained_model_name_or_path=CONFIG.model_path,
                                    id2label = id2label,
                                    label2id = label2id,
                                    num_labels = num_labels,
                                    ignore_mismatched_sizes = True
                                    ).to(CONFIG.device)

    # Initialize Dataset and DataLoader
    train_dataset = NERDataset(train_data, label2id=label2id, max_length=CONFIG.max_length, tokenizer=tokenizer, stopwords=stopwords)
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.train_batch_size, collate_fn=data_collator_fn, shuffle=True, pin_memory=False)

    valid_dataset = NERDataset(valid_data, label2id=label2id, max_length=CONFIG.max_length, tokenizer=tokenizer, stopwords=stopwords)
    valid_dataloader = DataLoader(valid_dataset, batch_size=CONFIG.valid_batch_size, collate_fn=data_collator_fn, shuffle=False, pin_memory=False)

    print('\n'+'='*150)
    print(f'Training will start with {len(train_dataset)} training datapoints and {len(valid_dataset)} validation datapoints.')
    print(f'Batch size being used for training is {CONFIG.train_batch_size} and maximum length for datapoints are {CONFIG.max_length}')
    print(f'The unique label number is {num_labels}, unique labels are {id2label.values()}')
    print('='*150)

    # Initiate Optimizer and Scheduler
    optimizer= AdamW(model.parameters(), lr= CONFIG.learning_rate, weight_decay= CONFIG.weight_decay)
    scheduler= lr_scheduler.CosineAnnealingLR(optimizer, T_max= CONFIG.T_max, eta_min= CONFIG.min_lr)

    # Training Loop
    NERTrainer(model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            num_epochs=CONFIG.num_epochs,
            id2label = id2label,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=CONFIG.output_dir,
            patience = CONFIG.patience,
            fold=fold)()
    
    del model, optimizer, scheduler, train_dataset, valid_dataset, train_dataloader, valid_dataloader
    torch.cuda.empty_cache()
    gc.collect()
    print(f'Fold {fold}, finished training.')
    print('+'*150)
