import re, torch
from torch.utils.data import Dataset
from banglanlptoolkit import BnNLPNormalizer

def create_id_label_conversion(unique_labels):
    id2label = {}
    label2id = {}
    for index, label in enumerate((unique_labels)):
        label2id[label] = int(index)
        id2label[index] = label
    return id2label, label2id, len(id2label)

class preprocess():
    def __init__(self, 
                 label2id = None,
                 return_target_ids = True,
                 stopword_dict=[], 
                 stopword_remove = True,
                 punct_remove = True, 
                 to_lower = True, 
                 strip = True):
        
        self.to_lower = to_lower
        self.return_ids = return_target_ids
        self.punct_remove = punct_remove
        self.strip = strip
        self.stopword_dict = stopword_dict
        self.stopword_remove = stopword_remove
        self.label2id = label2id
        
    def remove_punctuations(self, text, label):
        if self.punct_remove:
            text = re.sub(r'[^\w\s]', '', text.strip())
            return re.sub(r'[_]', '', text.strip()) # if label == 'O' else text
        else:
            return text
    
    def clean_data(self, text):
        text = re.sub(r'[|iœОабвгдезиклмнопрстуцчщь。いくけしたてなを一不业中丰为了产人们任优会使便保內其力务務区口可各后吸售回國地場孔家富専己市引心懂成手技抗护挽捷措撬改教文断新施晦普晰服术来果業標正洋流海涩清源漁焕然物現畅留発的目真瞳研碑磨社科章続练细育自艺节行表見见論质资距进遗重鑽門间难]','',text.strip())
        return re.sub(r'[Éàáãçéêíя–—，’]', '', text.strip())
    
    def remove_stopwords(self, text, label):
        if not self.stopword_remove:
            return text
        if text in self.stopword_dict:
            return '' if label == 'O' else text
        else:
            return text
    
    def strip_empty_strings(self, tokens, labels):
        new_labels = []
        new_tokens = []
        if not self.strip:
            return tokens, labels
        
        for token, label in zip(tokens, labels):
            if token != '':
                new_tokens.append(token)
                if self.return_ids and self.label2id != None:
                    new_labels.append(self.label2id[label])
                else:
                    new_labels.append(label)
        return new_tokens, new_labels
        
    def __call__(self, tokens, labels):
        if self.to_lower:
            tokens = [self.remove_punctuations(self.clean_data(text), label).lower() for text, label in zip(tokens, labels)]
        else:
            tokens = [self.remove_punctuations(self.clean_data(text), label) for text, label in zip(tokens, labels)]
        
        tokens = [self.remove_stopwords(text, label) for text, label in zip(tokens, labels)]
        tokens, labels = self.strip_empty_strings(tokens, labels)
        if len(tokens) != len(labels):
            raise ValueError(f'The length of tokens are {len(tokens)} while the length of labels are {len(labels)}')
        return tokens, labels
    
class tokenize_data():
    def __init__(self, 
                 tokenizer, 
                 add_special_tokens=False, 
                 max_length=None, 
                 padding=False, 
                 truncation=None):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        
    def align_labels_to_ids(self, labels, word_id_list):
        new_labels = []
        prev_word_id = -1
        none_label_token = -101
        
        for _, word_id in enumerate(word_id_list):
            if word_id is None:
                new_labels.append(none_label_token)
            elif word_id != prev_word_id:
                prev_word_id = word_id
                new_labels.append(labels[word_id])
            elif word_id == prev_word_id:
                prev_word_id = word_id
                new_labels.append(labels[word_id])
        return new_labels
    
    def __call__(self, tokens, labels):
        tokenized_inputs = self.tokenizer(' '.join(tokens), 
                                          add_special_tokens=self.add_special_tokens, 
                                          truncation=self.truncation, 
                                          padding=self.padding, 
                                          max_length = self.max_length)
        word_id_list = tokenized_inputs.word_ids()
        labels = self.align_labels_to_ids(labels, word_id_list)
        if len(tokenized_inputs['input_ids']) != len(labels):
            raise ValueError(f"The length of tokenized inputs are {len(tokenized_inputs['input_ids'])} while the length of labels are {len(labels)}")
        return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels
    
class CustomDataCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer= tokenizer
        self.device = device
        
    def __call__(self, batch):
        output= {}
        max_len = max([len(ids['input_ids']) for ids in batch])
        
        output["input_ids"] = [sample['input_ids'] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output['labels'] = [sample['labels'] for sample in batch]
        
        max_len= max([len(ids) for ids in output['input_ids']])
        
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = torch.tensor([ids + (max_len - len(ids))*[self.tokenizer.pad_token_id] for ids in output['input_ids']], dtype=torch.long, device=self.device)
            output['attention_mask']= torch.tensor([mask + (max_len - len(mask))*[0] for mask in output['attention_mask']], dtype=torch.long, device=self.device)
            output['labels']= torch.tensor([target + (max_len - len(target))*[-100] for target in output['labels']], dtype=torch.long, device=self.device)
        else:
            output["input_ids"] = torch.tensor([(max_len - len(ids))*[self.tokenizer.pad_token_id] + ids for ids in output['input_ids']], dtype=torch.long, device=self.device)
            output['attention_mask']= torch.tensor([(max_len - len(mask))*[0] + mask for mask in output['attention_mask']], dtype=torch.long, device=self.device)
            output['labels']= torch.tensor([(max_len - len(target))*[-100] + target for target in output['labels']], dtype=torch.long, device=self.device)

        return output
    
class NERDataset(Dataset):
    def __init__(self, 
                 data,
                 label2id,
                 max_length,
                 tokenizer,
                 stopwords,
                 stopword_remove=False,
                 punct_remove=False,
                 ):
        
        self.label2id = label2id
        self.tokens = data.tokens
        self.labels = data.labels
        self.max_length = max_length
        self.preprocess_fn = preprocess(label2id=self.label2id,
                                        stopword_dict=stopwords,
                                        stopword_remove=stopword_remove,
                                        punct_remove=punct_remove,
                                        to_lower=True,
                                        strip=True,
                                        )
        self.tokenizer_fn = tokenize_data(tokenizer = tokenizer, max_length=self.max_length, truncation=True)
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        tokens, labels = self.preprocess_fn(self.tokens[index], self.labels[index])
        input_ids, attention_mask, labels = self.tokenizer_fn(tokens, labels)
        return {'input_ids' : input_ids, 'attention_mask' : attention_mask, 'labels' : labels}