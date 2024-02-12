from tqdm import tqdm
import torch, gc, os
import numpy as np
# from seqeval.metrics import f1_score
# from seqeval.metrics import recall_score, precision_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
binarizer = MultiLabelBinarizer()

class NERTrainer():
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 valid_dataloader, 
                 optimizer, 
                 scheduler, 
                 num_epochs,
                 id2label, 
                 output_dir,
                 patience,
                 fold):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.id2label = id2label
        self.patience = patience
        self.fold = fold

    def clear_memories(self):
        torch.cuda.empty_cache()
        gc.collect()
        
    def compute_metrics(self,logits, labels):
        predictions = logits.argmax(-1).numpy()
        true_labels = [label for label in labels.numpy().flatten() if label != -100 and label != 0]
        true_predictions = [prediction for prediction, label in zip(predictions.flatten(), labels.numpy().flatten()) if label !=-100 and label != 0]
        
        score = f1_score(y_true = true_labels, y_pred=true_predictions, average='micro')
        if score is None:
            score=0.0
        precision = precision_score(y_true = true_labels, y_pred = true_predictions, average = 'micro')
        recall = recall_score(y_true = true_labels, y_pred = true_predictions, average = 'micro')
        f5_score = (1 + 5*5) * recall * precision / (5*5*precision + recall)
        if f5_score is None:
            f5_score=0.0
        return {'score':score, 'f5_score':f5_score}
    
    def train_one_epoch(self, epoch):
        self.model.train()
        
        running_loss = 0
        progress_bar = tqdm(enumerate(self.train_dataloader),total=len(self.train_dataloader))
        scores = []
        f5_scores = []
        for step, data in progress_bar:
            out = self.model(**data)
            loss = out.loss
            logits = out.logits
            
            loss.backward()
            # print(optimizer)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            score_dict = self.compute_metrics(logits=logits.detach().cpu(),
                                               labels= data['labels'].detach().cpu())
            
            scores.append(score_dict['score'])
            f5_scores.append(score_dict['f5_score'])
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            running_loss += loss.item()
            epoch_loss = running_loss/(step+1)
            
            score = sum(scores)/len(scores)
            f5_score = sum(f5_scores)/len(f5_scores)
            progress_bar.set_postfix(Epoch = epoch,
                                    TrainingLoss = epoch_loss,
                                    F1 = score,
                                    F5 = f5_score,
                                    )
            
        del out, loss, logits
        self.clear_memories()
        return epoch_loss, score, f5_score
    
    def valid_one_epoch(self, epoch):
        self.model.eval()
        
        running_loss = 0
        progress_bar = tqdm(enumerate(self.valid_dataloader),total=len(self.valid_dataloader))
        scores = []
        f5_scores = []
        for index, data in progress_bar:
            with torch.no_grad():
                out = self.model(**data)
            loss = out.loss
            logits = out.logits
                
            running_loss += loss.item()
            epoch_loss = running_loss/(index+1)
            
            scores.append(self.compute_metrics(logits=logits.detach().cpu(),
                                               labels= data['labels'].detach().cpu()))[0]
            
            f5_scores.append(self.compute_metrics(logits=logits.detach().cpu(),
                                               labels= data['labels'].detach().cpu()))[1]
            
            score = sum(scores)/len(scores)
            f5_score = sum(f5_scores)/len(f5_scores)
            progress_bar.set_postfix(Epoch = epoch,
                                    ValidationLoss = epoch_loss,
                                    F1 = score,
                                    F5 = f5_score,
                                    )
            
        del out, loss, logits
        self.clear_memories()
        return epoch_loss, score, f5_score

    def __call__(self):
        print('\n')
        prev_best_loss = np.inf
        best_score = -np.inf
        best_f5_score = -np.inf
        model_output_dir=self.output_dir
        
        early_break_count = 0
        for epoch in range(self.num_epochs):
            training_loss, training_score, training_f5 = self.train_one_epoch(epoch = epoch)
            
            validation_loss, validation_score, validation_f5 = self.valid_one_epoch(epoch = epoch)
            
            print('='*170 + '\n')
            print(f'Fold- {self.fold}, epoch- {epoch}')
            print(f'Training Loss for epoch: {epoch} is {training_loss}, F1 Score is: {training_score}')
            print(f'Validation Loss for epoch: {epoch} is {validation_loss}, F1 Score is: {validation_score}')

            if validation_f5 > best_f5_score:
                print(f'F1 Score improved from {best_f5_score} --> {validation_f5}')
                best_f5_score = validation_f5
                
                checkpoint_dir = os.path.join(model_output_dir,f'Checkpoint-Fold-{self.fold}-F5')
                
                if not os.path.exists(model_output_dir):
                    os.mkdir(model_output_dir)
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(os.path.join(checkpoint_dir))
                    
                self.model.save_pretrained(save_directory = checkpoint_dir)
                print(f"Model Saved at {checkpoint_dir}")
                
                if validation_f5 > 0.95:
                    break
            
            elif validation_score > best_score:
                print(f'F1 Score improved from {best_score} --> {validation_score}')
                best_score = validation_score
                
                checkpoint_dir = os.path.join(model_output_dir,f'Checkpoint-Fold-{self.fold}-F1')
                
                if not os.path.exists(model_output_dir):
                    os.mkdir(model_output_dir)
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(os.path.join(checkpoint_dir))
                    
                self.model.save_pretrained(save_directory = checkpoint_dir)
                print(f"Model Saved at {checkpoint_dir}")
                
                if validation_score > 0.95:
                    break
                
            elif validation_loss < prev_best_loss:
                print(f'Loss improved from {prev_best_loss} --> {validation_loss}')
                prev_best_loss = validation_loss
                
                checkpoint_dir = os.path.join(model_output_dir,f'Checkpoints-Fold-{self.fold}-loss')
                
                if not os.path.exists(model_output_dir):
                    os.mkdir(model_output_dir)
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(os.path.join(checkpoint_dir))
                
                self.model.save_pretrained(save_directory = checkpoint_dir)
                print(f"Model Saved at {checkpoint_dir}")
                
            else:
                early_break_count +=1
                print(f'Early break is at {early_break_count}, will stop training at {self.patience}')
                if early_break_count >= self.patience:
                    print(f'Early Stopping')
                    break
                            
            print('\n' + '='*170)
        
        print(f'Training over with best loss: {prev_best_loss} and best F1: {best_score}')
        
        fold_dir = os.path.join(model_output_dir, f'Fold-{self.fold}')
        if not os.path.exists(os.path.join(model_output_dir, f'Fold-{self.fold}')):
            os.mkdir(fold_dir)
        self.model.save_pretrained(save_directory = fold_dir)
        
        print(f'Model saved at {model_output_dir}')
        print('='*170)