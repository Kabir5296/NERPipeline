from tqdm import tqdm
import torch, gc, os
import numpy as np
from seqeval.metrics import f1_score

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
                 patience):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.id2label = id2label
        self.patience = patience

    def clear_memories(self):
        torch.cuda.empty_cache()
        gc.collect()
        
    def compute_metrics(self,logits, labels):
        predictions = logits.argmax(-1).numpy()
        true_labels = [[ self.id2label[l] for l in label if l != -100] for label in labels.numpy()]
        true_predictions = [[self.id2label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels.numpy())]
        
        score = f1_score(y_true = true_labels, y_pred=true_predictions, average='macro')
        return score
    
    def train_one_epoch(self, epoch):
        self.model.train()
        
        running_loss = 0
        progress_bar = tqdm(enumerate(self.train_dataloader),total=len(self.train_dataloader))
        scores = []
        
        for step, data in progress_bar:
            out = self.model(**data)
            loss = out.loss
            logits = out.logits
            
            loss.backward()
            # print(optimizer)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            scores.append(self.compute_metrics(logits=logits.detach().cpu(),
                                               labels= data['labels'].detach().cpu()))
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            running_loss += loss.item()
            epoch_loss = running_loss/(step+1)
            
            score = sum(scores)/len(scores)
            progress_bar.set_postfix(Epoch = epoch,
                                    TrainingLoss = epoch_loss,
                                    F1 = score
                                    )
            
        del out, loss, logits
        self.clear_memories()
        return epoch_loss, score
    
    def valid_one_epoch(self, epoch):
        self.model.eval()
        
        running_loss = 0
        progress_bar = tqdm(enumerate(self.valid_dataloader),total=len(self.valid_dataloader))
        scores = []
        for index, data in progress_bar:
            with torch.no_grad():
                out = self.model(**data)
            loss = out.loss
            logits = out.logits
                
            running_loss += loss.item()
            epoch_loss = running_loss/(index+1)
            scores.append(self.compute_metrics(logits=logits.detach().cpu(),
                            labels= data['labels'].detach().cpu()))
            
            score = sum(scores)/len(scores)
            progress_bar.set_postfix(Epoch = epoch,
                                    ValidationLoss = epoch_loss,
                                    F1 = score
                                    )
            
        del out, loss, logits
        self.clear_memories()
        return epoch_loss, score

    def __call__(self):
        print('\n')
        prev_best_loss = np.inf
        best_score = -np.inf
        model_output_dir=self.output_dir
        
        early_break_count = 0
        for epoch in range(self.num_epochs):
            training_loss, training_score = self.train_one_epoch(epoch = epoch)
            
            validation_loss, validation_score = self.valid_one_epoch(epoch = epoch)
            
            print('='*150 + '\n')
            print(f'Training Loss for epoch: {epoch} is {training_loss}, F1 Score is: {training_score}')
            print(f'Validation Loss for epoch: {epoch} is {validation_loss}, F1 Score is: {validation_score}')

            if validation_score > best_score:
                print(f'F1 Score improved from {best_score} --> {validation_score}')
                best_score = validation_score
                
                if not os.path.exists(model_output_dir):
                    os.mkdir(model_output_dir)
                if not os.path.exists(os.path.join(model_output_dir,f'Checkpoints')):
                    os.mkdir(os.path.join(model_output_dir,f'Checkpoints'))
                torch.save(self.model.state_dict(), os.path.join(model_output_dir,f'Checkpoints')+f'/model_epoch_{epoch}.bin')
                print(f"Model Saved at {os.path.join(model_output_dir,f'Checkpoints')+f'/model_epoch_{epoch}.bin'}")
                
                if validation_score > 0.95:
                    break
                
            elif validation_loss < prev_best_loss:
                print(f'Loss improved from {prev_best_loss} --> {validation_loss}')
                prev_best_loss = validation_loss
                
                if not os.path.exists(model_output_dir):
                    os.mkdir(model_output_dir)
                if not os.path.exists(os.path.join(model_output_dir,f'Checkpoints')):
                    os.mkdir(os.path.join(model_output_dir,f'Checkpoints'))
                torch.save(self.model.state_dict(), os.path.join(model_output_dir,f'Checkpoints')+f'/model_epoch_{epoch}.bin')
                print(f"Model Saved at {os.path.join(model_output_dir,f'Checkpoints')+f'/model_epoch_{epoch}.bin'}")
                
            else:
                early_break_count +=1
                print(f'Early break is at {early_break_count}, will stop training at {self.patience}')
                if early_break_count >= self.patience:
                    print(f'Early Stopping')
                    break
                            
            print('\n' + '='*150)
        print(f'Training over with best loss: {prev_best_loss} and best F1: {best_score}')
        self.model.save_pretrained(save_directory = model_output_dir)
        print(f'Model saved at {model_output_dir}')
        print('='*150)