import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch import cuda


device = 'cuda' if cuda.is_available() else 'cpu'
class NERTrainer:
    def __init__(self, model, training_loader, testing_loader, optimizer, max_grad_norm):
        self.model = model
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Training epoch: {epoch + 1}")
            self._train_epoch()

    def _train_epoch(self):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        self.model.train()
        
        for idx, batch in enumerate(self.training_loader):
            
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)

            outputs = self.model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if idx % 100==0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")
            
            # compute training accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, self.model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            tr_preds.extend(predictions)
            tr_labels.extend(targets)
            
            tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy
        
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=self.max_grad_norm
            )
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

class NERValidator:
    def __init__(self, model, testing_loader):
        self.model = model
        self.testing_loader = testing_loader

    def validate(self):
        # put model in evaluation mode
        model.eval()
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []
        
        with torch.no_grad():
            for idx, batch in enumerate(testing_loader):
                
                ids = batch['ids'].to(device, dtype = torch.long)
                mask = batch['mask'].to(device, dtype = torch.long)
                targets = batch['targets'].to(device, dtype = torch.long)
                
                outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
                loss, eval_logits = outputs.loss, outputs.logits
                
                eval_loss += loss.item()

                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)
            
                if idx % 100==0:
                    loss_step = eval_loss/nb_eval_steps
                    print(f"Validation loss per 100 evaluation steps: {loss_step}")
                
                # compute evaluation accuracy
                flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
                active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
                # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
                active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
                targets = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)
                
                eval_labels.extend(targets)
                eval_preds.extend(predictions)
                
                tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
                eval_accuracy += tmp_eval_accuracy
        
        #print(eval_labels)
        #print(eval_preds)

        labels = [id2label[id.item()] for id in eval_labels]
        predictions = [id2label[id.item()] for id in eval_preds]

        #print(labels)
        #print(predictions)
        
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}")

        return labels, predictions

# Your data loading and setup code here...

