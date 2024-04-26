import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification

from torch import cuda

from huggingface_hub import login
import os

'''
This file is used for bert model training
Can be directly run in the terminal
model will be pushed to hugging face: https://huggingface.co/Pot-l/bert-ner-skills
'''

class dataset(Dataset):
    '''
    The dataset class, padding and tokens are added to the input data
    '''
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    
    def __getitem__(self, index):
        sentence = self.data.sentence[index]
        labels = self.data.word_labels[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, labels, self.tokenizer)
        
        # Add [CLS] and [SEP] tokens
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        labels.insert(0, "O") # add outside label for [CLS] token
        labels.insert(-1, "O") # add outside label for [SEP] token
        
        # truncating/padding
        maxlen = self.max_len
        
        if (len(tokenized_sentence) > maxlen):
            # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            labels = labels[:maxlen]
        else:
            # pad
            tokenized_sentence += ['[PAD]' for _ in range(maxlen - len(tokenized_sentence))]
            labels += ["O" for _ in range(maxlen - len(labels))]
            
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        
        # convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        
        
        label_ids = [label2id[label] for label in labels]
        
        return {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def __len__(self):
        return self.len


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def load_data(MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, tokenizer):
    data = pd.read_csv('../data/dataset.csv')

    train_size = 0.8
    train_data = data.sample(frac=train_size, random_state=200)
    test_data = data.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_data.shape))
    print("TEST Dataset: {}".format(test_data.shape))

    training_set = dataset(train_data, tokenizer, MAX_LEN)
    testing_set = dataset(test_data, tokenizer, MAX_LEN)
    
    
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    
    return training_set, testing_set, training_loader, testing_loader


def valid_loss(model, testing_loader):
    '''
    This function calculates the validation loss
    used for train/validation loss tracking
    '''
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
    
    eval_loss = eval_loss/nb_eval_steps
    return eval_loss


def train(epoch, model, optimizer, training_loader, testing_loader, device, MAX_GRAD_NORM, tr_logits):
    model = model.to(device)
    model.train() #put model in training mode
    epoch_loss_list = []
    validation_loss_list = [] # list to track validation loss
    
    for epoch in range(epoch):
        
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        
        for idx, batch in enumerate(training_loader):
            
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if idx % 10==0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss per 10 training steps: {loss_step}")
            
            # compute training accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
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
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")
        
        epoch_loss_list.append(epoch_loss)
        
        validation_loss = valid_loss(model, testing_loader)
        print(f"Validation loss epoch: {validation_loss}")
        validation_loss_list.append(validation_loss)
        
        model.train()
    

    return model, epoch_loss_list, validation_loss_list 
        

def train_model(id2label, label2id, device, LEARNING_RATE, training_set, testing_set, training_loader, testing_loader):
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', 
                                                   num_labels=len(id2label),
                                                   id2label=id2label,
                                                   label2id=label2id)
    model.to(device)
    
    ids = training_set[0]["ids"].unsqueeze(0)
    mask = training_set[0]["mask"].unsqueeze(0)
    targets = training_set[0]["targets"].unsqueeze(0)
    ids = ids.to(device)
    mask = mask.to(device)
    targets = targets.to(device)
    outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
    initial_loss = outputs[0]
    
    tr_logits = outputs[1]
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    model, epoch_loss_list, validation_loss_list = train(EPOCHS, model, optimizer, training_loader, testing_loader, device, MAX_GRAD_NORM, tr_logits)
    
    return model, epoch_loss_list, validation_loss_list


def push_model(model, tokenizer):
    os.environ["HF_KEY"] = "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"
    
    login(
        token=os.environ.get('HF_KEY'),
        add_to_git_credential=True
        )
    
    model_name = "bert-ner-skills"

    tokenizer.push_to_hub("Pot-l/bert-ner-skills", use_temp_dir=True)
    model.push_to_hub("Pot-l/bert-ner-skills", use_temp_dir=True)
    
    return
    

if __name__ == "__main__":
    device = 'cuda' if cuda.is_available() else 'cpu'
    print("device: ", device)
    
    worddf = pd.read_csv('../data/words_df.csv')
    label2id = {k: v for v, k in enumerate(worddf.Tag.unique())}
    id2label = {v: k for v, k in enumerate(worddf.Tag.unique())}
    
    print("Number of tags: {}".format(len(worddf.Tag.unique())))
    frequencies = worddf.Tag.value_counts()
    print("Tag frequencies: \n{}".format(frequencies))
    
    # training config
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 20
    LEARNING_RATE = 1e-05
    MAX_GRAD_NORM = 10
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    training_set, testing_set, training_loader, testing_loader = load_data(MAX_LEN, 
                                                                           TRAIN_BATCH_SIZE, 
                                                                           VALID_BATCH_SIZE, 
                                                                           tokenizer)
    
    model, epoch, validation_loss = train_model(id2label, label2id, device, LEARNING_RATE, training_set, testing_set, training_loader, testing_loader)
    
    # push model to hugging face
    push_model(model, tokenizer)

    
    

    