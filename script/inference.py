import pandas as pd
import numpy as np

import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForTokenClassification

from huggingface_hub import login
import os
from dotenv import load_dotenv

import PyPDF2

def read_pdf_as_string(file_path):
    # Open the PDF file
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        text = []
        
        # Loop through each page in the PDF
        for page in pdf_reader.pages:
            # Extract text from each page and append to the list
            text.append(page.extract_text())
        
        # Join all texts into a single string
        return '\n'.join(text)


def get_id_label_pair(data_path='../data/words_df.csv'):
    worddf = pd.read_csv(data_path)
    label2id = {k: v for v, k in enumerate(worddf.Tag.unique())}
    id2label = {v: k for v, k in enumerate(worddf.Tag.unique())}
    return label2id, id2label


def infer(pdf_text, tokenizer, model, label2id, id2label, device, MAX_LEN=128):
    '''
    This function allows user to input text and used the model to do inference on the text.
    '''
    sentence = pdf_text
    # divide pdf_text into sub sentences, this process is necessary 
    # because the model can only process 512 tokens at a time
    sub_sentences = [sentence[i:i+300] for i in range(0, len(sentence), 300)]

    skills = []; meta_wp_preds = []

    for sub_sentence in sub_sentences:
        inputs = tokenizer(sub_sentence, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding="max_length")
        
        ids = inputs['input_ids'].to(device, dtype = torch.long)
        mask = inputs['attention_mask'].to(device, dtype = torch.long)
        
        outputs = model(ids, mask)
        logits = outputs[0]
        
        active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) 

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        meta_wp_preds.extend(wp_preds)

    
    cleaned_preds = [t for t in meta_wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]
    data = cleaned_preds

    current_skill = ""
    for token, tag in data:
        if 'Skill' in tag:
            # If the token starts with '##', remove '##' and append the rest to the current skill
            if token.startswith('##'):
                current_skill += token[2:]
            # Otherwise, add a space before appending the token to the current skill
            else:
                current_skill += ' ' + token
        elif current_skill:
            # If the current skill is not empty and the current tag is not a skill, append the current skill to the skills list and reset the current skill
            skills.append(current_skill.strip())
            current_skill = ""

    # If the last token was a part of a skill, append the current skill to the skills list
    if current_skill:
        skills.append(current_skill.strip())
    
    skills = list(set(skills))
    
    return data, skills

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return tokenizer, model

def get_device():
    device = 'cuda' if cuda.is_available() else 'cpu'
    return device


def main():
    model_name = "Pot-l/bert-ner-skills"    
    tokenizer, model = load_model(model_name)

    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 20
    LEARNING_RATE = 1e-05
    MAX_GRAD_NORM = 10
    
    label2id, id2label = get_id_label_pair()
    
    pdf_text = read_pdf_as_string('../test_file/test_resume.pdf')
    
    skills = infer(pdf_text, tokenizer, model, label2id, id2label, device, MAX_LEN=MAX_LEN)
    print(skills)


if __name__ == '__main__':
    load_dotenv(override=True)
    
    device = get_device()
    print(device)
    
    login(
        token=os.getenv("HF_KEY"),
        add_to_git_credential=True
        )
    main()