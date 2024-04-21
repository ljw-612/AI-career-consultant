import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import PyPDF2

import os

from script.inference import infer, load_model, get_id_label_pair, get_device
from script.jd_extraction import get_webpage_text
from script.match_skills import make_embeddings, make_meta_embedding, get_missing_skills

from dotenv import load_dotenv
from openai import OpenAI

model_name = "Pot-l/bert-ner-skills"    
tokenizer, model = load_model(model_name)
label2id, id2label = get_id_label_pair(data_path='data/words_df.csv')
device = get_device()

load_dotenv(override=True)
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai_key)


def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)


if __name__ == '__main__':
    
    uploaded_file = st.file_uploader("Choose a file")
    url = st.text_input("Enter a webpage URL")
    
    if uploaded_file is not None:
        pdf_text = read_uploaded_file(uploaded_file)
        data_r, skills_r = infer(pdf_text, tokenizer, model, label2id, id2label, device)
        st.write("resume skills: ", skills_r)
        
    if url:
        webpage_text = get_webpage_text(url)
        data_j, skills_j = infer(webpage_text, tokenizer, model, label2id, id2label, device)
        st.write("JD skills: ", skills_j)
    
    if uploaded_file is not None and url:

        meta_embedding_j = make_meta_embedding(make_embeddings(client=client, chunks=skills_j), skills_j)
        meta_embedding_r = make_meta_embedding(make_embeddings(client=client, chunks=skills_r), skills_r)
        
        missing_skills = get_missing_skills(meta_embedding_j, meta_embedding_r)
        st.write(missing_skills)
