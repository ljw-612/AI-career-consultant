import streamlit as st
import pandas as pd
import numpy as np

import os

from script.inference import infer, load_model, get_id_label_pair, get_device
from script.jd_extraction import get_webpage_text
from script.match_skills import make_embeddings, make_meta_embedding, get_missing_skills

from script.utils import read_uploaded_file, make_single_embedding, generate_df

from dotenv import load_dotenv
from openai import OpenAI

from pinecone import Pinecone, ServerlessSpec

model_name = "Pot-l/bert-ner-skills"    
tokenizer, model = load_model(model_name)
# label2id, id2label = get_id_label_pair(data_path='data/words_df.csv')
label2id = {'O': 0, 'B-Skill': 1, 'I-Skill': 2}; id2label = {0: 'O', 1: 'B-Skill', 2: 'I-Skill'}
device = get_device()

load_dotenv(override=True)
# openai_key = os.getenv("OPENAI_KEY")
# pinecone_key = os.getenv("PINECONE_API_KEY")

openai_key = st.secrets['OPENAI_API_KEY']
pinecone_key = st.secrets['PINECONE_API_KEY']

client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("coursera")



if __name__ == '__main__':
    
    st.title("AI Career Consultant")
    st.subheader("Missing skills in your resume for a job? We got you covered!")
    st.write("🤖️" + " Upload your resume and enter the url of the job description")
    st.write("🚀" + " We will identify the skills you are missing and suggest courses to learn them")
    
    
    uploaded_file = st.file_uploader("Upload your resume: ")
    url = st.text_input("Enter a job description URL")
    
    if uploaded_file is not None:
        pdf_text = read_uploaded_file(uploaded_file)
        data_r, skills_r = infer(pdf_text, tokenizer, model, label2id, id2label, device)
        # st.write("resume skills: ", skills_r)
        print("resume skills: ", skills_r)
        
    if url:
        webpage_text = get_webpage_text(url)
        data_j, skills_j = infer(webpage_text, tokenizer, model, label2id, id2label, device)
        # st.write("JD skills: ", skills_j)
        print("JD skills: ", skills_j)
    
    if uploaded_file is not None and url:

        meta_embedding_j = make_meta_embedding(make_embeddings(client=client, chunks=skills_j), skills_j)
        meta_embedding_r = make_meta_embedding(make_embeddings(client=client, chunks=skills_r), skills_r)
        
        missing_skills = get_missing_skills(meta_embedding_j, meta_embedding_r)
        if missing_skills == []:
            st.write("You have all the skills required for the job")
        else:
            missing_skills_str = ', '.join(missing_skills)
            print("missing skills: ", missing_skills_str)
            st.write("The skills that you don't have are -> ", missing_skills_str)
            
            missing_skills_embedding = make_single_embedding(client, missing_skills_str)
            
            result = index.query(
                vector=missing_skills_embedding,
                top_k=5,
                include_values=False,
                include_metadata=True
            )
            course_df = generate_df(result)
            st.write("Coursera courses we recommend for you:")
            st.write(course_df)
        