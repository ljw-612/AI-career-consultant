import numpy as np
import pandas as pd

from pinecone import Pinecone, ServerlessSpec
import os
from tqdm import tqdm

import ast
import streamlit as st


def upload_to_db(data):
    '''
    This function upload the data after embedding to the pinecone db
    pinecone db host: https://coursera-2ct3vx8.svc.aped-4627-b74a.pinecone.io, name: coursera
    '''
    pc = Pinecone(api_key=st.secrets['PINECONE_API_KEY'])
    index = pc.Index("coursera")
    
    batch_size = 50
    
    test = []
    
    for i in tqdm(range(0, len(data), batch_size)):
        vectors = []
        batch = data.iloc[i:i+batch_size]
        for idx in range(len(batch)):
            vector = {}; metadata = {}
            vector["id"] = str(idx+i)
            vector["values"] = batch.iloc[idx]['course_skills_embeddings']

            metadata["course_name"] = batch.iloc[idx]['course_title']; test.append(batch.iloc[idx]['course_title'])
            metadata["course_organization"] = batch.iloc[idx]['course_organization']
            metadata["course_Certificate_type"] = batch.iloc[idx]['course_certificate_type']
            metadata["course_rating"] = batch.iloc[idx]['course_rating']
            metadata["course_difficulty"] = batch.iloc[idx]['course_difficulty']
            metadata["course_URL"] = batch.iloc[idx]['course_url']
            metadata["course_skills"] = batch.iloc[idx]['course_skills']
            metadata["course_time"] = batch.iloc[idx]['course_time']

            vector["metadata"] = metadata
            vectors.append(vector)
        
        index.upsert(vectors=vectors)
        print(f"Uploaded batch {i+1} to {i+batch_size} to Pinecone")


def load_data():
    coursera = pd.read_csv('../data/coursera_courses_embeddings.csv')
    coursera['course_skills_embeddings'] = coursera['course_skills_embeddings'].apply(ast.literal_eval)
    coursera = coursera.dropna()
    coursera = coursera.reset_index(drop=True)
    coursera['course_rating'] = coursera['course_rating'].astype(str)
    
    return coursera

def main():
    coursera = load_data()
    upload_to_db(coursera)

if __name__ == '__main__':
    
    pass