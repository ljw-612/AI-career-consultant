from dotenv import load_dotenv
from openai import OpenAI
import os

from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

load_dotenv(override=True)
    
try:
    openai_key = st.secrets['OPENAI_API_KEY']
except:
    openai_key = os.getenv("OPENAI_KEY")

client = OpenAI(api_key=openai_key)

def make_embeddings(client, chunks):
    '''
    This function creates embeddings for the chunks of text using the OpenAI API.
    '''
    def _make_embedding(client, chunk, model="text-embedding-3-small"):
        chunk = chunk.replace("\n", " ")
        return client.embeddings.create(input = [chunk], model=model).data[0].embedding
    
    embeddings = []
    for chunk in chunks:
        embedding = _make_embedding(client, chunk)
        embeddings.append(embedding)
    return embeddings

def make_meta_embedding(embeddings, skills):
    meta_embedding = {}
    for i, embedding in enumerate(embeddings):
        meta_embedding[skills[i]] = embedding
    return meta_embedding

def get_missing_skills(meta_embedding_j, meta_embedding_r):
    meta_embedding_j_copy = meta_embedding_j.copy()
    
    for skill_j, embedding_j in meta_embedding_j.items():
        for skill_r, embedding_r in meta_embedding_r.items():
            
            # Compute the cosine similarity between the two embeddings
            similarity = cosine_similarity([embedding_j], [embedding_r])[0][0]
            if similarity > 0.6 and skill_j in meta_embedding_j_copy.keys():
                del meta_embedding_j_copy[skill_j]
    return list(meta_embedding_j_copy.keys())

if __name__ == '__main__':
    
    list_j = ['data scraping', 'ai', 'data science', 'database', 'data', 'data analytics', 'software', 'database querying', 'python']
    list_r = ['dart', 'sas', 'ge', 'analytics', 'neural', 'excel', 'sea', 'ma', 'or', 'data management', 'n', 'scikit', 'sql', 'git', 'database management', 'machine', 's', 'data visualization', 'machine learn', 'vision', 'r', 'gcp', 'data science', 'javascript', 'matlab', 'pyspark', 'java', 'databricks', 'p', 'python']

    meta_embedding_j = make_meta_embedding(make_embeddings(client, list_j), list_j)
    meta_embedding_r = make_meta_embedding(make_embeddings(client, list_r), list_r)
    
    missing_skills = get_missing_skills(meta_embedding_j, meta_embedding_r)
    
    print("the skills that you don't have are: ", missing_skills)
    