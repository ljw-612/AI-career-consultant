import PyPDF2
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

import os

def make_single_embedding(client, text, model="text-embedding-3-small"):
    
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)

def generate_df(result):
    df = None
    for match in result['matches']:
        if df is None:
            df = pd.DataFrame(match['metadata'], index=[0])
        else:
            df = pd.concat([df, pd.DataFrame(match['metadata'], index=[len(df)])])
    new_order = ['course_name', 'course_organization', 'course_rating',
                 'course_Certificate_type', 'course_time', 'course_skills',
                 'course_URL']
    df = df.reindex(columns=new_order)
    df = df.rename(columns={'course_name': 'Course Name',
                            'course_organization': 'Organization',
                            'course_rating': 'Rating',
                            'course_Certificate_type': 'Certificate Type',
                            'course_time': 'Time',
                            'course_skills': 'Skills',
                            'course_URL': 'URL'})
    return df

if __name__ == '__main__':
    load_dotenv(override=True)
    openai_key = os.getenv("OPENAI_KEY")
    client = OpenAI(api_key=openai_key)
    
    # print(make_single_embedding(client, "data science"))