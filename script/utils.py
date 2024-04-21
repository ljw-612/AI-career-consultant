import PyPDF2

def make_single_embedding(client, text, model="text-embedding-3-small"):
    
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)