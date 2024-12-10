import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load the SentenceTransformer model to generate embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

# Load the data (adjust the file path as necessary)
data_path = 'Possible_case_Preprocessing_small.csv'
df = pd.read_csv(data_path)

# Preprocess and combine human and GPT text into one document
df['combined'] = df['human_clean'] + " " + df['gpt_clean']

# Generate embeddings for each combined text
embeddings = np.array([embedding_model.encode(text) for text in df['combined']])

# Create a FAISS index (Flat Index in this example)
dim = embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dim)  # L2 distance metric for similarity search

# Add embeddings to the index
index.add(embeddings)

# Create an IVF index
nlist = 20  # Number of clusters (adjust based on dataset size)
quantizer = faiss.IndexFlatL2(dim)  # Quantizer used for training
index = faiss.IndexIVFFlat(quantizer, dim, nlist)

# Train the index
index.train(embeddings)

# Add embeddings to the index
index.add(embeddings)

# Function to retrieve the top-k most similar documents for a query
def search_faiss(query, k=10):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    D, I = index.search(query_embedding, k)
    results = [df.iloc[i] for i in I[0]]
    return results

# Load the FLAN-T5 model
llm_model_name = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model on the appropriate device (either CPU or GPU)
model = AutoModelForSeq2SeqLM.from_pretrained(
    llm_model_name,
    device_map="auto" if device == "cuda" else "cpu",  # Automatically chooses device
    torch_dtype=torch.float16 if device == "cuda" else torch.float32  # Set dtype based on device
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

# Define the pipeline (without specifying the device here)
llm = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=500,
    num_beams=2,
    num_return_sequences=1,
    temperature=0.4
)

# Function to generate a chatbot response using the query and retrieved context
def generate_response(query, k=5):
    # Retrieve top-k similar documents
    retrieved_results = search_faiss(query, k)
    
    # Combine results into a context string
    context = "\n".join([
        f"User said: {result['human_clean']}\nResponse: {result['gpt_clean']}"
        for result in retrieved_results[:3]  # Use only the most relevant results
    ])

    prompt = (
        f"Context:\n{context}\n\n"
        f"Query: {query}\n\n"
        f"As an empathetic assistant, consider the user's situation and the context provided above. "
        f"Respond with detailed and actionable advice that addresses their concerns thoughtfully."
    )

    # Generate a response
    response = llm(prompt)[0]['generated_text']
    return response

# Streamlit UI
st.title("Empathetic Chatbot for Stress Management")

# Textbox to input the query
query = st.text_input("Ask a question about stress management:")

# If the user submits a query
if query:
    # Displaying the response to the user
    st.write("Generating response...")
    response = generate_response(query)
    st.write(f"Chatbot Response: {response}")

    # Option to enter another query
    st.write("---")
    query = st.text_input("Ask another question:")

# Enable multiple queries to be handled in sequence
if query:
    with st.form(key='query_form'):
        st.text_area('Chat with the Assistant', value='', height=200)
        submit_button = st.form_submit_button(label='Submit Query')

        if submit_button:
            st.write(generate_response(query))
