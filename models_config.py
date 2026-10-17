from google import genai
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_client(api_key):
    client = genai.Client(api_key=api_key)
    return client

@st.cache_resource
def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return embeddings

@st.cache_resource
def get_multilingual_model():
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    return model

def get_multilingual_embeddings(model, passages):
    passage_vectors = model.encode(passages, show_progress_bar=True, normalize_embeddings=True)
    return passage_vectors




