from google import genai
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings

@st.cache_resource
def get_client():
    client = genai.Client()
    return client

@st.cache_resource
def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return embeddings




