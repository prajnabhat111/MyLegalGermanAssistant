from langchain_chroma import Chroma
from models import get_embeddings
import streamlit as st
import re

USER_PROMPT_TEMPLATE = """
You are a helpful assistant that provides information about German legal laws based on the provided context.

If the context does not contain relevant information to answer the question, reply:
"Sorry, I will not be able to help you with that."

If the topic is related to German Laws, you cannot find the context to answer the question, reply:
"Sorry, the information requested is not available in the source documents."

Context:
{context}

Question:
{question}

Provide a structured summary using both points and paragraphs.

Ask further if there is anything else they need to know.
Be polite.
"""

@st.cache_resource
def load_vectorstore():
    vectorstore = Chroma( 
        collection_name= "v1.0",
        embedding_function=get_embeddings(),
        persist_directory="./ChromaDB"
    )
    return vectorstore

def clean_text(text):

    # Normalize unicode spaces
    text = text.replace('\xa0', ' ')

    # Remove table of contents
    text = re.sub(r'table of contents', '', text, flags=re.IGNORECASE)

    # Remove page numbers like:
    # "Page 3"
    # "3."
    # "- 5 -"
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text)
    text = re.sub(r'\n\d+\.\s+', '\n', text)

    # Remove repeated short broken lines
    text = re.sub(r'\n\s*\n', '\n', text)

    # Merge broken line wraps
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

@st.cache_resource
class RAGBase:

    def __init__(self,
        vectorstore=load_vectorstore(),
        prompt_template=USER_PROMPT_TEMPLATE
    ):
        self.vectorstore = vectorstore
        self.prompt_template = prompt_template

    def create_prompt(self, context, question):
        return self.prompt_template.format(context=context, question=question)

    def search_vectorstore(self, query, top_k=5):
        results = self.vectorstore.similarity_search(query, k=top_k)
        return results
    
    def get_context(self, query):
        relevant_docs = self.search_vectorstore(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        final_prompt = self.create_prompt(context, query)

        return final_prompt
