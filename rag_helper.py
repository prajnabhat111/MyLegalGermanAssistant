from langchain_chroma import Chroma
from models_config import get_embeddings, get_multilingual_model,get_multilingual_embeddings
import streamlit as st
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
from prompt_template import USER_PROMPT_TEMPLATE
import mlflow

@st.cache_resource
def load_vectorstore():
    vectorstore = Chroma( 
        collection_name= "v1.0",
        embedding_function=get_embeddings(),
        persist_directory="./ChromaDB"
    )
    return vectorstore

@st.cache_resource
def load_qdrant(collection_name: str = "german_rag_documents",
    url: str | None = None,
    api_key: str | None = None):

    url = url or os.getenv("QDRANT_URL")
    api_key = api_key or os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError(
            "QDRANT_URL is not set. Check your .env file or environment variables."
        )
    client = QdrantClient(url, api_key=api_key)  # or ":memory:"
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    return client, collection_name

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


@st.cache_resource
# class RAGBaseQdrant:

#     def __init__(self,
#         prompt_template=USER_PROMPT_TEMPLATE,
#         collection_name: str = "german_rag_documents",
#         qdrant_url: str = None,
#         qdrant_api_key: str = None,
#     ):
#         self.collection_name = collection_name
#         # 1. Load E5 embedding model
#         self.embedding_model = get_multilingual_model()
#         # 2. Connect to Qdrant
#         qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
#         qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
#         self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key )
#         self.prompt_template = prompt_template

#     def create_prompt(self, context, question):
#         return self.prompt_template.format(context=context, question=question)

#     def search_qdrant(self, query, top_k=5, collection_name="german_rag_documents"):
#         prefixed_query = f"query: {query}"
#         query_vector = get_multilingual_embeddings(self.embedding_model, prefixed_query)

#         results = self.qdrant_client.query_points(
#             collection_name=collection_name, query=query_vector, limit=top_k
#         )
#         retrieved_contexts = [
#             hit.payload["text"] for hit in results.points
#         ]

#         context = (
#             "\n\n".join(retrieved_contexts) if retrieved_contexts else "No relevant context found."
#         )

#         return context
    
#     def get_context(self, query):
#         context = self.search_qdrant(query)
#         # context = "\n\n".join([doc.page_content for doc in relevant_docs])

#         final_prompt = self.create_prompt(f"passage: {context}", query)

#         return final_prompt
class RAGBaseQdrant:

    def __init__(
        self,
        prompt_name: str = None,
        prompt_version: int = None,
        collection_name: str = "german_rag_documents",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
    ):
        self.collection_name = collection_name

        # 1. Load E5 embedding model
        self.embedding_model = get_multilingual_model()

        # 2. Connect to Qdrant
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.qdrant_client = QdrantClient(
            url=qdrant_url, api_key=qdrant_api_key
        )

        # 3. Load prompt from MLflow Registry or fallback to default
        self.prompt_template = self._load_mlflow_prompt(
            prompt_name, prompt_version
        )

    def _load_mlflow_prompt(self, prompt_name: str, version: int):
        """Fetches registered prompt from MLflow Prompt Registry."""
        if prompt_name:
            try:
                # Load from registry using MLflow 2.x/3.x GenAI API
                prompt_obj = mlflow.genai.load_prompt(
                    name_or_uri=prompt_name, version=version
                )
                return prompt_obj.template
            except Exception as e:
                print(f"Warning: Could not load prompt from MLflow ({e}). Using default.")
        return USER_PROMPT_TEMPLATE

    def retrieve_docs(
        self, query: str, top_k: int = 5, collection_name: str = None
    ) -> list[str]:
        """Retrieves raw matching text chunks from Qdrant as a list of strings.

        MLflow requires a list of strings to calculate Faithfulness & Groundedness.
        """
        target_collection = collection_name or self.collection_name
        prefixed_query = f"query: {query}"
        query_vector = get_multilingual_embeddings(
            self.embedding_model, prefixed_query
        )

        results = self.qdrant_client.query_points(
            collection_name=target_collection, query=query_vector, limit=top_k
        )

        # Return list of strings
        return [
            hit.payload.get("text", "")
            for hit in results.points
            if hit.payload
        ]

    def format_prompt(self, query: str, docs: list[str]) -> str:
        """Formats the retrieved document list into the registered prompt template."""
        context_str = (
            "\n\n".join(docs) if docs else "No relevant context found."
        )
        return self.prompt_template.format(
            context=f"passage: {context_str}", question=query
        )

    def get_context_and_prompt(self, query: str, top_k: int = 5):
        """Combined convenience helper for Streamlit."""
        docs = self.retrieve_docs(query, top_k=top_k)
        final_prompt = self.format_prompt(query, docs)
        return docs, final_prompt