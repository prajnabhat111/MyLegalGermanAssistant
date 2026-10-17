import requests
import bs4
from langchain_text_splitters  import HTMLHeaderTextSplitter
import re 
import time
from rag_helper import load_vectorstore, clean_text, load_qdrant
from qdrant_client.models import PointStruct
from models_config import get_multilingual_embeddings, get_multilingual_model
import os
from dotenv import load_dotenv


load_dotenv()

def load_data():

    url = "https://www.gesetze-im-internet.de/englisch_bmg/englisch_bmg.html"
    response = requests.get(url)

    print("Parsing the text...")
    soup = bs4.BeautifulSoup(response.content, 'html.parser')
    for e in soup:
        if isinstance(e, bs4.element.ProcessingInstruction):
            e.extract()
            break
    
    source_title =soup.title.string.strip() if soup.title else "General German Laws"

    for p in soup.find_all("p", style="text-align: center; font-weight: bold"):
        text_content = p.get_text().strip()
        
        # Logic: If it starts with "Division", make it H1
        if text_content.startswith("Division"):
            p.name = "h1"
        
        # Logic: If it starts with "Section", make it H2
        elif text_content.startswith("Subdivision"):
            p.name = "h2"
        
        elif text_content.startswith("Section"):
            p.name = "h3"

    headers_to_split_on  =  [("h1", "Division"), ("h2", "Subdivision"), ("h3","Section")]

    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_element=False
    )

    docs = splitter.split_text(soup.decode())

    for doc in docs:
        doc.metadata["source"] = source_title
    
    docs = [d for d in docs if d.page_content.strip().lower() != "table of contents"]
    docs = [d for d in docs if 'Division' in d.metadata if d.page_content.strip().lower() != d.metadata['Division'].strip().lower()]
    docs = [d for d in docs if 'Section' in d.metadata if d.page_content.strip().lower() != d.metadata['Section'].strip().lower()]

    for d in docs:
        d.page_content = clean_text(d.page_content)

    return docs


# def insert_into_vectorstore(docs, vectorstore, batch_size=50):
#     safe_batch_size = min(batch_size, 100) # Assuming the API has a hard limit of 100 per request
#     for i in range(0, len(docs), safe_batch_size):
#         batch = docs[i : i + safe_batch_size]
        
#         try:
#             vectorstore.add_documents(batch)
#             print(f"Successfully stored chunks {i} to {i + len(batch)}")
            
#             # Give the API "breathing room"
#             # 100 RPM means you can do 1 request every 0.6 seconds.
#             # We'll wait 5 seconds to be extra safe for the free tier.
#             time.sleep(5) 
            
#         except Exception as e:
#             if "429" in str(e):
#                 print("🚨 Rate limit hit! Sleeping for 60 seconds...")
#                 time.sleep(60) # Wait a full minute if we get blocked
#                 vectorstore.add_documents(batch)
#                 print(f"Successfully stored chunks {i} to {i + len(batch)}")

# def ingest():
#     docs = load_data()
#     vectorstore = load_vectorstore()
#     insert_into_vectorstore(docs, vectorstore)


def ingest_into_qdrant(client, passage_vectors,raw_chunks, start_idx: int = 0,  collection_name="german_rag_documents"):
    points = [
        PointStruct(
            id=start_idx + idx,
            vector=vector.tolist() if hasattr(vector, "tolist") else vector,
            payload={"text": chunk},
        )
        for idx, (vector, chunk) in enumerate(zip(passage_vectors, raw_chunks))
    ]
    
    client.upsert(collection_name=collection_name, points=points)


def ingest(batch_size=50):
    docs = load_data()
    passages = [f"passage: {doc}" for doc in docs]
    qdrant_client, collection_name = load_qdrant()
    embedding_model = get_multilingual_model()
    print("Uploading points to Qdrant Cloud...")
    for i in range(0, len(docs), batch_size):
        raw_chunks = passages[i : i + batch_size]
        passage_vectors = get_multilingual_embeddings(embedding_model, raw_chunks)
        ingest_into_qdrant(qdrant_client, passage_vectors,raw_chunks, collection_name=collection_name)


if __name__ == "__main__":
    ingest()
