# Dev Log

## v1.0
Started with basic LLM(gemini). Loaded official html document on german federal laws (for now just registration law), cleaned, chunked, embedded, and stored to the vector database ChromaDB. Created a generic prompt template. Using vector search provided by chromadb, created a rag functionality that answers users questions on german registration law with the help of retrieved information. Created a simple streamlit frontend for visualisation.


## v1.1 
Moving from prototype to production style code. 
    - Choosing a multilingual huggingface embedding model to accomodate future moving to german docs
    - Changing the vector DB from croma db to Qdrant to experience production level tradeoffs and choices 

## v1.2
Integrating MlFlow for evaluation
- Evaluating RAG for prompt performance, faithfulness, and latency