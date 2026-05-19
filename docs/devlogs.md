# Dev Log

## v1.0
Started with basic LLM(gemini). Loaded official html document on german federal laws (for now just registration law), cleaned, chunked, embedded, and stored to the vector database ChromaDB. Created a generic prompt template. Using vector search provided by chromadb, created a rag functionality that answers users questions on german registration law with the help of retrieved information. Created a simple streamlit frontend for visualisation. 