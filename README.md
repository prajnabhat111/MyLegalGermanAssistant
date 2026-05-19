# MyLegalGermanAssistant

A Retrieval Augmented Generation Chatbot that helps internationals living in Germany with limited knowledge of the german language and information to learn about legal requirements/regulations in Germany. 

## Problem Statement

When I first landed in Germany, I had a dream, an admission letter for master's course, and a lot of questions. For an international who has just started learning German, it is often very difficult to know all the legal requirements of staying in Germany. One may argue "Oh there are a lot of resources available on the internet to help!". Yes there are but, how sure are you when it comes to the source of that information? 

This chatbot is specifically designed to answer any question in English that one may have regarding the rules and regulations of living in Germany, using the official documents provided by the government. 

## Motivation

I built this project to deepen my understanding of:
- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Prompt engineering
- Context management in LLM applications
- Hallucination reduction techniques
- End-to-end LLM application deployment

Instead of using a fully managed framework, I wanted to implement the pipeline step-by-step to understand the design tradeoffs behind modern GenAI systems.

These are some of the sources I used to gain deeper understanding of topics.
- Build Your First RAG Application with LLM  https://www.youtube.com/live/KSItlTAsMsk?si=K5DRpJteJF2erqqf
- AI Agents in LangGraph https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/
- MLFlow for beginners https://www.youtube.com/playlist?list=PLQqR_3C2fhUUOmaeowgv4WquvH515zVmo
- Stanford CS336 Language Modelling from Scratch http://youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_ 

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini 3 Flash |
| Framework | LangChain |
| Vector DB | ChromaDB |
| Frontend | Streamlit |
| Embeddings | Google Generative AI Embeddings |

