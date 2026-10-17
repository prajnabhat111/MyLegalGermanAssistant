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
- RAG Evaluation techniques https://qdrant.tech/blog/rag-evaluation-guide/

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini 3 Flash |
| Framework | LangChain |
| Vector DB | ChromaDB/Qdrant |
| Frontend | Streamlit |
| Embeddings | Langchain Sentense Transformer multilingual model |

## Architectural Choices

Most of the tools and models are chosen based on the free tier availibility as this project is for learning and experimenting.

### LLM
I've used Gemini 2.5 Flash model as my base LLM. It can easily be replaced with any other general purpose LLM such as OpenAI models, Llama, Claude, Gemini pro, etc. Google is providing it for free through api access from google ai studio. So the simple reason for choosing this is it's free! 

However, depending on the usecase, pricing, context window, latency, etc, this choice may and will differ in production environments.

### Vector DB
When it comes to vector databases. There are several options we can choose from - ChromaDB, FAISS, Qdrant, Pinecone, Milvus. 
I started this project as a lerning journey so I initially used local db such as ChromaDB. FAISS utilizes GPU acceleration very nicely. So, if you have GPU cards, FAISS is a natural option. 

As I got comfortable with the concepts, I decided to go with Qdrant as it would give me production grade exposure.

## Chunking

It is very important to decide the chunking stratergy when you are building a RAG system. You do start with a fixed size chunk. But, the problem with that is your chunks can break abruptly and not capture the essential meaning. Then you move to having some overlap so that you have some continuity. But with fixed size chunks you still have the issue of ignoring the document's semantic structure adding overlap creates redundancy reducing the retreival quality. 

So, the best chunking stratergy is usually understanding the underlying sematic structure of the document. For example if the documents are HTML docs, it is much easier to chunk based on the ```<div>, <p> or <h1>``` tags. In case of Json, it gets much simpler. So to choose the best suited chunking method, you need to first understand your data. 

Ofcourse the chunking also depends on the embedding model's context window. If your ```<p>``` tag has more text than your context window, a hybrid method or tag heirarchy must be introduced to get the best quality embeddings. 

## Retreival Stratergy

There are a few different stratergies when it comes to retreival. 

    -  Keyword Search
    -  Dense Retreival
    -  Hybrid Search
    -  Metadata Filtering
    -  Reranking

How to choose the correct stratergy for your data? Keyword is one of the basic, earliest and easiest methods of retreival. You look for the matching keywords and retreive that chunk. But, the problem with this would be the "language". For example, if you have a question about cartwheel and if the model just returns random information of physical wheels, carts or historical concepts, that wouldn't be so accurate, wouldn't you agree? 

That leads us to dense retreival. The main concept RAGs are based of. It is simple vector semantic search. It retreives the chunks where the query semantically/meaningfully matches the information from your documents. 

A better and more effective stratergy would be to combine the two methods and do a hybrid search, capturing the exact match and the semantic meaning. It is the most choosen stratergies among all. 

Now, let's talk about metadata filtering. If you want your information from the year 2026, but the same information exists in every year from 1999, the retreived data would be too huge and just cuts off based on your cutoff(k) variable. But, if your data had the time period as a meta data and you filtered your data on that field, you would end up with a much accurate information. It helps with the data retreived saving you some tokens definitely, reducing your latency and a better answer quality. 

Even with metadata filtering, let's say you used hybrid search and now you have information from both the stratergies but you have got to pict the top k chunks, how'd you do it? Reranking. You use a reranking model to rank the retreived chunks and then pick from the top ones of a new list. However, you should be careful as it can quickly increase your latency and cost.

## RAG Evaluation Techniques

RAG evaluation should yeild results on:
1. Retrieval Effectiveness
2. Relevance of responses
3. Coherence of generated responses
4. Promt performance
5. Latency

(Precision@K, Recall@K, MRR, NDCG, Context relevance)

## Why RAG? and more...

Let's address some of the basic questions. 

Why choose RAG and not fine tuning? Well, if you think your data is not going to change very frequently, and you have good amount of data to be able to fine tune then you can definitely opt for the later. However, fine tuning is also more expensive. Additionally, fine tuning updates the internal parameters of the model, so again, the model answers your question from its memory, not from the latest data. There are chances of hallucination in this case and with RAG you can easily verify the retreived data using citations, and ofcourse, RAG requires lower training costs. 


How do we identify and reduce hallucinations? 
Better retreival, reranking, context validation, clear grounding instructions, citation requirements, refusing unsupported questions.


What happens when correct document is not retreived? fallback stratergy, rewrite the query, expand the search, use hybrid retreival, adjust similarity thresholds, ask clarifying question, return insufficient information. 

How to debug a RAG response that looks correct but is actually worng? Query, retreival, reranking, context, generation, validation. 


