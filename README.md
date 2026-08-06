# RAG System for Enterprise Document Q&A

A Retrieval-Augmented Generation system that lets users ask natural-language questions over a long-form enterprise document, grounding every answer in the source text to reduce hallucination.

## Business Context

Framed around a venture capital analyst use case (Andreessen Horowitz): analysts need fast, accurate answers from long-form research and case materials without reading the entire document each time, and without an LLM inventing plausible-sounding but unsupported answers. The source document used here is the Harvard Business Review case study "How Apple Is Organized for Innovation" (Joel M. Podolny and Morten T. Hansen, Nov-Dec 2020).

## What it does

The system chunks the source PDF, embeds the chunks, indexes them in a vector store, and retrieves the most relevant chunks for a given question before passing them to an LLM to generate a grounded answer. Two implementations are built and compared side by side:

- Local/open-source stack: llama.cpp running Llama-2-13B-chat, FAISS vector store, `BAAI/bge-base-en-v1.5` sentence-transformer embeddings, recursive character text splitting (chunk size 500, overlap 50).
- OpenAI stack: PyMuPDF for parsing, token-based recursive splitting via tiktoken (chunk size 800, overlap 100), `text-embedding-ada-002` embeddings, Chroma vector store, MMR retrieval (k=6, fetch_k=20), and gpt-4o-mini for generation behind a custom system/user prompt template.

## Key findings

The OpenAI stack was more turn-key to stand up and tune. Token-based splitting combined with MMR retrieval measurably reduced hallucination and duplicate/near-duplicate context versus naive similarity search. The system was designed to answer "I don't know" when the retrieved context doesn't support an answer, treated as a feature rather than a failure mode. The comparison also surfaces a practical model-selection tradeoff: local models for sensitive documents that can't leave a private environment, versus cloud models for speed and stronger reasoning. As the document library scales beyond a handful of files, the recommendation is to move from Chroma/FAISS to a managed vector database such as Pinecone.

## Tech stack

LangChain, OpenAI (gpt-4o-mini, text-embedding-ada-002), Chroma, FAISS, llama.cpp, sentence-transformers, PyMuPDF, Python.

## Data

The source document is a publicly available Harvard Business Review case study, used here for educational/research purposes only.

Built as part of a postgraduate program in AI Agents for Business Applications at UT Austin's McCombs School of Business.
