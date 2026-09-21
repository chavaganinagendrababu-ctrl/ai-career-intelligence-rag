# AI Career Intelligence & Document RAG

A RAG-powered Streamlit application for general PDF question answering and job-description-based resume intelligence.

## Overview

This project combines LangChain, Gemini, FAISS, Hugging Face Sentence Transformers, and Streamlit to build two workflows:

1. General Document RAG
2. Resume + Job Description Career Intelligence

Users can upload a PDF, retrieve relevant information, and ask questions grounded in the uploaded content.

For career analysis, users upload a resume PDF and paste a job description as text. The system estimates job alignment, identifies skill gaps, and provides evidence-based improvement suggestions.

## Features

### General Document RAG

- Upload PDF documents
- Extract PDF text with PyPDFLoader
- Split documents into chunks
- Generate embeddings using a local Hugging Face Sentence Transformer model
- Use `all-MiniLM-L6-v2` for embeddings
- Store vectors in FAISS
- Retrieve relevant chunks
- Generate grounded answers with Gemini
- Show source information
- Support large document Q&A within configured file limits
- Clear uploaded document state

### Resume + Job Matching

- Upload resume as PDF
- Paste job description directly into the application
- Create separate retrieval indexes for resume and JD
- Estimate ATS alignment score from 0 to 100
- Identify strong matches
- Identify partial matches
- Identify missing required skills
- Identify missing technologies
- Analyze experience and project alignment
- Prioritize skill gaps
- Recommend resume improvements
- Recommend learning priorities

### Resume + JD RAG Q&A

Users can ask questions such as:

- Why is my ATS score low?
- Which required skills are missing?
- Which JD requirements only partially match my resume?
- What should I improve first?
- Which project is most relevant to this role?
- Which skills should I learn for this position?

Answers are grounded in retrieved resume and JD evidence.

## Architecture

```text
                         Streamlit UI
                              |
              +---------------+---------------+
              |                               |
       General Document                  Career Mode
              |                               |
          PDF Upload                  Resume PDF + JD Text
              |                               |
        PyPDFLoader                 Document Processing
              |                               |
           Chunking                  Separate RAG Indexes
              |                               |
 Hugging Face Sentence Transformer           |
              |                               |
            FAISS                             |
              |                               |
          Retriever                           |
              +---------------+---------------+
                              |
                         Gemini LLM
                              |
              +---------------+---------------+
              |               |               |
           RAG Q&A       ATS Matching     Skill Gaps
                              |
                       Recommendations