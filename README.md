# AI Career Intelligence & Document RAG

A RAG-powered Streamlit application for general PDF question answering and JD-based resume intelligence.

## Overview

This project combines LangChain, Gemini, FAISS, and Streamlit to build two workflows:

1. General Document RAG
2. Resume + Job Description Career Intelligence

Users can upload a PDF, retrieve relevant information, and ask questions grounded in the uploaded content.

For career analysis, users upload a resume PDF and paste a job description as text. The system estimates job alignment, identifies skill gaps, and provides evidence-based improvement suggestions.

## Features

### General Document RAG

- Upload PDF documents
- Extract PDF text with PyPDFLoader
- Split documents into chunks
- Generate Gemini embeddings
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
        PyPDFLoader                    Document Processing
              |                               |
          Chunking                    Separate RAG Indexes
              |                               |
       Gemini Embeddings                     |
              |                               |
            FAISS                            |
              |                               |
          Retriever                          |
              +---------------+---------------+
                              |
                         Gemini LLM
                              |
              +---------------+---------------+
              |               |               |
           RAG Q&A       ATS Matching     Skill Gaps
                              |
                       Recommendations
```

## RAG Pipeline

```text
PDF
 ↓
PyPDFLoader
 ↓
RecursiveCharacterTextSplitter
 ↓
Gemini Embeddings
 ↓
FAISS Vector Store
 ↓
Retriever
 ↓
Relevant Context
 ↓
Gemini
 ↓
Grounded Answer
```

## Career Intelligence Pipeline

```text
Resume PDF
     ↓
Resume RAG Index
     ↓
     ├──────────────┐
     │              │
     ↓              ↓
JD Text         Resume + JD Q&A
     ↓
JD RAG Index
     ↓
Relevant Evidence
     ↓
JD-Based Matching
     ↓
ATS Alignment
     ↓
Matched / Partial / Missing
     ↓
Skill Gaps
     ↓
Improvement Recommendations
```

## Technology Stack

- Python
- Streamlit
- LangChain
- LangChain Community
- LangChain Google GenAI
- Google Gemini
- Gemini Embeddings
- FAISS
- PyPDFLoader
- RecursiveCharacterTextSplitter
- python-dotenv

## Project Structure

```text
RAG_Project/
│
├── be.py
├── srt.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── myenv/
```

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG_Project
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure Gemini API key

Create a `.env` file:

```text
GOOGLE_API_KEY=your_google_api_key
```

Never commit `.env` to GitHub.

### 5. Run the application

```powershell
python -m streamlit run srt.py
```

## Usage

### General RAG

1. Open the Streamlit application.
2. Upload a PDF.
3. Wait for the RAG index to be created.
4. Enter a question.
5. Click Ask Document.
6. Review the answer and sources.

### Resume + JD Matching

1. Upload your resume PDF.
2. Copy a job description from a job portal or company website.
3. Paste it into the Job Description field.
4. Click Analyze Job Match.
5. Review:
   - ATS alignment score
   - Strong matches
   - Partial matches
   - Missing skills
   - Missing technologies
   - Skill gaps
   - Resume improvements
   - Learning recommendations
6. Ask follow-up questions using Resume + JD RAG.

## Security Practices

- API keys are loaded through environment variables.
- `.env` is excluded from Git.
- Uploaded PDF size is limited.
- Temporary PDF files are deleted after processing.
- The system instructs the LLM not to invent resume or JD information.
- The ATS score is presented as an estimated alignment score, not a guaranteed hiring or ATS result.

## Important Design Decision

The job description is accepted as text instead of requiring a PDF.

This makes the workflow practical for real job searching because job descriptions are commonly copied from:

- LinkedIn
- Naukri
- Company career pages
- Job portals
- Recruiter messages
- Emails

## Limitations

- ATS score is an estimated alignment score.
- Results depend on the quality and completeness of the resume and JD.
- Gemini API quotas depend on the configured Google API project.
- FAISS indexes are created during the application session.
- The current application is designed as a Streamlit application and does not require FastAPI.

## Future Improvements

Potential advanced RAG upgrades:

- Hybrid keyword + semantic retrieval
- Reranking
- Query rewriting
- Better metadata filtering
- RAG evaluation
- Retrieval quality metrics
- Answer faithfulness evaluation
- Persistent vector databases
- Authentication
- Docker deployment
- Cloud deployment
- Multi-document workspaces

## Example Interview Explanation

> I built an AI Career Intelligence system using LangChain, Gemini, FAISS, and Streamlit. The system supports general PDF RAG as well as resume and job-description matching. For career analysis, it retrieves evidence from the resume and JD, estimates their alignment, identifies matched, partial, and missing requirements, and generates prioritized improvement recommendations. I also added conversational RAG so users can ask why their score is low or which JD requirements they need to address.

## License

Add your preferred license before publishing the repository.
