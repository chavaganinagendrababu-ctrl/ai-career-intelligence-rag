from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import GoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


if not api_key:
    try:
        import streamlit as st

        api_key = st.secrets["GOOGLE_API_KEY"]

    except Exception:
        api_key = None


if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is not configured. "
        "Add it to .env locally or Streamlit Secrets when deployed."
    )


GENERATION_MODEL = "gemini-3.5-flash-lite"


# =========================================================
# LOCAL EMBEDDING MODEL
# =========================================================

def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return embeddings


# =========================================================
# LOAD PDF
# =========================================================

def load_pdf(pdf_path):

    loader = PyPDFLoader(
        pdf_path
    )

    documents = loader.load()

    return documents


# =========================================================
# SPLIT DOCUMENTS
# =========================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# =========================================================
# CREATE RAG FROM DOCUMENTS
# =========================================================

def create_rag_from_documents(documents):

    chunks = split_documents(
        documents
    )

    embeddings = get_embeddings()

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 5
        }
    )

    return retriever


# =========================================================
# CREATE RAG FROM PDF
# =========================================================

def create_rag(pdf_path):

    documents = load_pdf(
        pdf_path
    )

    retriever = create_rag_from_documents(
        documents
    )

    return retriever


# =========================================================
# FORMAT DOCUMENTS
# =========================================================

def format_docs(docs):

    formatted_docs = []

    for doc in docs:

        page_number = (
            doc.metadata.get(
                "page",
                0
            ) + 1
        )

        source = doc.metadata.get(
            "source",
            "Document"
        )

        formatted_docs.append(
            f"[Source: {source}, Page {page_number}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(
        formatted_docs
    )


# =========================================================
# GEMINI MODEL
# =========================================================

def get_llm():

    return GoogleGenerativeAI(
        model=GENERATION_MODEL,
        api_key=api_key
    )


# =========================================================
# GENERAL DOCUMENT RAG
# =========================================================

def ask_document(
    retriever,
    question
):

    client = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a document question-answering assistant.

        Answer the question using only the retrieved
        document context.

        Do not use outside knowledge.

        If the answer is not available in the context,
        say:

        "I couldn't find this information in the uploaded document."

        Give a clear and useful answer.

        At the end, provide the source pages used.

        Format:

        ANSWER:
        ...

        SOURCES:
        - Page X
        - Page Y

        CONTEXT:
        {context}

        QUESTION:
        {question}
        """
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | client
    )

    result = chain.invoke(
        question
    )

    return result


# =========================================================
# GET RELEVANT DOCUMENTS
# =========================================================

def get_relevant_documents(
    retriever,
    queries,
    max_documents=10
):

    collected = []

    seen = set()

    for query in queries:

        documents = retriever.invoke(
            query
        )

        for document in documents:

            page = document.metadata.get(
                "page",
                0
            )

            source = document.metadata.get(
                "source",
                ""
            )

            key = (
                source,
                page,
                document.page_content[:100]
            )

            if key not in seen:

                seen.add(key)

                collected.append(
                    document
                )

    return collected[:max_documents]


# =========================================================
# JOB MATCHING
# =========================================================

def analyze_job_match(
    resume_retriever,
    jd_retriever
):

    resume_queries = [
        "technical skills programming languages tools technologies",
        "projects machine learning data science software development",
        "work experience responsibilities achievements",
        "education certifications qualifications"
    ]

    jd_queries = [
        "required skills technical skills programming languages",
        "required technologies tools frameworks platforms",
        "experience requirements qualifications",
        "responsibilities preferred skills requirements"
    ]

    resume_docs = get_relevant_documents(
        resume_retriever,
        resume_queries,
        max_documents=12
    )

    jd_docs = get_relevant_documents(
        jd_retriever,
        jd_queries,
        max_documents=12
    )

    resume_context = format_docs(
        resume_docs
    )

    jd_context = format_docs(
        jd_docs
    )

    client = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert ATS and career matching system.

        Compare the candidate's resume against the
        provided job description.

        Use only the evidence provided.

        Do not invent skills, experience, projects,
        certifications, or qualifications.

        The ATS score represents the estimated alignment
        between this resume and THIS job description.

        It is not a guaranteed real ATS score.

        Analyze:

        1. ATS MATCH SCORE
        2. STRONG MATCHES
        3. PARTIAL MATCHES
        4. MISSING REQUIRED SKILLS
        5. MISSING TECHNOLOGIES
        6. EXPERIENCE MATCH
        7. PROJECT MATCH
        8. HIGH PRIORITY SKILL GAPS
        9. MEDIUM PRIORITY SKILL GAPS
        10. WHAT TO FOCUS ON FIRST
        11. RESUME IMPROVEMENTS
        12. LEARNING RECOMMENDATIONS

        Rules:

        - Strong match means the resume contains
          clear evidence of the requirement.

        - Partial match means related evidence exists,
          but the JD asks for something more specific.

        - Missing means the JD requires the skill and
          there is no supporting evidence in the resume.

        - Give more importance to required skills than
          preferred skills.

        - Do not tell the candidate to add a skill
          they do not genuinely possess.

        - Separate skills to learn from skills already
          possessed but poorly demonstrated.

        - Give practical recommendations.

        Return this structure:

        ATS MATCH SCORE:
        XX/100

        STRONG MATCHES:
        - ...
        - ...

        PARTIAL MATCHES:
        - Skill:
          Resume Evidence:
          JD Requirement:
          Explanation:

        MISSING REQUIRED SKILLS:
        - ...
        - ...

        MISSING TECHNOLOGIES:
        - ...
        - ...

        EXPERIENCE MATCH:
        ...

        PROJECT MATCH:
        ...

        HIGH PRIORITY SKILL GAPS:

        1. Skill:
           Reason:
           Action:

        2. Skill:
           Reason:
           Action:

        MEDIUM PRIORITY SKILL GAPS:

        1. Skill:
           Reason:
           Action:

        WHAT TO FOCUS ON FIRST:

        1. ...
        2. ...
        3. ...

        RESUME IMPROVEMENTS:

        - ...
        - ...
        - ...

        LEARNING RECOMMENDATIONS:

        - ...
        - ...
        - ...

        Do not claim that learning a skill guarantees
        a 90-95 ATS score.

        Explain which improvements would strengthen
        alignment with this particular JD.

        RESUME EVIDENCE:
        {resume_context}

        JOB DESCRIPTION EVIDENCE:
        {jd_context}
        """
    )

    chain = prompt | client

    result = chain.invoke(
        {
            "resume_context": resume_context,
            "jd_context": jd_context
        }
    )

    return result


# =========================================================
# RESUME + JD RAG Q&A
# =========================================================

def ask_resume_jd(
    resume_retriever,
    jd_retriever,
    question
):

    resume_documents = resume_retriever.invoke(
        question
    )

    jd_documents = jd_retriever.invoke(
        question
    )

    resume_context = format_docs(
        resume_documents
    )

    jd_context = format_docs(
        jd_documents
    )

    client = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI career intelligence assistant.

        The user has provided:

        1. A resume
        2. A job description

        Answer the user's question using only the
        retrieved evidence from these documents.

        Do not invent skills, experience, projects,
        qualifications, or job requirements.

        Clearly separate:

        RESUME EVIDENCE
        JOB DESCRIPTION EVIDENCE
        ANALYSIS

        If enough evidence is not available, say:

        "I couldn't find enough evidence in the uploaded
        resume or job description."

        Give a practical answer.

        At the end, provide sources.

        Format:

        ANSWER:
        ...

        RESUME EVIDENCE:
        ...

        JOB DESCRIPTION EVIDENCE:
        ...

        ANALYSIS:
        ...

        SOURCES:
        Resume - Page X
        Job Description

        RESUME CONTEXT:
        {resume_context}

        JOB DESCRIPTION CONTEXT:
        {jd_context}

        QUESTION:
        {question}
        """
    )

    chain = prompt | client

    result = chain.invoke(
        {
            "resume_context": resume_context,
            "jd_context": jd_context,
            "question": question
        }
    )

    return result