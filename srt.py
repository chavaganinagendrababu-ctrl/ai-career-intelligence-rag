import streamlit as st
import tempfile
import os

from langchain_core.documents import Document

from be import (
    create_rag,
    create_rag_from_documents,
    ask_document,
    analyze_job_match,
    ask_resume_jd
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Career Intelligence",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# INITIALIZE RESET COUNTERS
# =========================================================

if "general_reset" not in st.session_state:
    st.session_state.general_reset = 0

if "career_reset" not in st.session_state:
    st.session_state.career_reset = 0


# =========================================================
# CLEAR GENERAL RAG
# =========================================================

def clear_general():

    st.session_state.general_retriever = None
    st.session_state.general_file_name = None
    st.session_state.general_question = ""

    st.session_state.general_reset += 1


# =========================================================
# CLEAR CAREER ANALYSIS
# =========================================================

def clear_career():

    st.session_state.resume_retriever = None
    st.session_state.resume_file_name = None

    st.session_state.jd_retriever = None
    st.session_state.jd_text = ""

    st.session_state.job_analysis = None
    st.session_state.career_question = ""

    st.session_state.career_reset += 1


# =========================================================
# API ERROR HANDLER
# =========================================================

def handle_api_error(error):

    error_message = str(error)

    if "RESOURCE_EXHAUSTED" in error_message:

        st.warning(
            "Gemini API quota has been reached. "
            "Please wait for the quota to reset."
        )

    elif "API_KEY" in error_message:

        st.error(
            "Gemini API key is invalid or missing."
        )

    else:

        st.error(
            "Something went wrong while processing the request."
        )

        st.exception(error)


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Career Intelligence")

st.write(
    "RAG-powered document Q&A, resume-job matching, "
    "skill-gap analysis and career recommendations."
)

st.divider()


# =========================================================
# GENERAL DOCUMENT RAG
# =========================================================

st.header("📄 General Document RAG")

st.write(
    "Upload a PDF and ask questions based on its content."
)


document_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    key=f"general_document_{st.session_state.general_reset}"
)


if document_file:

    MAX_FILE_SIZE = 20 * 1024 * 1024

    if document_file.size > MAX_FILE_SIZE:

        st.error(
            "File is too large. Maximum allowed size is 20 MB."
        )

    else:

        current_file = document_file.name

        if (
            st.session_state.get("general_file_name")
            != current_file
        ):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    document_file.getvalue()
                )

                document_path = temp_file.name


            with st.spinner(
                "Processing document..."
            ):

                try:

                    st.session_state.general_retriever = (
                        create_rag(document_path)
                    )

                    st.session_state.general_file_name = (
                        current_file
                    )

                    st.success(
                        "Document processed successfully."
                    )

                except Exception as e:

                    handle_api_error(e)

                finally:

                    if os.path.exists(document_path):

                        os.remove(document_path)


if (
    st.session_state.get("general_retriever")
    is not None
):

    question = st.text_input(
        "Ask a question",
        placeholder="Example: What are the main findings?",
        key=f"general_question_{st.session_state.general_reset}"
    )


    col1, col2 = st.columns(2)


    with col1:

        ask_general = st.button(
            "🔍 Ask Document",
            use_container_width=True,
            key=f"ask_general_{st.session_state.general_reset}"
        )


    with col2:

        clear_general_button = st.button(
            "🗑️ Clear Document",
            use_container_width=True,
            key=f"clear_general_{st.session_state.general_reset}"
        )


    if clear_general_button:

        clear_general()

        st.rerun()


    if ask_general:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document..."
            ):

                try:

                    result = ask_document(
                        st.session_state.general_retriever,
                        question
                    )

                    st.subheader(
                        "💬 Answer"
                    )

                    st.write(result)

                except Exception as e:

                    handle_api_error(e)


st.divider()


# =========================================================
# CAREER INTELLIGENCE
# =========================================================

st.header("🎯 Resume + Job Matching")

st.write(
    "Upload your resume and paste a job description "
    "to understand your alignment with the role."
)


# =========================================================
# RESUME UPLOAD
# =========================================================

st.subheader("1. Upload Resume")


resume_file = st.file_uploader(
    "Resume PDF",
    type=["pdf"],
    key=f"resume_file_{st.session_state.career_reset}"
)


if resume_file:

    MAX_RESUME_SIZE = 10 * 1024 * 1024

    if resume_file.size > MAX_RESUME_SIZE:

        st.error(
            "Resume is too large. Maximum allowed size is 10 MB."
        )

    else:

        current_resume = resume_file.name

        if (
            st.session_state.get("resume_file_name")
            != current_resume
        ):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    resume_file.getvalue()
                )

                resume_path = temp_file.name


            with st.spinner(
                "Processing resume..."
            ):

                try:

                    st.session_state.resume_retriever = (
                        create_rag(resume_path)
                    )

                    st.session_state.resume_file_name = (
                        current_resume
                    )

                    st.success(
                        "Resume processed successfully."
                    )

                except Exception as e:

                    handle_api_error(e)

                finally:

                    if os.path.exists(resume_path):

                        os.remove(resume_path)


# =========================================================
# JOB DESCRIPTION
# =========================================================

st.subheader("2. Paste Job Description")


job_description = st.text_area(
    "Job Description",
    placeholder=(
        "Copy the complete job description "
        "and paste it here..."
    ),
    height=300,
    key=f"job_description_{st.session_state.career_reset}"
)


# =========================================================
# CREATE JD RAG
# =========================================================

if (
    job_description.strip()
    and st.session_state.get("resume_retriever")
    is not None
):

    if (
        st.session_state.get("jd_text")
        != job_description
    ):

        with st.spinner(
            "Preparing job description..."
        ):

            try:

                jd_document = Document(
                    page_content=job_description,
                    metadata={
                        "source": "Job Description"
                    }
                )

                st.session_state.jd_retriever = (
                    create_rag_from_documents(
                        [jd_document]
                    )
                )

                st.session_state.jd_text = (
                    job_description
                )

                st.success(
                    "Job description processed successfully."
                )

            except Exception as e:

                handle_api_error(e)


# =========================================================
# ATS ANALYSIS
# =========================================================

if (
    st.session_state.get("resume_retriever")
    is not None
    and
    st.session_state.get("jd_retriever")
    is not None
):

    st.divider()

    st.subheader("3. ATS Job Match")


    col1, col2 = st.columns(2)


    with col1:

        analyze_button = st.button(
            "🎯 Analyze Job Match",
            use_container_width=True,
            key=f"analyze_{st.session_state.career_reset}"
        )


    with col2:

        clear_career_button = st.button(
            "🗑️ Clear Resume + JD",
            use_container_width=True,
            key=f"clear_career_{st.session_state.career_reset}"
        )


    if clear_career_button:

        clear_career()

        st.rerun()


    if analyze_button:

        with st.spinner(
            "Comparing resume with job description..."
        ):

            try:

                result = analyze_job_match(
                    st.session_state.resume_retriever,
                    st.session_state.jd_retriever
                )

                st.session_state.job_analysis = result

            except Exception as e:

                handle_api_error(e)


# =========================================================
# DISPLAY ATS RESULT
# =========================================================

if (
    st.session_state.get("job_analysis")
    is not None
):

    st.subheader(
        "📊 Job Match Analysis"
    )

    st.info(
        "The ATS score is an estimated alignment score "
        "based on your resume and this job description."
    )

    st.write(
        st.session_state.job_analysis
    )


    st.download_button(
        "⬇️ Download Job Analysis",
        data=str(
            st.session_state.job_analysis
        ),
        file_name="job_match_analysis.txt",
        mime="text/plain",
        key=f"download_{st.session_state.career_reset}"
    )


# =========================================================
# RESUME + JD RAG Q&A
# =========================================================

if (
    st.session_state.get("resume_retriever")
    is not None
    and
    st.session_state.get("jd_retriever")
    is not None
):

    st.divider()

    st.subheader(
        "4. Ask AI About Your Resume + JD"
    )


    career_question = st.text_input(
        "Ask your question",
        placeholder=(
            "Example: Why is my ATS score low?"
        ),
        key=f"career_question_{st.session_state.career_reset}"
    )


    if st.button(
        "💬 Ask AI",
        use_container_width=True,
        key=f"ask_career_{st.session_state.career_reset}"
    ):

        if not career_question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Retrieving evidence from resume and JD..."
            ):

                try:

                    result = ask_resume_jd(
                        st.session_state.resume_retriever,
                        st.session_state.jd_retriever,
                        career_question
                    )

                    st.subheader(
                        "🤖 AI Answer"
                    )

                    st.write(result)

                except Exception as e:

                    handle_api_error(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Career Intelligence | "
    "LangChain + FAISS + Local Embeddings + Gemini + Streamlit"
)