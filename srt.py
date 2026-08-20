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
# HELPER FUNCTIONS
# =========================================================

def clear_general():

    keys = [
        "general_retriever",
        "general_file_name",
        "general_question"
    ]

    for key in keys:

        if key in st.session_state:
            del st.session_state[key]


def clear_career():

    keys = [
        "resume_retriever",
        "resume_file_name",
        "jd_retriever",
        "jd_text",
        "job_analysis",
        "career_question"
    ]

    for key in keys:

        if key in st.session_state:
            del st.session_state[key]


def handle_api_error(error):

    error_message = str(error)

    if "RESOURCE_EXHAUSTED" in error_message:

        st.warning(
            "Gemini API quota has been reached. "
            "Please wait for the quota to reset."
        )

    elif "API_KEY" in error_message:

        st.error(
            "Gemini API key is invalid. "
            "Check your .env file."
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
    "RAG-powered resume analysis, job matching, "
    "skill-gap identification and document Q&A."
)

st.divider()


# =========================================================
# GENERAL DOCUMENT RAG
# =========================================================

st.header("📄 General Document RAG")

st.write(
    "Upload any PDF and ask questions using its content."
)


document_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    key="general_document"
)


if document_file:

    # Maximum file size: 20 MB
    MAX_FILE_SIZE = 20 * 1024 * 1024

    if document_file.size > MAX_FILE_SIZE:

        st.error(
            "File is too large. Maximum allowed size is 20 MB."
        )

        st.stop()


    if (
        "general_retriever" not in st.session_state
        or st.session_state.get("general_file_name")
        != document_file.name
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
            "Processing document and creating RAG index..."
        ):

            try:

                st.session_state.general_retriever = (
                    create_rag(document_path)
                )

                st.session_state.general_file_name = (
                    document_file.name
                )

                st.success(
                    "Document processed successfully."
                )

            except Exception as e:

                handle_api_error(e)

            finally:

                if os.path.exists(document_path):

                    os.remove(document_path)


    if "general_retriever" in st.session_state:

        question = st.text_input(
            "Ask a question",
            placeholder="Example: What are the main findings?",
            key="general_question"
        )


        col1, col2 = st.columns(2)


        with col1:

            ask_button = st.button(
                "🔍 Ask Document",
                use_container_width=True
            )


        with col2:

            clear_button = st.button(
                "🗑️ Clear Document",
                use_container_width=True
            )


        if clear_button:

            clear_general()

            st.rerun()


        if ask_button:

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
    "Upload your resume and paste the job description "
    "to evaluate your alignment with the role."
)


# =========================================================
# RESUME
# =========================================================

st.subheader("1. Upload Resume")


resume_file = st.file_uploader(
    "Resume PDF",
    type=["pdf"],
    key="resume_file"
)


if resume_file:

    # Maximum resume size: 10 MB
    MAX_FILE_SIZE = 10 * 1024 * 1024

    if resume_file.size > MAX_FILE_SIZE:

        st.error(
            "Resume is too large. Maximum allowed size is 10 MB."
        )

        st.stop()


    if (
        "resume_retriever" not in st.session_state
        or st.session_state.get("resume_file_name")
        != resume_file.name
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
                    resume_file.name
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
        "Copy the complete job description and paste it here..."
    ),
    height=300,
    key="job_description"
)


# =========================================================
# CREATE JD RAG
# =========================================================

if (
    job_description.strip()
    and "resume_retriever" in st.session_state
):

    if (
        "jd_retriever" not in st.session_state
        or st.session_state.get("jd_text")
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
    "resume_retriever" in st.session_state
    and "jd_retriever" in st.session_state
):

    st.divider()

    st.subheader("3. ATS Job Match")


    col1, col2 = st.columns(2)


    with col1:

        analyze_button = st.button(
            "🎯 Analyze Job Match",
            use_container_width=True
        )


    with col2:

        clear_career_button = st.button(
            "🗑️ Clear Resume + JD",
            use_container_width=True
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

                st.session_state.job_analysis = (
                    result
                )

            except Exception as e:

                handle_api_error(e)


# =========================================================
# DISPLAY ATS RESULT
# =========================================================

if "job_analysis" in st.session_state:

    st.subheader(
        "📊 Job Match Analysis"
    )

    st.info(
        "The ATS score is an estimated alignment score "
        "based on the uploaded resume and job description."
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
        mime="text/plain"
    )


# =========================================================
# CAREER RAG Q&A
# =========================================================

if (
    "resume_retriever" in st.session_state
    and "jd_retriever" in st.session_state
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
        key="career_question"
    )


    if st.button(
        "💬 Ask AI",
        use_container_width=True
    ):

        if not career_question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Retrieving evidence..."
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
    "AI Career Intelligence | LangChain + FAISS + Gemini + Streamlit"
)