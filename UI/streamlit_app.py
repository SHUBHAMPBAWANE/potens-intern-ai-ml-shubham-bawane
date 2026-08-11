import re
import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_URL = "http://127.0.0.1:8000"

ASK_API_URL = f"{BASE_URL}/api/ask"
UPLOAD_API_URL = f"{BASE_URL}/api/upload"


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multilingual RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-card {
        padding: 18px 20px;
        border-radius: 10px;
        background-color: #111827;
        border: 1px solid #374151;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .info-title {
        font-size: 18px;
        font-weight: 650;
        margin-bottom: 12px;
    }

    .language-card {
        padding: 15px 20px;
        border-radius: 10px;
        background-color: #1f2937;
        border: 1px solid #374151;
        font-size: 18px;
        font-weight: 600;
    }

    .document-card {
        padding: 18px;
        border-radius: 10px;
        background-color: #111827;
        border: 1px solid #374151;
        margin-bottom: 12px;
    }

    .document-title {
        font-size: 18px;
        font-weight: 650;
        margin-bottom: 12px;
    }

    .source-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #111827;
        border: 1px solid #374151;
        margin-bottom: 15px;
    }

    .source-title {
        font-size: 18px;
        font-weight: 650;
        margin-bottom: 15px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 17px;
        font-weight: 650;
    }

    .quality-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin-top: 5px;
    }

    .quality-good {
        background-color: #14532d;
        color: #86efac;
    }

    .quality-fair {
        background-color: #713f12;
        color: #fde68a;
    }

    .quality-weak {
        background-color: #7f1d1d;
        color: #fca5a5;
    }

    .quality-unknown {
        background-color: #374151;
        color: #d1d5db;
    }

    .answer-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #111827;
        border: 1px solid #374151;
        margin-bottom: 10px;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #374151;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🤖 RAG Assistant")

    st.markdown(
        """
        ### About

        A multilingual Retrieval-Augmented Generation
        assistant that answers questions using information
        retrieved from uploaded documents.

        **Supported languages**
        - 🇬🇧 English
        - 🇮🇳 Hindi
        - 🇮🇳 Marathi

        **Core capabilities**
        - 📄 Document upload
        - 🔍 Document retrieval
        - 🧠 Semantic similarity search
        - 🌐 Multilingual question answering
        - 📚 Source citations
        - 📊 Retrieval quality scoring
        - ⚠️ Contradiction detection
        - 📑 Document and page references
        """
    )

    st.divider()

    st.markdown("### 🔌 System Status")

    st.caption(
        f"FastAPI: `{BASE_URL}`"
    )

    st.caption(
        "Gemini is called only when you submit a question."
    )


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 Multilingual RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Ask questions about your documents in
        <b>English</b>, <b>Marathi</b>, or <b>Hindi</b>.
        Answers are grounded in retrieved document content.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# DOCUMENT MANAGEMENT
# ==================================================

st.markdown(
    '<div class="section-title">📄 Document Management</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "docx"],
    help="Upload a PDF, TXT, or DOCX document.",
)


if uploaded_file is not None:

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    upload_button = st.button(
        "⬆️ Upload & Process",
        type="secondary",
    )

    if upload_button:

        with st.spinner(
            "Uploading and processing document..."
        ):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(
                    UPLOAD_API_URL,
                    files=files,
                    timeout=120,
                )

                response.raise_for_status()

                upload_result = response.json()

                st.session_state["upload_result"] = upload_result

                st.success(
                    upload_result.get(
                        "message",
                        "Document uploaded successfully.",
                    )
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the FastAPI server. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ Document processing timed out. "
                    "Please try again."
                )

            except requests.exceptions.HTTPError as exc:

                try:

                    error_detail = response.json().get(
                        "detail",
                        str(exc),
                    )

                except Exception:

                    error_detail = str(exc)

                st.error(
                    f"❌ Document upload failed: {error_detail}"
                )

            except Exception as exc:

                st.error(
                    f"❌ Unexpected upload error: {exc}"
                )


# ==================================================
# UPLOADED DOCUMENT INFORMATION
# ==================================================

if "upload_result" in st.session_state:

    upload_result = st.session_state["upload_result"]

    filename = upload_result.get(
        "filename",
        "Unknown",
    )

    ingestion = upload_result.get(
        "ingestion",
        {},
    )

    documents = ingestion.get(
        "documents",
        0,
    )

    chunks = ingestion.get(
        "chunks",
        0,
    )

    vectors = ingestion.get(
        "vectors",
        0,
    )

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                📋 Uploaded Document Information
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📄 File",
            filename,
        )

    with col2:

        st.metric(
            "📚 Documents",
            documents,
        )

    with col3:

        st.metric(
            "🧩 Chunks",
            chunks,
        )

    with col4:

        st.metric(
            "🔢 Vectors",
            vectors,
        )


# ==================================================
# QUESTION INPUT
# ==================================================

st.markdown(
    '<div class="section-title">💬 Ask your question</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Question",
    placeholder=(
        "Example: What is the total amount of the invoice?"
    ),
    height=120,
    label_visibility="collapsed",
)

ask_button = st.button(
    "🔍 Ask",
    type="primary",
)


# ==================================================
# API REQUEST
# ==================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question before clicking Ask."
        )

        st.stop()


    with st.spinner(
        "Searching documents and generating answer..."
    ):

        try:

            response = requests.post(
                ASK_API_URL,
                json={
                    "question": question.strip()
                },
                timeout=120,
            )


            # ------------------------------------------
            # Handle 429
            # ------------------------------------------

            if response.status_code == 429:

                st.error(
                    "⚠️ Gemini API quota has been exceeded. "
                    "Please wait before trying again."
                )

                st.stop()


            # ------------------------------------------
            # Handle Server Errors
            # ------------------------------------------

            if response.status_code >= 500:

                error_text = response.text.lower()

                if (
                    "resource_exhausted" in error_text
                    or "quota" in error_text
                    or "429" in error_text
                ):

                    st.error(
                        "⚠️ Gemini API quota has been exceeded "
                        "or temporarily exhausted."
                    )

                    st.info(
                        "Your RAG pipeline is still working, "
                        "but Gemini generation cannot be used "
                        "until the quota becomes available."
                    )

                else:

                    st.error(
                        "❌ The backend API encountered an "
                        "internal error."
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.code(
                            response.text
                        )

                st.stop()


            response.raise_for_status()

            result = response.json()

            st.success(
                "✅ Answer generated successfully."
            )


        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the API."
            )

            st.info(
                f"Make sure FastAPI is running on {BASE_URL}"
            )

            st.stop()


        except requests.exceptions.Timeout:

            st.error(
                "⏱️ The API request timed out. "
                "Please try again."
            )

            st.stop()


        except requests.exceptions.HTTPError as exc:

            try:

                error_detail = response.json().get(
                    "detail",
                    str(exc),
                )

            except Exception:

                error_detail = str(exc)

            st.error(
                f"❌ API request failed: {error_detail}"
            )

            st.stop()


        except ValueError:

            st.error(
                "❌ The API returned an invalid JSON response."
            )

            st.stop()


        except Exception as exc:

            st.error(
                f"❌ Unexpected error: {exc}"
            )

            st.stop()


    # ==================================================
    # DETECTED LANGUAGE
    # ==================================================

    st.markdown(
        '<div class="section-title">🌐 Detected Language</div>',
        unsafe_allow_html=True,
    )

    language = result.get(
        "language",
        "Unknown",
    )

    st.markdown(
        f"""
        <div class="language-card">
            🌐 {language}
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ==================================================
    # ANSWER
    # ==================================================

    st.markdown(
        '<div class="section-title">💡 Answer</div>',
        unsafe_allow_html=True,
    )

    answer = result.get(
        "answer",
        "No answer returned.",
    )

    with st.container(border=True):

        st.markdown(
            answer
        )


    # ==================================================
    # DOCUMENT INFORMATION
    # ==================================================

    sources = result.get(
        "sources",
        [],
    )

    st.markdown(
        '<div class="section-title">📄 Document Information</div>',
        unsafe_allow_html=True,
    )

    document_names = set()
    page_references = []


    for source in sources:

        citation = source.get(
            "citation",
            "Unknown source",
        )

        # ------------------------------------------
        # Extract Filename
        # ------------------------------------------

        filename_match = re.match(
            r"(.+?)\s*\(Page\s+\d+\)",
            citation,
            re.IGNORECASE,
        )

        if filename_match:

            filename = filename_match.group(1).strip()

        else:

            filename = citation.strip()

        document_names.add(
            filename
        )


        # ------------------------------------------
        # Extract Page Number
        # ------------------------------------------

        page_match = re.search(
            r"Page\s+(\d+)",
            citation,
            re.IGNORECASE,
        )

        if page_match:

            page_number = page_match.group(1)

            page_references.append(
                f"{filename} — Page {page_number}"
            )


    # Remove duplicate pages

    page_references = list(
        dict.fromkeys(
            page_references
        )
    )


    # ------------------------------------------
    # Document Metrics
    # ------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Documents Referenced",
            len(document_names),
        )

    with col2:

        st.metric(
            "Retrieved Sources",
            len(sources),
        )

    with col3:

        st.metric(
            "Pages Referenced",
            len(page_references),
        )


    # ------------------------------------------
    # Referenced Files
    # ------------------------------------------

    st.markdown(
        '<div class="document-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="document-title">📁 Referenced Files</div>',
        unsafe_allow_html=True,
    )

    if document_names:

        for document in sorted(
            document_names
        ):

            st.write(
                f"📄 {document}"
            )

    else:

        st.write(
            "No document information available."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    # ------------------------------------------
    # Referenced Pages
    # ------------------------------------------

    if page_references:

        with st.expander(
            "📑 View referenced pages"
        ):

            for page in page_references:

                st.write(
                    f"• {page}"
                )


    # ==================================================
    # CONTRADICTION CHECK
    # ==================================================

    contradiction = result.get(
        "contradiction"
    )


    if contradiction:

        st.markdown(
            '<div class="section-title">⚠️ Contradiction Check</div>',
            unsafe_allow_html=True,
        )

        has_contradiction = contradiction.get(
            "has_contradiction",
            False,
        )

        explanation = contradiction.get(
            "explanation",
            "No contradiction information available.",
        )


        if has_contradiction:

            st.warning(
                f"⚠️ Contradiction detected\n\n"
                f"{explanation}"
            )

        else:

            st.success(
                "✅ No contradiction found."
            )


    # ==================================================
    # RETRIEVAL QUALITY + SOURCES
    # ==================================================

    st.markdown(
        '<div class="section-title">📊 Retrieval Quality</div>',
        unsafe_allow_html=True,
    )

    sources = result.get(
        "sources",
        [],
    )


    if not sources:

        st.info(
            "No retrieval sources were returned."
        )

    else:

        # --------------------------------------------------
        # Calculate overall retrieval score
        # --------------------------------------------------

        similarity_scores = []

        for source in sources:

            score = source.get(
                "similarity_score",
                0,
            )

            try:

                score = float(score)

            except (
                TypeError,
                ValueError,
            ):

                score = 0.0

            similarity_scores.append(
                score
            )


        if similarity_scores:

            overall_score = (
                sum(similarity_scores)
                / len(similarity_scores)
            )

        else:

            overall_score = 0.0


        # --------------------------------------------------
        # Overall Retrieval Quality
        # --------------------------------------------------

        if overall_score >= 0.75:

            overall_quality = "Strong"
            quality_icon = "🟢"

        elif overall_score >= 0.60:

            overall_quality = "Good"
            quality_icon = "🟢"

        elif overall_score >= 0.40:

            overall_quality = "Fair"
            quality_icon = "🟡"

        else:

            overall_quality = "Weak"
            quality_icon = "🔴"


        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-title">
                    {quality_icon} Overall Retrieval Quality
                </div>

                <div style="font-size: 28px; font-weight: 700;">
                    {overall_quality}
                </div>

                <div style="color:#9ca3af; margin-top:5px;">
                    Average similarity score:
                    <b>{overall_score:.3f}</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # --------------------------------------------------
        # Source Details
        # --------------------------------------------------

        st.markdown(
            '<div class="section-title">📚 Retrieved Sources</div>',
            unsafe_allow_html=True,
        )


        for index, source in enumerate(
            sources,
            start=1,
        ):

            citation = source.get(
                "citation",
                "Unknown source",
            )

            similarity = source.get(
                "similarity_score",
                0,
            )

            quality = source.get(
                "quality",
                "Unknown",
            )


            try:

                similarity = float(
                    similarity
                )

            except (
                TypeError,
                ValueError,
            ):

                similarity = 0.0


            # Make sure score stays between 0 and 1

            similarity = max(
                0.0,
                min(
                    similarity,
                    1.0,
                ),
            )


            # --------------------------------------------------
            # Quality display
            # --------------------------------------------------

            quality_lower = str(
                quality
            ).lower()


            if quality_lower == "strong":

                quality_icon = "🟢"

            elif quality_lower == "good":

                quality_icon = "🟢"

            elif quality_lower == "fair":

                quality_icon = "🟡"

            elif quality_lower == "weak":

                quality_icon = "🔴"

            else:

                quality_icon = "⚪"


            # --------------------------------------------------
            # Source Card
            # --------------------------------------------------

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### 📄 Source {index}"
                )

                st.write(
                    f"**Citation:** {citation}"
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Similarity Score",
                        f"{similarity:.3f}",
                    )


                with col2:

                    st.metric(
                        "Retrieval Quality",
                        f"{quality_icon} {quality}",
                    )


                # --------------------------------------------------
                # Similarity Progress
                # --------------------------------------------------

                st.progress(
                    similarity,
                    text=(
                        f"Similarity: "
                        f"{similarity * 100:.1f}%"
                    ),
                )


    # ==================================================
    # QUERY INFORMATION
    # ==================================================

    st.markdown(
        '<div class="section-title">🔎 Query Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Detected Language",
            language,
        )

    with col2:

        st.metric(
            "Sources Retrieved",
            len(sources),
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Multilingual RAG Assistant ·
        FastAPI + Streamlit + ChromaDB + Gemini
    </div>
    """,
    unsafe_allow_html=True,
)