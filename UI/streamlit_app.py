import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/api/ask"


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multilingual RAG Assistant",
    page_icon="🤖",
    layout="wide",
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

    .language-card {
        padding: 15px 20px;
        border-radius: 10px;
        background-color: #1f2937;
        border: 1px solid #374151;
        font-size: 18px;
        font-weight: 600;
    }

    .source-card {
        padding: 18px;
        border-radius: 10px;
        background-color: #111827;
        border: 1px solid #374151;
        margin-bottom: 12px;
    }

    .source-title {
        font-size: 17px;
        font-weight: 650;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 14px;
    }

    .metric-value {
        font-size: 16px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
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
    Ask questions about your uploaded documents in
    <b>English</b>, <b>Marathi</b>, or <b>Hindi</b>.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Question Input
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Ask your question</div>',
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
    use_container_width=False,
)


# --------------------------------------------------
# API Request
# --------------------------------------------------

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question before clicking Ask."
        )

    else:

        with st.spinner("Searching documents and generating answer..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

                st.success(
                    "Answer generated successfully."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the API. "
                    "Make sure FastAPI is running on "
                    "http://127.0.0.1:8000"
                )

                st.stop()

            except requests.exceptions.Timeout:

                st.error(
                    "The API request timed out. "
                    "Please try again."
                )

                st.stop()

            except requests.exceptions.HTTPError as exc:

                st.error(
                    f"API request failed: {exc}"
                )

                st.stop()

            except Exception as exc:

                st.error(
                    f"Unexpected error: {exc}"
                )

                st.stop()


        # --------------------------------------------------
        # Detected Language
        # --------------------------------------------------

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
                {language}
            </div>
            """,
            unsafe_allow_html=True,
        )


        # --------------------------------------------------
        # Answer
        # --------------------------------------------------

        st.markdown(
            '<div class="section-title">💡 Answer</div>',
            unsafe_allow_html=True,
        )

        answer = result.get(
            "answer",
            "No answer returned.",
        )

        st.markdown(answer)


        # --------------------------------------------------
        # Contradiction Check
        # --------------------------------------------------

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
                    explanation
                )

            else:

                st.success(
                    "No contradiction found."
                )


        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        st.markdown(
            '<div class="section-title">📚 Sources</div>',
            unsafe_allow_html=True,
        )

        sources = result.get(
            "sources",
            [],
        )

        if not sources:

            st.info(
                "No sources were returned."
            )

        else:

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

                st.markdown(
                    f"""
                    <div class="source-card">

                        <div class="source-title">
                            📄 Source {index}: {citation}
                        </div>

                        <div class="metric-label">
                            Similarity Score
                        </div>

                        <div class="metric-value">
                            {similarity:.3f}
                        </div>

                        <br>

                        <div class="metric-label">
                            Retrieval Quality
                        </div>

                        <div class="metric-value">
                            {quality}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )