import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/api"


st.set_page_config(
    page_title="Multilingual RAG Assistant",
    page_icon="🤖",
    layout="wide",
)


st.title("🤖 Multilingual RAG Assistant")
st.write(
    "Ask questions about your uploaded documents "
    "in English, Marathi, or Hindi."
)


question = st.text_area(
    "Ask your question",
    placeholder="Example: What is the total amount of the tactile push button switch?",
    height=120,
)


if st.button("Ask", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Processing your question..."):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question
                    },
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"Could not connect to the API: {exc}"
                )

            else:

                st.success("Answer generated successfully.")

                # -----------------------------------------
                # Language
                # -----------------------------------------

                st.subheader("Detected Language")

                st.write(
                    result.get(
                        "language",
                        "Unknown",
                    )
                )

                # -----------------------------------------
                # Answer
                # -----------------------------------------

                st.subheader("Answer")

                st.write(
                    result.get(
                        "answer",
                        "No answer returned.",
                    )
               )
                # -----------------------------------------
                # Contradiction
                # -----------------------------------------

                contradiction = result.get(
                    "contradiction"
                )

                if contradiction:

                    st.subheader(
                        "Contradiction Check"
                    )

                    if contradiction.get(
                        "has_contradiction",
                        False,
                    ):
                        st.warning(
                            contradiction.get(
                                "explanation",
                                "Contradiction detected.",
                            )
                        )
                    else:
                        st.success(
                            "No contradiction found."
                        )

                # -----------------------------------------
                # Sources
                # -----------------------------------------

                sources = result.get(
                    "sources",
                    [],
                )

                st.subheader("Sources")

                if not sources:

                    st.info(
                        "No sources were returned."
                    )

                else:

                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):

                        with st.expander(
                            f"Source {index}: "
                            f"{source.get('citation', 'Unknown')}"
                        ):

                            st.write(
                                f"Similarity: "
                                f"{source.get('similarity_score', 'N/A')}"
                            )

                            st.write(
                                f"Quality: "
                                f"{source.get('quality', 'N/A')}"
                            )