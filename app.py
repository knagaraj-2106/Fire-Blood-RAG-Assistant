import streamlit as st

from services.rag_service import RAGService
from utils.logger import logger


# -------------------------------
# Streamlit Configuration
# -------------------------------

st.set_page_config(
    page_title="Fire & Blood RAG Assistant",
    page_icon="🐉",
    layout="wide"
)


# -------------------------------
# Application Title
# -------------------------------

st.title(
    "🐉 Fire & Blood - RAG Assistant"
)

st.write(
    """
    Ask questions about House Targaryen history,
    characters, battles, dragons and events.
    """
)


# -------------------------------
# Initialize RAG Pipeline
# -------------------------------

@st.cache_resource
def load_rag_service():

    logger.info(
        "Initializing RAG Service..."
    )

    service = RAGService()

    logger.info(
        "RAG Service initialized successfully"
    )

    return service



try:

    rag_service = load_rag_service()


except Exception as e:

    logger.exception(
        f"Failed to initialize RAG Service: {e}"
    )

    st.error(
        "Unable to initialize RAG pipeline"
    )

    st.stop()



# -------------------------------
# User Question
# -------------------------------

question = st.text_input(
    "Ask your question:",
    placeholder="Example: Who killed Rhaenyra?"
)



# -------------------------------
# Generate Answer
# -------------------------------

if st.button(
    "Ask Dragon"
):


    if not question.strip():

        st.warning(
            "Please enter a question"
        )


    else:

        try:

            logger.info(
                f"User Question: {question}"
            )


            with st.spinner(
                "Searching the archives of Westeros..."
            ):


                response = rag_service.answer(
                    question
                )


            logger.info(
                "Answer generated successfully"
            )


            # --------------------------------
            # Answer Section
            # --------------------------------

            st.subheader(
                "🔥 Answer"
            )


            st.write(
                response["answer"]
            )


            # --------------------------------
            # Performance Metrics
            # --------------------------------

            st.divider()

            st.subheader(
                "⚡ RAG Performance Metrics"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Retrieved Chunks",
                    response.get(
                        "retrieved_chunks",
                        0
                    )
                )


            with col2:

                st.metric(
                    "Retrieval Time",
                    f"{response.get('retrieval_time',0)} sec"
                )


            with col3:

                st.metric(
                    "Generation Time",
                    f"{response.get('generation_time',0)} sec"
                )


            with col4:

                st.metric(
                    "Similarity Score",
                    round(
                        response.get(
                            "top_similarity",
                            0
                        ),
                        3
                    )
                )



            # --------------------------------
            # Query Transformation
            # --------------------------------

            with st.expander(
                "🔄 Query Rewriting Details"
            ):

                st.write(
                    "Original Question:"
                )

                st.write(
                    response.get(
                        "original_question",
                        ""
                    )
                )


                st.write(
                    "Rewritten Question:"
                )

                st.write(
                    response.get(
                        "rewritten_question",
                        ""
                    )
                )



            # --------------------------------
            # Source Documents
            # --------------------------------

            st.divider()

            st.subheader(
                "📚 Retrieved Sources"
            )


            sources = response.get(
                "sources",
                []
            )


            if sources:


                for index, source in enumerate(
                    sources
                ):


                    with st.expander(
                        f"Source Chunk {index+1}"
                    ):


                        st.write(
                            source.text
                        )


                        if hasattr(
                            source,
                            "metadata"
                        ):

                            st.write(
                                "Metadata:"
                            )

                            st.json(
                                source.metadata
                            )


            else:

                st.info(
                    "No source chunks retrieved"
                )



        except Exception as e:


            logger.exception(
                f"Error while answering question: {e}"
            )


            st.error(
                f"Error details: {str(e)}"
            )



# -------------------------------
# Footer
# -------------------------------

st.divider()


st.caption(
    "Powered by OpenAI + LangChain + ChromaDB + RAG"
)