import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile

# ✅ Streamlit app
st.title("PDF Question Answering with Hugging Face and Llama-3")

# ✅ User input for API keys
st.subheader("Enter API Keys")
hf_api_key = st.text_input("Enter your Hugging Face API Token", type="password")
groq_api = st.text_input("Enter your Groq API Key", type="password")

# ✅ Check if API keys are provided
if not hf_api_key or not groq_api:
    st.warning("Please enter both Hugging Face and Groq API keys to proceed.")
    st.stop()  # Stop the app if API keys are not provided

# Set Hugging Face API key as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# ✅ Initialize the Groq LLM
try:
    llm = ChatGroq(groq_api_key=groq_api, model_name="llama-3.1-8b-instant")
except Exception as e:
    st.error(f"Failed to initialize Groq LLM: {e}")
    st.stop()

# ✅ Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}
    """
)

# ✅ Function to create vector embeddings from a PDF file
def create_vector_embeddings(pdf_file):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load the PDF file
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        # Initialize Hugging Face Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Convert document text into a list before passing to FAISS
        texts = [doc.page_content for doc in final_documents]

        # Store embeddings in FAISS
        vectors = FAISS.from_texts(texts, embeddings)

        # Save in session state
        st.session_state.vectors = vectors
        st.session_state.final_documents = final_documents

        os.unlink(tmp_file_path)  # Cleanup temp file
    except Exception as e:
        st.error(f"Failed to create vector embeddings: {e}")
        st.stop()

# ✅ File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# ✅ Button to create vector embeddings
if uploaded_file and st.button("Create Vector Embeddings"):
    create_vector_embeddings(uploaded_file)
    st.write("✅ Vector Database is ready!")

# ✅ User input for questions
user_prompt = st.text_input("Enter your question about the document")

# ✅ Process the question if vector database is ready
if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # ✅ Get the response
    response = retrieval_chain.invoke({'input': user_prompt})
    st.write("Answer:", response['answer'])

    # ✅ Display document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write("-------------------------------")