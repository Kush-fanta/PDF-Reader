Here’s a comprehensive `README.md` file for your project. It provides an overview of the project, setup instructions, usage guidelines, and other relevant details.

---

# PDF Question Answering with Hugging Face and Llama-3

This project is a Streamlit-based web application that allows users to upload a PDF file, create vector embeddings using Hugging Face's `sentence-transformers`, and ask questions about the document using the **Llama-3** model via the **Groq API**. The app leverages **LangChain** for document processing, embeddings, and question-answering.

---

## Features

- **PDF Upload**: Users can upload a PDF file for processing.
- **Vector Embeddings**: The app creates vector embeddings for the uploaded PDF using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model.
- **Question Answering**: Users can ask questions about the document, and the app provides answers using the **Llama-3** model via the Groq API.
- **Document Similarity Search**: The app displays the relevant parts of the document that were used to generate the answer.

---

## Prerequisites

Before running the app, ensure you have the following:

1. **Python 3.8 or higher** installed.
2. **Hugging Face API Token**: Obtain your API token from [Hugging Face](https://huggingface.co/settings/tokens).
3. **Groq API Key**: Obtain your API key from [Groq](https://console.groq.com/keys).

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pdf-qa-with-huggingface-llama3.git
   cd pdf-qa-with-huggingface-llama3
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**:
   - Open the `app.py` file and replace the following placeholders with your actual API keys:
     ```python
     groq_api = "your_groq_api_key_here"  # Replace with your Groq API key
     hf_api_key = "your_huggingface_api_key_here"  # Replace with your Hugging Face API key
     ```

---

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Enter API Keys**:
   - When the app launches, you will be prompted to enter your **Hugging Face API Token** and **Groq API Key**.

3. **Upload a PDF**:
   - Use the file uploader to upload a PDF file.

4. **Create Vector Embeddings**:
   - Click the "Create Vector Embeddings" button to process the PDF and generate embeddings.

5. **Ask Questions**:
   - Enter your question in the text input field and press Enter. The app will provide an answer based on the content of the PDF.

6. **View Results**:
   - The app displays the answer and the relevant parts of the document that were used to generate the answer.

---

## Project Structure

```
pdf-qa-with-huggingface-llama3/
├── app.py                  # Main Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # List of dependencies
└── .env                    # Optional: Store API keys (not used in this version)
```

---

## Dependencies

The project uses the following Python libraries:

- **Streamlit**: For building the web interface.
- **LangChain**: For document processing, embeddings, and question-answering.
- **Hugging Face Transformers**: For generating vector embeddings.
- **FAISS**: For efficient similarity search of embeddings.
- **PyPDFLoader**: For loading and processing PDF files.

You can install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Customization

- **Change Embedding Model**:
  - To use a different Hugging Face embedding model, update the `model_name` parameter in the `HuggingFaceEmbeddings` initialization:
    ```python
    embeddings = HuggingFaceEmbeddings(model_name="your-model-name")
    ```

- **Change LLM Model**:
  - To use a different model with the Groq API, update the `model_name` parameter in the `ChatGroq` initialization:
    ```python
    llm = ChatGroq(groq_api_key=groq_api, model_name="your-model-name")
    ```

---
## Acknowledgments

- **Hugging Face**: For providing the `sentence-transformers` models.
- **Groq**: For providing access to the Llama-3 model via their API.
- **LangChain**: For simplifying document processing and question-answering workflows.

---

