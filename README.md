# PDF Chatbot

A powerful AI-powered langchain chatbot that allows you to upload PDF documents and have conversations about their contents.



## 📝 Description

This application enables users to upload PDF documents and engage in natural language conversations about the content of those documents. The system utilizes advanced natural language processing to understand user queries and retrieve relevant information from the processed documents.

🌐 Live Demo:
Try out the live demo of this application at:
https://ai-pdf-langchain-chatbot-app.streamlit.app/

## 🔄 Workflow

The application follows this workflow:

1. **PDF Upload**
   - User uploads a PDF document through the Streamlit interface
   - The document is temporarily saved and processed

2. **Text Extraction**
   - PyMuPDFLoader extracts text and metadata from the PDF
   - Each page is processed into a document object with content and metadata

3. **Text Chunking**
   - The extracted text is divided into smaller, manageable chunks
   - RecursiveCharacterTextSplitter ensures chunks maintain context with proper overlap

4. **Vector Embedding**
   - Each text chunk is converted into a vector embedding using HuggingFace's sentence transformer
   - These embeddings capture the semantic meaning of the text chunks

5. **Vector Storage**
   - The embeddings are stored in Pinecone, a vector database
   - Each document's chunks are stored under a unique namespace based on the filename

6. **Query Processing**
   - User queries are converted to the same vector space
   - Pinecone finds the most semantically similar chunks from the document
   - The LLM generates responses based on the retrieved context and conversation history

7. **Response Generation**
   - Google's Gemini model processes the retrieved chunks and user query
   - The system maintains conversation memory to provide contextual responses
   - Source documents are displayed for transparency

## 🛠️ Technology Stack

### Core Components

- **Streamlit**: Web application framework for the user interface
- **LangChain**: Framework for building applications with large language models
- **Pinecone**: Vector database for storing and retrieving embeddings
- **Google Gemini**: Large language model for generating responses
- **HuggingFace Embeddings**: For converting text to vector representations

### Document Processing

- **PyMuPDFLoader**: Extracts text and metadata from PDF documents
- **RecursiveCharacterTextSplitter**: Splits text into chunks while preserving context

### Conversational AI

- **ConversationalRetrievalChain**: LangChain component that combines document retrieval with conversation history
- **ConversationBufferMemory**: Maintains conversation context for more coherent responses

### Monitoring & Debugging

- **LangSmith Integration**: Optional tracing and monitoring for LangChain components
- **Logging System**: Comprehensive logging for debugging and monitoring
- **Debug Mode**: Built-in UI for examining application components and processes

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys (see API Keys section below)

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🔑 API Keys Setup

The application requires API keys from several services:

### Required API Keys

1. **Pinecone API Key**
   - Sign up at [Pinecone](https://app.pinecone.io)
   - Create a new project and index (if not already done)
   - Configure your index with:
     - Dimensions: 384
     - Metric: Cosine
     - Cloud Provider: AWS
     - Region: us-east-1
     - Capacity mode: Serverless
   - Copy your API key from the console

2. **Google Gemini API Key**
   - Sign up for [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the API key for use in the application

3. **Pinecone Host URL**
   - This is the endpoint URL for your Pinecone index
   - Format: `your-index-name-xxxx.svc.region.pinecone.io`
   - Find this in your Pinecone console after creating an index

### Optional API Key

4. **LangSmith API Key** (Optional, for advanced monitoring)
   - Sign up at [LangSmith](https://smith.langchain.com)
   - Create a new project and get your API key
   - This enables detailed tracing and debugging of LangChain components

### Setting up secrets.toml

Create a `.streamlit` directory in your project folder and add a `secrets.toml` file with the following content:

```toml
# Required API keys
PINECONE_API_KEY = "your-pinecone-api-key"
GOOGLE_API_KEY = "your-google-api-key"
PINECONE_HOST = "your-index-name-xxxx.svc.region.pinecone.io"

# Optional LangSmith configuration (for advanced monitoring)
LANGCHAIN_API_KEY = "your-langsmith-api-key"
LANGCHAIN_PROJECT = "pdf-chatbot"
LANGCHAIN_TRACING_V2 = "true"
```

## 💡 Features

- **File Upload**: Easily upload PDF documents through the UI
- **Interactive Chat**: Have natural conversations about document content
- **Source Citations**: View the exact source of information from your documents
- **Debug Mode**: Toggle detailed debugging information for developers
- **LangSmith Integration**: Monitor and trace operations through LangSmith (optional)

## ⚠️ Limitations

- Currently only supports PDF files
- Large PDFs may take longer to process
- Response quality depends on the clarity and structure of the original document

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com)
- Vector embeddings powered by [Pinecone](https://www.pinecone.io)
- LLM capabilities provided by [Google Gemini](https://ai.google.dev/)
- PDF processing via [PyMuPDF](https://pymupdf.readthedocs.io)
