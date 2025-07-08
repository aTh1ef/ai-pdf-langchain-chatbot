# app.py
import os
import streamlit as st
import tempfile
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import StdOutCallbackHandler

# Fix for LangSmithTracer import error
try:
    from langchain.callbacks.tracers.langchain import LangChainTracer
    has_langsmith = True
except ImportError:
    has_langsmith = False
    class LangChainTracer:
        def __init__(self, *args, **kwargs):
            pass

# Updated Pinecone import for version compatibility
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("langchain_processes.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("pdf_chatbot")

# Set up environment variables from Streamlit secrets
def setup_environment():
    # Set required API keys
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # Set LangSmith variables if they exist in secrets
    if "LANGCHAIN_API_KEY" in st.secrets:
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "pdf-chatbot")
        os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "true")
        logger.info("LangSmith tracing enabled")
    else:
        logger.info("LangSmith tracing not configured")

# Initialize environment
setup_environment()

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    logger.info("Initialized Pinecone client")
except Exception as e:
    logger.error(f"Error initializing Pinecone: {str(e)}")
    st.error(f"Error initializing Pinecone: {str(e)}")
    pc = None

# Initialize Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class PDFChatbot:
    def __init__(self):
        logger.info("Initializing PDF Chatbot")

        # Initialize components first
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.index_name = "pdf-chatbot"
        
        # Store Pinecone client reference
        self.pc = pc

        # Initialize callbacks list
        callbacks = None

        # Only set up callbacks if LangSmith is properly configured
        if "LANGCHAIN_API_KEY" in os.environ and has_langsmith:
            try:
                tracer = LangChainTracer(
                    project_name=os.environ.get("LANGCHAIN_PROJECT", "pdf-chatbot")
                )
                callbacks = [StdOutCallbackHandler(), tracer]
                logger.info("LangSmith tracing enabled with callback handlers")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith tracing: {e}")
                callbacks = None

        # Initialize LLM
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            logger.info("Successfully initialized LLM with gemini-1.5-flash model")
        except Exception as e:
            logger.warning(f"Error initializing gemini-1.5-flash: {e}")
            try:
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro")
                logger.info("Successfully initialized LLM with gemini-1.0-pro model")
            except Exception as e2:
                logger.warning(f"Error initializing gemini-1.0-pro: {e2}")
                self.llm = ChatGoogleGenerativeAI(model="models/gemini-pro")
                logger.info("Using legacy model name format: models/gemini-pro")

        # Setup Pinecone index
        self._setup_pinecone_index()

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Store callbacks for reuse
        self.callbacks = callbacks

        logger.info("PDF Chatbot initialization complete")

    def _setup_pinecone_index(self):
        """Set up Pinecone index"""
        if self.pc is None:
            logger.error("Pinecone client not initialized")
            st.error("Pinecone client not initialized")
            return
            
        try:
            # Get list of existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error with Pinecone index setup: {e}")
            st.error(f"Error with Pinecone index setup: {e}")

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        logger.info(f"Starting text extraction from PDF: {pdf_file.name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            pdf_path = tmp_file.name

        logger.info(f"Temporary PDF file created: {pdf_path}")

        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            logger.info(f"Extracted {len(documents)} pages from PDF")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            st.error(f"Error extracting text: {str(e)}")
            documents = []
        finally:
            os.unlink(pdf_path)
            logger.info(f"Temporary PDF file deleted")

        return documents

    def chunk_text(self, documents):
        """Split text into manageable chunks"""
        logger.info("Starting text chunking process")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks from documents")

        if chunks:
            logger.info(f"Sample chunk content: {chunks[0].page_content[:100]}...")

        return chunks

    def create_embeddings_and_store(self, chunks, namespace):
        """Create embeddings and store in Pinecone"""
        logger.info(f"Creating embeddings for {len(chunks)} chunks and storing in Pinecone namespace: {namespace}")

        try:
            # Get the index
            index = self.pc.Index(self.index_name)
            
            # Delete existing vectors in this namespace to avoid conflicts
            try:
                index.delete(namespace=namespace, delete_all=True)
                logger.info(f"Deleted existing vectors in namespace: {namespace}")
            except Exception as e:
                logger.warning(f"No existing vectors to delete or error: {str(e)}")
                
            # Create the vector store with LangChain
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace=namespace
            )
            logger.info("Embeddings created and stored successfully")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            st.error(f"Error creating embeddings: {str(e)}")
            raise e

    def get_conversational_chain(self, vectorstore):
        """Create a conversational chain for Q&A"""
        logger.info("Creating conversational retrieval chain")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"
        )

        logger.info("Conversational retrieval chain created")
        return chain

    def process_query(self, chain, query):
        """Process a user query and return a response"""
        logger.info(f"Processing query: {query}")

        try:
            result = chain.invoke({"question": query})
            logger.info("Used chain.invoke() successfully")
        except AttributeError:
            result = chain({"question": query})
            logger.info("Used chain() call successfully")

        logger.info(f"Retrieved {len(result['source_documents'])} source documents")
        logger.info(f"Generated answer (first 100 chars): {result['answer'][:100]}...")

        return result["answer"], result["source_documents"]


# Streamlit UI
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("AI PDF Chatbot")
st.write("Upload a PDF and chat with it!")

# Initialize session state
if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = PDFChatbot()
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        st.error(f"Error initializing chatbot: {str(e)}")
        st.session_state.chatbot = None

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Define the layout
left_col, right_col = st.columns([1, 2])

# Sidebar content
with left_col:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    debug_mode = st.toggle("Debug Mode", value=False)

    if "LANGCHAIN_API_KEY" in st.secrets:
        langsmith_url = "https://smith.langchain.com/projects/" + st.secrets.get("LANGCHAIN_PROJECT", "pdf-chatbot")
        st.markdown(f"[View Traces in LangSmith]({langsmith_url})")

    if uploaded_file and not st.session_state.document_processed and st.session_state.chatbot:
        with st.spinner("Processing document..."):
            status_container = st.empty()

            try:
                # Extract text from PDF
                status_container.info("Extracting text from PDF...")
                documents = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)

                if not documents:
                    status_container.error("Failed to extract text from PDF. Please try another file.")
                else:
                    status_container.success(f"✅ Extracted {len(documents)} pages from PDF")

                    # Chunk text
                    status_container.info("Chunking text...")
                    chunks = st.session_state.chatbot.chunk_text(documents)
                    status_container.success(f"✅ Created {len(chunks)} text chunks")

                    # Create namespace from filename
                    namespace = uploaded_file.name.replace(" ", "_").replace(".", "_").lower()

                    # Create embeddings and store in Pinecone
                    status_container.info("Creating embeddings and storing in Pinecone...")
                    vectorstore = st.session_state.chatbot.create_embeddings_and_store(chunks, namespace)
                    status_container.success("✅ Created embeddings and stored in Pinecone")

                    # Set up conversation chain
                    status_container.info("Setting up conversation chain...")
                    st.session_state.conversation = st.session_state.chatbot.get_conversational_chain(vectorstore)
                    st.session_state.document_processed = True
                    status_container.empty()
                    st.success("Document processed successfully!")
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                status_container.error(f"Error processing document: {str(e)}")

    # Display document stats if processed
    if st.session_state.document_processed:
        st.subheader("Document Information")
        st.info(f"Filename: {uploaded_file.name}")

        # Add reset button
        if st.button("Process Another Document"):
            st.session_state.document_processed = False
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.rerun()

# Chat interface
with right_col:
    st.header("Chat with your PDF")

    if not st.session_state.document_processed:
        st.info("Please upload a PDF document to start chatting.")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

        # User input
        user_query = st.chat_input("Ask a question about your document")

        if user_query:
            st.chat_message("user").write(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Get response from model
            with st.spinner("Thinking..."):
                try:
                    response, sources = st.session_state.chatbot.process_query(st.session_state.conversation, user_query)
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    response = f"I'm sorry, I encountered an error while processing your query: {str(e)}"
                    sources = []

            # Display assistant response
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Display sources
            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"Source {i + 1}:")
                        st.write(source.page_content)
                        st.write(f"Page: {source.metadata.get('page', 'N/A')}")
                        st.divider()

# Debug panel
if debug_mode:
    st.header("Debug Information")

    debug_tabs = st.tabs(["Components", "Process", "Logs", "Pinecone"])

    with debug_tabs[0]:
        st.subheader("LangChain Components")
        st.code("PyMuPDFLoader - PDF text extraction")
        st.code("RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)")
        st.code("HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')")
        st.code(f"PineconeVectorStore(index_name='{st.session_state.chatbot.index_name if st.session_state.chatbot else 'pdf-chatbot'}')")
        st.code("ChatGoogleGenerativeAI(model='gemini-1.5-flash')")
        st.code("ConversationalRetrievalChain")

    with debug_tabs[1]:
        st.subheader("Process Flow")
        st.markdown("""
        **Upload Process:**
        1. PDF Upload → Text Extraction → Text Chunking → Embedding Creation → Vector Storage

        **Query Process:**
        1. Query Embedding → Vector Search → Context Retrieval → LLM Generation
        """)

    with debug_tabs[2]:
        st.subheader("Recent Logs")
        try:
            with open("langchain_processes.log", "r") as log_file:
                logs = log_file.readlines()
                recent_logs = logs[-50:] if len(logs) > 50 else logs
                st.code("".join(recent_logs))
        except:
            st.info("No logs available yet")

    with debug_tabs[3]:
        st.subheader("Pinecone Information")
        if st.session_state.chatbot and st.session_state.chatbot.pc:
            try:
                indexes = [index.name for index in st.session_state.chatbot.pc.list_indexes()]
                st.write("Available Indexes:", indexes)
                st.write("Current Index:", st.session_state.chatbot.index_name)
            except Exception as e:
                st.error(f"Error fetching Pinecone info: {str(e)}")

# Setup help
if not st.session_state.document_processed:
    with st.expander("API Keys Setup"):
        st.markdown("""
        ### Create `.streamlit/secrets.toml` file:

        ```toml
        # Required API keys
        PINECONE_API_KEY = "your-pinecone-api-key"
        GOOGLE_API_KEY = "your-google-api-key"

        # Optional LangSmith configuration
        LANGCHAIN_API_KEY = "your-langsmith-api-key"
        LANGCHAIN_PROJECT = "pdf-chatbot"
        LANGCHAIN_TRACING_V2 = "true"
        ```

        Get your API keys from:
        - [Pinecone Console](https://app.pinecone.io)
        - [Google AI Studio](https://makersuite.google.com/app/apikey)
        - [LangSmith](https://smith.langchain.com) (optional)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
    Built with ❤️ using <strong>Streamlit</strong>, <strong>LangChain</strong>, <strong>Pinecone</strong>, and <strong>Google Gemini</strong>
</div>
""", unsafe_allow_html=True)
