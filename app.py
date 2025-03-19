import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import BedrockLLM
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# =======================
# AWS Credentials Config
# =======================
# Configure your AWS credentials here or in .env file
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Styling
st.markdown("""
<style>
    .main { background-color: #1a1a1a; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #2d2d2d; }
    .stTextInput textarea { color: #ffffff !important; background-color: #3d3d3d; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stButton>button:hover { background-color: #45a049; }
    .chat-message { padding: 10px; border-radius: 5px; margin: 5px 0; color: #000000; }
    .user-message { background-color: #d3d3d3; }
    .assistant-message { background-color: #e0e0e0; }
    .password-input input { background-color: #3d3d3d !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# Constants
PROMPT_TEMPLATE = """
You are a highly trained medical assistant designed to help users identify potential causes of their symptoms through a conversational, step-by-step process. Your goal is to ask focused questions, narrow down possibilities, and provide clear, concise guidance based on the user's input.

- Begin by asking about the user's symptoms if no specific details are provided.
- For each symptom or detail shared, ask targeted follow-up questions to refine the possibilities (e.g., duration, severity, location).
- Use the provided context to inform your responses, but keep the conversation natural and avoid medical jargon unless necessary.
- If the user's input is vague, ask for clarification rather than guessing.
- After 3-4 exchanges or when sufficient details are provided, summarize the likely causes and ask: "Could you tell me which area of Chennai you live in? I can suggest a nearby doctor to consult."
- Stay empathetic, concise, and avoid overwhelming the user with too many questions at once.

Chat History:
{chat_history}

User Query: {user_query}
Context: {document_context}
Answer:
"""

DEFAULT_KNOWLEDGE_BASE = "knowledge_base.txt"
DOCTOR_LIST_FILE = "doctor_list.txt"

# AWS Credentials setup function
def setup_aws_credentials():
    if 'aws_credentials_configured' not in st.session_state:
        try:
            os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
            os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
            os.environ["AWS_REGION"] = AWS_REGION

            # Validate credentials with a simple boto3 operation
            boto3.client('sts').get_caller_identity()
            st.session_state.aws_credentials_configured = True
            st.success("AWS credentials configured successfully!")
        except Exception as e:
            st.error(f"AWS credentials validation failed: {str(e)}")
            st.session_state.aws_credentials_configured = False
    return st.session_state.aws_credentials_configured

# Initialize AWS Bedrock models if credentials are configured
def initialize_bedrock_models():
    if 'bedrock_models_initialized' not in st.session_state or not st.session_state.bedrock_models_initialized:
        try:
            # Create Bedrock session with credentials
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv("AWS_REGION", AWS_REGION),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            # Initialize models with client
            embedding_model = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v1",
                client=bedrock_client
            )
            
            # Initialize language model with additional parameters for context length
            language_model = BedrockLLM(
                model_id="meta.llama3-70b-instruct-v1:0",
                client=bedrock_client,
                model_kwargs={
                    "max_tokens": 4096,   
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            
            st.session_state.embedding_model = embedding_model
            st.session_state.language_model = language_model
            st.session_state.bedrock_client = bedrock_client
            st.session_state.bedrock_models_initialized = True
            
            # Initialize vector database
            document_vector_db = InMemoryVectorStore(embedding_model)
            st.session_state.document_vector_db = document_vector_db
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize Bedrock models: {str(e)}")
            st.session_state.bedrock_models_initialized = False
            return False
    return True

# Load and chunk knowledge base
def load_and_chunk_knowledge_base(file_path=DEFAULT_KNOWLEDGE_BASE, chunk_size=1000, chunk_overlap=200):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Knowledge base file '{file_path}' not found.")
        
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        if not content.strip():
            raise ValueError("Knowledge base is empty.")
        
        raw_document = Document(page_content=content)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        document_chunks = text_splitter.split_documents([raw_document])
        
        st.session_state.document_vector_db.add_documents(document_chunks)
        st.success(f"Loaded and indexed {len(document_chunks)} chunks from '{file_path}'.")
        st.session_state.vector_store_initialized = True
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        st.session_state.vector_store_initialized = False

# Load doctor list with ratings
def load_doctor_list(file_path=DOCTOR_LIST_FILE):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            doctors = [line.strip().split(" | ") for line in file.readlines() if line.strip()]
        return [{"name": d[0], "specialty": d[1], "area": d[2], "rating": float(d[3])} for d in doctors]
    except Exception as e:
        st.error(f"Error loading doctor list: {str(e)}")
        return []

# Find nearby doctors based on user location with ratings
def find_nearby_doctors(user_area, doctor_list):
    user_area = user_area.lower().strip()
     
    if "adayar" in user_area or "adyer" in user_area:
        user_area = "adyar"
    nearby_doctors = [doc for doc in doctor_list if user_area in doc["area"].lower()]
    if not nearby_doctors:
        # Sort by rating as fallback
        nearby_doctors = sorted(doctor_list, key=lambda x: x["rating"], reverse=True)[:3]
        return nearby_doctors, "I couldn't find doctors exactly in your area, so here are some top-rated recommendations in Chennai:"
    # Sort nearby doctors by rating
    nearby_doctors = sorted(nearby_doctors, key=lambda x: x["rating"], reverse=True)[:3]
    return nearby_doctors, f"Here are some top-rated doctors near {user_area}:"

def find_related_documents(query, k=8):   
    try:
        return st.session_state.document_vector_db.similarity_search(query, k=k)
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def generate_answer(user_query, context_documents, chat_history):
    try:
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        full_chat_history = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history])
        
        conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        response_chain = conversation_prompt | st.session_state.language_model
        response = response_chain.invoke({
            "user_query": user_query,
            "document_context": context_text,
            "chat_history": full_chat_history
        })
        
        response = re.sub(r"<[^>]+>", "", response).strip()
        return response
    except Exception as e:
        return f"Sorry, I encountered an issue: {str(e)}. Please try again."

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "awaiting_location" not in st.session_state:
    st.session_state.awaiting_location = False

# UI
st.title("🧠 MedGPT")
st.caption("🚀 Your conversational guide to symptom analysis and doctor recommendation")
st.markdown("---")

# Check and setup AWS credentials
credentials_configured = setup_aws_credentials()

if credentials_configured:
    # Initialize Bedrock models
    models_initialized = initialize_bedrock_models()
    
    if models_initialized:
        # Sidebar for settings
        with st.sidebar:
            st.header("Settings")
            # Updated model selection list with Llama and Mistral models
            selected_model = st.selectbox(
                "Choose Model", 
                [
                    "meta.llama3-70b-instruct-v1:0",
                    "meta.llama3-1-70b-instruct-v1:0", 
                    "mistral.mixtral-8x7b-instruct-v0:1",
                    "mistral.mistral-7b-instruct-v0:2",
                    "deepseek.r1-v1:0"
                ], 
                index=0
            )
            
            # Update model based on selection
            if st.button("Update Model"):
                try:
                    st.session_state.language_model = BedrockLLM(
                        model_id=selected_model,
                        client=st.session_state.bedrock_client,
                        model_kwargs={
                            "max_tokens": 4096,   
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    )
                    st.success(f"Model updated to {selected_model}")
                except Exception as e:
                    st.error(f"Failed to update model: {str(e)}")
            
            st.divider()
            chunk_size = st.slider("Chunk Size", 100, 2000, 500, 100)   
            chunk_overlap = st.slider("Chunk Overlap", 10, 500, 100, 50)  
            
            if st.button("Reload Knowledge Base"):
                load_and_chunk_knowledge_base(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.awaiting_location = False
                st.rerun()

        # Initialize vector store and doctor list if needed
        if "vector_store_initialized" not in st.session_state or not st.session_state.vector_store_initialized:
            load_and_chunk_knowledge_base()
        if "doctor_list" not in st.session_state:
            st.session_state["doctor_list"] = load_doctor_list()

        # Display chat history
        for chat in st.session_state.chat_history:
            with st.container():
                col1, col2 = st.columns([1, 9])
                with col1:
                    st.markdown("*You:*")
                with col2:
                    st.markdown(f'<div class="chat-message user-message">{chat["user"]}</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 9])
                with col1:
                    st.markdown("*Assistant:*")
                with col2:
                    st.markdown(f'<div class="chat-message assistant-message">{chat["assistant"]}</div>', unsafe_allow_html=True)

        # User input
        user_input = st.chat_input("Tell me about your symptoms...")

        if user_input:
            with st.container():
                col1, col2 = st.columns([1, 9])
                with col1:
                    st.markdown("*You:*")
                with col2:
                    st.markdown(f'<div class="chat-message user-message">{user_input}</div>', unsafe_allow_html=True)
            
            with st.spinner("🧠 Thinking..."):
                if st.session_state.awaiting_location:
                    # User provided location
                    nearby_doctors, message = find_nearby_doctors(user_input, st.session_state.doctor_list)
                    ai_response = f"{message}\n\n" + "\n".join([f"- Dr. {doc['name']} ({doc['specialty']}) in {doc['area']}" for doc in nearby_doctors])
                    ai_response += "\n\nPlease consult a healthcare professional for a proper diagnosis."
                    st.session_state.awaiting_location = False
                else:
                    # Process symptoms or regular query
                    relevant_docs = find_related_documents(user_input)
                    ai_response = generate_answer(user_input, relevant_docs, st.session_state.chat_history)
                    if "Could you tell me which area of Chennai you live in?" in ai_response:
                        st.session_state.awaiting_location = True
            
            with st.container():
                col1, col2 = st.columns([1, 9])
                with col1:
                    st.markdown("*Assistant:*")
                with col2:
                    st.markdown(f'<div class="chat-message assistant-message">{ai_response}</div>', unsafe_allow_html=True)
            
            st.session_state.chat_history.append({"user": user_input, "assistant": ai_response})
    else:
        st.warning("Please configure AWS Bedrock models properly. Check the console for errors.")
else:
    st.info("AWS credentials are not configured properly. Please check your configuration.")