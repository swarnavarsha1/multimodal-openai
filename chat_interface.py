import os
import streamlit as st
import logging
from rag_backend import (
    check_openai_credentials, 
    load_or_initialize_stores,
    generate_multimodal_embeddings,
    invoke_gpt_model,  
    save_stores
)
import numpy as np
import sys
import codecs
from dotenv import load_dotenv

# Ensure UTF-8 encoding for stdout/stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Modify logging configuration:
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_validation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def handle_clear_chat():
    """Clear only the chat history from current session"""
    st.session_state.chat_history = []
    st.rerun()

def generate_and_verify_response(query, matched_items):
    """Generate response without verification"""
    response = invoke_gpt_model(query, matched_items)
    return response

def main():
    st.set_page_config(layout="wide", page_title="Chat Interface")
    
    # Check for OpenAI API key in environment variables
    if 'OPENAI_API_KEY' not in os.environ:
        try:
            os.environ['OPENAI_API_KEY'] = st.secrets['openai_api_key']
        except FileNotFoundError:
            st.error("OpenAI API key not found. Please add it to your .env file or Streamlit secrets.")
            st.stop()
    
    if not check_openai_credentials():
        st.error("OpenAI API key not properly configured.")
        st.stop()
    
    # Custom CSS for styling (remains unchanged)
    st.markdown(
        '<meta charset="UTF-8">',
        unsafe_allow_html=True
    )
    
    st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e6f3ff;
            margin-left: 2rem;
        }
        .assistant-message {
            background-color: #f0f0f0;
            margin-right: 2rem;
        }
        
        /* Hide default Streamlit elements */
        div[data-testid="stToolbar"] {
            display: none;
        }
        button[title="View fullscreen"] {
            display: none;
        }
        
        /* Style for clear button */
        button[data-testid="clear_chat"] {
            background-color: #FF0000 !important;
            color: white !important;
        }
        button[data-testid="clear_chat"]:hover {
            background-color: #CC0000 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Handle clear chat from URL parameter
    if 'clear_chat' in st.query_params:
        handle_clear_chat()
        st.query_params.clear()
    
    # Title and header area with clear button
    col1, col2 = st.columns([6,1])
    with col1:
        st.title("Chat Interface")
    with col2:
        if st.button("🗑️ Clear Chat", key="clear_chat", help="Clear chat history"):
            handle_clear_chat()
    
    # Check OpenAI credentials
    if not check_openai_credentials():
        st.error("OpenAI API key not properly configured. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    # Load stores (but don't initialize new ones)
    index, all_items, query_embeddings_cache = load_or_initialize_stores()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            st.write(message["answer"])
    
    # Chat input and processing
    if query := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if all_items:
                    # Generate query embedding
                    if query in query_embeddings_cache:
                        query_embedding = query_embeddings_cache[query]
                    else:
                        query_embedding = generate_multimodal_embeddings(prompt=query)
                        query_embeddings_cache[query] = query_embedding
                        save_stores(index, all_items, query_embeddings_cache)
                    
                    # Search for relevant content
                    distances, result = index.search(
                        np.array(query_embedding, dtype=np.float32).reshape(1,-1), 
                        k=10
                    )
                    
                    # Get matched items
                    matched_items = [{k: v for k, v in all_items[idx].items() if k != 'embedding'} 
                                   for idx in result.flatten()]
                    
                    # Generate and verify response
                    response = generate_and_verify_response(query, matched_items)
                    st.write(response)
                    
                    # Update chat history
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": response
                    })
                else:
                    st.warning("Please upload some documents first.")

if __name__ == "__main__":
    main()