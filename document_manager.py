import streamlit as st
import os
import base64
from rag_backend import (
    check_openai_credentials,
    create_directories,
    process_pdf,
    load_or_initialize_stores,
    generate_multimodal_embeddings,
    save_stores
)
import numpy as np
import json
from dotenv import load_dotenv

load_dotenv()

def get_pdf_download_link(pdf_path):
    """Generate a download link for a PDF file"""
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    return f'data:application/pdf;base64,{b64_pdf}'

def format_doc_name(filename):
    """Format document name for display with UTF-8 support"""
    filename = filename.encode('utf-8').decode('utf-8', errors='replace')
    name = os.path.splitext(filename)[0]
    name = name.replace('_', ' ')
    name = ' '.join(word.capitalize() for word in name.split())
    return name

def show_document_list():
    """Display the list of documents in the sidebar"""
    with st.sidebar:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" style="fill: #0066cc;">
                    <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
                </svg>
                <h3 style="margin: 0;">Document Viewer</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize session states if not exists
        if 'selected_document' not in st.session_state:
            st.session_state.selected_document = None
            
        # Get list of PDF files from data directory
        files = []
        if os.path.exists("data"):
            files = [f for f in os.listdir("data") if f.endswith('.pdf')]
        
        if files:
            # Search box
            search_term = st.text_input("🔍 Search documents", key="doc_search").lower()
            
            # Filter files based on search
            filtered_files = [f for f in files if search_term in f.lower()] if search_term else files
            
            # Display document count
            st.markdown(f"""
                <div style="margin: 10px 0;">
                    Found {len(filtered_files)} document{'s' if len(filtered_files) != 1 else ''}
                </div>
            """, unsafe_allow_html=True)

            # Style for document list
            st.markdown("""
                <style>
                .doc-list {
                    margin-top: 1rem;
                }
                .doc-item {
                    padding: 8px 12px;
                    margin: 4px 0;
                    border: 1px solid #e0e0e0;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .doc-item:hover {
                    border-color: #0066cc;
                    background-color: #f0f7ff;
                }
                .doc-item.selected {
                    border-color: #0066cc;
                    background-color: #e6f3ff;
                    font-weight: 500;
                }
                </style>
            """, unsafe_allow_html=True)

            # Document list container
            st.markdown('<div class="doc-list">', unsafe_allow_html=True)
            
            # Display documents as clickable items
            for file in filtered_files:
                # Create unique key for each button
                button_key = f"doc_btn_{file}"
                
                # Determine if this document is selected
                is_selected = st.session_state.selected_document == file
                
                # Create button with appropriate styling
                if st.button(
                    file,
                    key=button_key,
                    help=f"Click to view {file}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state.selected_document = file
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return st.session_state.selected_document
        else:
            st.info("No documents uploaded yet. Upload PDF files using the uploader above.")
            return None

def main():
    st.set_page_config(layout="wide", page_title="Document Manager")
    
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
    
    create_directories()
    
    # Document Management in Sidebar
    with st.sidebar:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px;">
                <svg width="30" height="30" viewBox="0 0 24 24" style="fill: #0066cc;">
                    <path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/>
                </svg>
                <h2 style="margin: 0; display: inline;">Document Management</h2>
            </div>
        """, unsafe_allow_html=True)

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files"
        )

        # Process uploaded files
        if uploaded_files:
            # Initialize stores
            index, all_items, query_embeddings_cache = load_or_initialize_stores()
            
            for uploaded_file in uploaded_files:
                items, filepath = process_pdf(uploaded_file)
                if items:
                    st.success(f"Processed {uploaded_file.name}")
                    
                    with st.spinner("Generating embeddings..."):
                        for item in items:
                            if item['type'] in ['text', 'table']:
                                item['embedding'] = generate_multimodal_embeddings(prompt=item['text'])
                            else:
                                item['embedding'] = generate_multimodal_embeddings(image=item['image'])
                    
                    new_embeddings = np.array([item['embedding'] for item in items])
                    index.add(np.array(new_embeddings, dtype=np.float32))
                    all_items.extend(items)
                    save_stores(index, all_items, query_embeddings_cache)
    
    # Show document list and get selected document
    selected_file = show_document_list()
    
    # Show PDF viewer in main content area if file is selected
    if selected_file:
        try:
            pdf_path = os.path.join("data", selected_file)
            pdf_link = get_pdf_download_link(pdf_path)
            st.markdown(
                f'<iframe src="{pdf_link}" width="100%" height="800px"></iframe>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
    else:
        st.markdown("""
            ## Welcome to the Document Manager
            
            Upload PDF documents using the sidebar to get started.
            
            Features:
            - Upload multiple PDF documents
            - Search through your documents
            - View documents in the built-in PDF viewer
            - Documents are automatically processed for the RAG system
        """)

if __name__ == "__main__":
    main()