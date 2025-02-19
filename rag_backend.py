import os
import boto3
import faiss
import base64
import fitz as pymupdf
import logging
import warnings
import pickle
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tabula
import json
from botocore.exceptions import ClientError
import xlrd
from openpyxl import load_workbook
import pandas as pd
import numpy as np

# Constants
BASE_DIR = "data"
VECTOR_STORE = "vector_store"
FAISS_INDEX = "faiss.index"
ITEMS_PICKLE = "items.pkl"
QUERY_EMBEDDINGS_CACHE = "query_embeddings.pkl"

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def create_directories():
    """Create necessary directories for storing data"""
    dirs = [BASE_DIR, VECTOR_STORE]
    subdirs = ["images", "text", "tables", "page_images"]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(BASE_DIR, subdir), exist_ok=True)

def process_tables(doc, page_num, items, filepath):
    """Process tables with better table handling"""
    try:
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            # Skip empty tables
            if table.empty:
                continue
                
            # Clean table data
            table = table.fillna('')  # Handle NaN values
            
            # Create a more readable markdown table
            headers = table.columns.tolist()
            markdown_rows = []
            
            # Add headers
            markdown_rows.append("| " + " | ".join(str(h) for h in headers) + " |")
            markdown_rows.append("| " + " | ".join(['---' for _ in headers]) + " |")
            
            # Add data rows
            for _, row in table.iterrows():
                markdown_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
            table_text = f"### Table {table_idx + 1}\n" + "\n".join(markdown_rows)
            
            table_file_name = os.path.join(BASE_DIR, "tables", 
                f"{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt")
                
            with open(table_file_name, 'w', encoding='utf-8') as f:
                f.write(table_text)
                
            items.append({
                "page": page_num,
                "type": "table",
                "text": table_text,
                "path": table_file_name,
                "raw_table": table.to_dict('records')
            })
    except Exception as e:
        logger.warning(f"Table processing error on page {page_num + 1}: {str(e)}")

def process_text_chunks(text, text_splitter, page_num, items, filepath):
    """Enhanced text processing with better structure preservation"""
    import re
    
    # Document structure patterns
    patterns = {
        'heading': re.compile(r'^(?:#{1,6}|\d+\.|[A-Z][^.]+:)\s*(.+)$', re.MULTILINE),
        'bullet_list': re.compile(r'^\s*[-*•]\s+(.+)$', re.MULTILINE),
        'numbered_list': re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),
        'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
        'table': re.compile(r'^\s*\|(?:[^|]+\|)+\s*$', re.MULTILINE)
    }
    
    def extract_structure(text_block):
        """Extract structural elements while preserving their exact format"""
        elements = []
        current_element = {'type': 'text', 'content': []}
        lines = text_block.split('\n')
        code_block = False
        
        for line in lines:
            # Handle code blocks
            if line.startswith('```'):
                if code_block:
                    current_element['content'].append(line)
                    elements.append(current_element)
                    current_element = {'type': 'text', 'content': []}
                    code_block = False
                else:
                    if current_element['content']:
                        elements.append(current_element)
                    current_element = {'type': 'code', 'content': [line]}
                    code_block = True
                continue
                
            if code_block:
                current_element['content'].append(line)
                continue
                
            # Check for structural elements
            for elem_type, pattern in patterns.items():
                if pattern.match(line):
                    if current_element['content']:
                        elements.append(current_element)
                    current_element = {'type': elem_type, 'content': [line]}
                    break
            else:
                if line.strip():
                    current_element['content'].append(line)
                elif current_element['content']:
                    elements.append(current_element)
                    current_element = {'type': 'text', 'content': []}
        
        if current_element['content']:
            elements.append(current_element)
            
        return elements
    
    def save_element(element, section_num):
        """Save a structural element while preserving its format"""
        content = '\n'.join(element['content'])
        file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_{element['type']}_{page_num}_{section_num}.txt"
        
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content)
            
        metadata = {
            'type': element['type'],
            'has_code': element['type'] == 'code',
            'has_list': element['type'] in ['bullet_list', 'numbered_list'],
            'has_table': element['type'] == 'table',
            'is_heading': element['type'] == 'heading'
        }
        
        return {
            "page": page_num,
            "type": "text",
            "text": content,
            "path": file_name,
            "metadata": metadata
        }
    
    try:
        # First extract structural elements
        elements = extract_structure(text)
        
        # Save elements while preserving their structure
        for i, element in enumerate(elements):
            item = save_element(element, i)
            items.append(item)
        
        # Process any remaining text traditionally
        remaining_text = text_splitter.split_text(text)
        for i, chunk in enumerate(remaining_text):
            # Only save chunks that aren't part of structural elements
            if not any(chunk in elem['content'] for elem in elements):
                text_file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
                with open(text_file_name, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                items.append({
                    "page": page_num,
                    "type": "text",
                    "text": chunk,
                    "path": text_file_name,
                    "metadata": {'type': 'text'}
                })
                
    except Exception as e:
        logger.error(f"Error processing text chunks on page {page_num}: {str(e)}")
        # Fall back to basic processing
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            text_file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
            with open(text_file_name, 'w', encoding='utf-8') as f:
                f.write(chunk)
            items.append({
                "page": page_num,
                "type": "text",
                "text": chunk,
                "path": text_file_name
            })

def process_images(page, page_num, items, filepath, doc):
    """Process images from PDF pages"""
    images = page.get_images()
    for idx, image in enumerate(images):
        try:
            xref = image[0]
            pix = pymupdf.Pixmap(doc, xref)
            
            # Improve image quality by converting to RGB if needed
            if pix.n - pix.alpha < 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                
            image_name = os.path.join(BASE_DIR, "images", 
                f"{os.path.basename(filepath)}_image_{page_num}_{idx}_{xref}.png")
            
            # Save image without quality parameter
            pix.save(image_name)
            
            with open(image_name, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf8')
            items.append({
                "page": page_num,
                "type": "image",
                "path": image_name,
                "image": encoded_image
            })
        except Exception as e:
            logger.warning(f"Image processing error on page {page_num + 1}, image {idx}: {str(e)}")
            continue

def process_page_images(page, page_num, items, filepath):
    """Process full page images"""
    pix = page.get_pixmap()
    page_path = os.path.join(BASE_DIR, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, 'rb') as f:
        page_image = base64.b64encode(f.read()).decode('utf8')
    items.append({"page": page_num, "type": "page", "path": page_path, "image": page_image})

def process_pdf(uploaded_file):
    """Process uploaded PDF file and extract content"""
    if uploaded_file is None:
        return None, None
    
    filepath = os.path.join(BASE_DIR, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    doc = pymupdf.open(filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)
    items = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        process_tables(doc, page_num, items, filepath)
        process_text_chunks(text, text_splitter, page_num, items, filepath)
        process_images(page, page_num, items, filepath, doc)
        process_page_images(page, page_num, items, filepath)

    return items, filepath

def preprocess_excel_data(df):
    """
    Preprocess Excel data for better embeddings, handling various data types
    and structures generically
    """
    processed_items = []
    
    # Get column types for better handling
    column_types = df.dtypes.to_dict()
    
    # Process each row
    for idx, row in df.iterrows():
        fields = []
        
        for col, value in row.items():
            # Skip empty values
            if pd.isna(value):
                continue
                
            # Handle different data types
            if pd.api.types.is_numeric_dtype(column_types[col]):
                # Format numbers without scientific notation and with reasonable precision
                if isinstance(value, (int, np.integer)):
                    formatted_value = str(value)
                else:
                    formatted_value = f"{value:.4f}".rstrip('0').rstrip('.')
                fields.append(f"{col}: {formatted_value}")
                
            elif pd.api.types.is_datetime64_any_dtype(column_types[col]):
                # Format datetime consistently
                formatted_value = pd.to_datetime(value).strftime('%Y-%m-%d %H:%M:%S')
                fields.append(f"{col}: {formatted_value}")
                
            elif pd.api.types.is_categorical_dtype(column_types[col]):
                # Handle categorical data
                fields.append(f"{col}: {str(value)}")
                
            else:
                # Handle text and other types
                # Clean and normalize text
                cleaned_value = str(value).strip()
                if cleaned_value:
                    fields.append(f"{col}: {cleaned_value}")
        
        # Create semantic text that preserves column relationships
        semantic_text = ". ".join(fields)
        
        # Add metadata about the data types present in this row
        data_types = {
            col: {
                'type': str(dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(dtype),
                'is_categorical': pd.api.types.is_categorical_dtype(dtype),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(dtype),
                'is_text': pd.api.types.is_string_dtype(dtype)
            }
            for col, dtype in column_types.items()
            if not pd.isna(row[col])
        }
        
        processed_items.append({
            'text': semantic_text,
            'row_index': idx,
            'original_data': row.to_dict(),
            'data_types': data_types
        })
    
    return processed_items

def process_excel(uploaded_file):
    """Process uploaded Excel file with enhanced data handling"""
    if uploaded_file is None:
        return None, None
    
    filepath = os.path.join(BASE_DIR, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    items = []
    excel_file = pd.ExcelFile(filepath)
    
    for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
        # Read the sheet with appropriate data types
        df = pd.read_excel(
            filepath, 
            sheet_name=sheet_name,
            parse_dates=True,  # Automatically parse dates
            na_filter=True     # Handle missing values
        )
        
        # Convert appropriate columns to categorical
        for col in df.select_dtypes(include=['object']):
            # If column has low cardinality (few unique values), make it categorical
            if df[col].nunique() < len(df) * 0.5:  # If unique values are less than 50% of rows
                df[col] = df[col].astype('category')
        
        # Process data with type awareness
        processed_items = preprocess_excel_data(df)
        
        # Create the markdown table format for display
        headers = df.columns.tolist()
        table_data = []
        table_data.append("| " + " | ".join(str(h) for h in headers) + " |")
        table_data.append("| " + " | ".join(['---' for _ in headers]) + " |")
        
        for _, row in df.iterrows():
            formatted_row = []
            for col in headers:
                value = row[col]
                if pd.isna(value):
                    formatted_row.append('')
                elif pd.api.types.is_numeric_dtype(df[col].dtype):
                    if isinstance(value, (int, np.integer)):
                        formatted_row.append(str(value))
                    else:
                        formatted_row.append(f"{value:.4f}".rstrip('0').rstrip('.'))
                else:
                    formatted_row.append(str(value))
            table_data.append("| " + " | ".join(formatted_row) + " |")
        
        table_text = f"### Sheet: {sheet_name}\n" + "\n".join(table_data)
        
        # Save the table view
        table_file_name = os.path.join(BASE_DIR, "tables", 
            f"{os.path.basename(filepath)}_sheet_{sheet_idx}.txt")
            
        with open(table_file_name, 'w', encoding='utf-8') as f:
            f.write(table_text)
        
        # Add the table view
        items.append({
            "page": sheet_idx,
            "type": "table",
            "text": table_text,
            "path": table_file_name,
            "sheet_name": sheet_name
        })
        
        # Add processed items for semantic search
        for processed_item in processed_items:
            items.append({
                "page": sheet_idx,
                "type": "text",
                "text": processed_item['text'],
                "path": table_file_name,
                "metadata": {
                    "type": "excel_row",
                    "row_index": processed_item['row_index'],
                    "original_data": processed_item['original_data'],
                    "data_types": processed_item['data_types']
                }
            })
    
    return items, filepath

def process_document(uploaded_file):
    """Process uploaded document (PDF or Excel)"""
    if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
        return process_excel(uploaded_file)
    else:
        return process_pdf(uploaded_file)

def generate_multimodal_embeddings(prompt=None, image=None, output_embedding_length=384):
    """Generate embeddings using AWS Bedrock"""
    if not prompt and not image:
        raise ValueError("Please provide either a text prompt, base64 image, or both as input")
    
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )
    model_id = "amazon.titan-embed-image-v1"
    
    body = {"embeddingConfig": {"outputEmbeddingLength": output_embedding_length}}
    if prompt:
        body["inputText"] = prompt
    if image:
        body["inputImage"] = image

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        result = json.loads(response.get("body").read())
        return result.get("embedding")
    except ClientError as err:
        logger.error(f"Error generating embeddings: {str(err)}")
        return None

def load_or_initialize_stores():
    """Load or initialize vector store and cache with efficient large-scale search"""
    embedding_vector_dimension = 384
    
    if os.path.exists(os.path.join(VECTOR_STORE, FAISS_INDEX)):
        index = faiss.read_index(os.path.join(VECTOR_STORE, FAISS_INDEX))
        with open(os.path.join(VECTOR_STORE, ITEMS_PICKLE), 'rb') as f:
            all_items = pickle.load(f)
            for item in all_items:
                if 'text' in item:
                    item['text'] = item['text'].encode('utf-8').decode('utf-8', errors='replace')
    else:
        # Initialize with HNSW index for efficient large-scale search
        index = faiss.IndexHNSWFlat(embedding_vector_dimension, 32)  # 32 neighbors per node
        # Configure for better recall vs speed tradeoff
        index.hnsw.efConstruction = 40  # Higher value = better accuracy, slower construction
        index.hnsw.efSearch = 32  # Higher value = better accuracy, slower search
        all_items = []
    
    query_cache_path = os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE)
    if os.path.exists(query_cache_path):
        with open(query_cache_path, 'rb') as f:
            query_embeddings_cache = pickle.load(f)
            query_embeddings_cache = {
                k.encode('utf-8').decode('utf-8', errors='replace'): v 
                for k, v in query_embeddings_cache.items()
            }
    else:
        query_embeddings_cache = {}
    
    return index, all_items, query_embeddings_cache

def save_stores(index, all_items, query_embeddings_cache):
    """Save vector store and cache with UTF-8 support"""
    os.makedirs(VECTOR_STORE, exist_ok=True)
    
    faiss.write_index(index, os.path.join(VECTOR_STORE, FAISS_INDEX))
    
    # Ensure UTF-8 encoding for text content before saving
    items_to_save = []
    for item in all_items:
        item_copy = item.copy()
        if 'text' in item_copy:
            item_copy['text'] = item_copy['text'].encode('utf-8').decode('utf-8', errors='replace')
        items_to_save.append(item_copy)
    
    with open(os.path.join(VECTOR_STORE, ITEMS_PICKLE), 'wb') as f:
        pickle.dump(items_to_save, f)
    
    # Ensure UTF-8 encoding for cache before saving
    cache_to_save = {
        k.encode('utf-8').decode('utf-8', errors='replace'): v 
        for k, v in query_embeddings_cache.items()
    }
    
    with open(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE), 'wb') as f:
        pickle.dump(cache_to_save, f)

def invoke_gpt_model(prompt, matched_items):
    """Generate response using GPT-4 with integrated natural interaction and document accuracy."""
    try:
        system_msg = """You are a helpful and highly intelligent assistant document-based assistant that combines natural conversation. Your role is to extract and provide information only from the uploaded documents with zero hallucination. Follow these strict guidelines:

1. STRICT DOCUMENT RELIANCE
- Use ONLY information directly from the provided documents
- Never add external knowledge, assumptions, or guesses
- If information is not found, respond EXACTLY with: "I don't have any information about that in the provided documents."
- No suggestions, inferences, or external explanations

2. NATURAL LANGUAGE UNDERSTANDING
- Match different phrasings of the same question
- Identify synonyms and related terms
- Connect semantically similar content
- Understand context and implied relationships

3. HANDLING MISSING INFORMATION
- Provide only information that exists in documents
- No explanations of missing content
- No related or similar information suggestions
- If partial answer exists, provide only that part

4. DOCUMENT STRUCTURE PRESERVATION
- Keep ALL original formatting intact:
  * Tables exactly as presented
  * Bullet points and numbering
  * Headings and subheadings
  * Special characters and notation
- Use proper markdown for structured content
- Maintain domain-specific terminology exactly

5. DOMAIN ADAPTATION
- No assumptions about document domain
- Process all content types consistently
- Maintain technical accuracy across domains

6. RESPONSE QUALITY
- Clear, structured responses
- Appropriate formatting (lists, tables, paragraphs)
- Cite sources only at the end under References section
- Consistent citation format: "Source: filename, page X"

7. MULTIMODAL HANDLING
- Reference images and charts explicitly
- Describe visual content when relevant
- Maintain table structures
- Indicate when answer comes from visual content

🚨 CRITICAL RULES
- ZERO information generation
- NO external knowledge
- EXACT document adherence
- If unsure: "I don't have any information about that in the provided documents."
- DO NOT include source citations within the response text
"""
    
        if not matched_items:
            return "I don't have any information about that in the provided documents."
            
        # Initialize the chat model with specific parameters
        chat = ChatOpenAI(
            model="gpt-4o",
            max_tokens=1500,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )
            
        # Organize matched items and track sources with their content
        message_content = []
        source_content_map = {}  # Map to track which content came from which source
        context_window = []  # Store context from previous messages
        
        # Sort matched items by relevance
        for item in matched_items:
            source_file = os.path.basename(item['path']).split('_')[0]
            page_num = item['page'] + 1
            source_info = f"[Source: {source_file}, page {page_num}]"
            
            if item['type'] == 'text':
                # Clean up number and price formatting
                text = re.sub(r'(\$\d+\.?\d*)\s+', r'\1 ', item['text'])
                text = re.sub(r'(\d+\.?\d*)\s+', r'\1 ', text)
                
                # Clean up formatting
                # Remove excessive newlines
                text = re.sub(r'\n{3,}', '\n\n', text)
                # Ensure consistent bullet point formatting
                text = re.sub(r'(?m)^[•○]\s*', '- ', text)
                # Ensure consistent spacing after colons
                text = re.sub(r':\n(?!\n)', ': ', text)
                
                # Add context about document structure
                if 'metadata' in item and item['metadata'].get('is_heading'):
                    context_window.append(f"Section: {text}")
                
                # Add text without source info in the content
                message_content.append({
                    "type": "text",
                    "text": text,
                    "source": source_info,
                    "content_hash": hash(text.strip())  # Used to track which content is actually used
                })
                source_content_map[hash(text.strip())] = (source_info, page_num)
                
            elif item['type'] == 'table':
                # Add table without source info in the content
                message_content.append({
                    "type": "text",
                    "text": item['text'],
                    "source": source_info,
                    "content_hash": hash(item['text'].strip())
                })
                source_content_map[hash(item['text'].strip())] = (source_info, page_num)
                
            elif item['type'] in ['image', 'page']:
                image_url = f"data:image/png;base64,{item['image']}"
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
                source_content_map[hash(item['image'])] = (source_info, page_num)

        # Add context window to message content
        if context_window:
            message_content.insert(0, {
                "type": "text",
                "text": "Document Context:\n" + "\n".join(context_window)
            })

        # Add query context with enhanced instructions
        query_context = (
            f"\nQuestion: {prompt}\n\n"
            "Please provide a response based on the above content, considering:\n"
            "1. Direct matches to the query\n"
            "2. Related information that provides context\n"
            "3. Similar items or concepts that might be relevant\n"
            "4. Document structure and organization\n"
            "Include only information from the provided documents and list all sources under References at the end."
        )
        message_content.append({
            "type": "text",
            "text": query_context
        })

        # Prepare messages for the chat model
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=message_content)
        ]

        # Get response from the model
        response = chat.invoke(messages)
        response_content = response.content

        # Ensure response has References section at the end with all sources
        if "I don't have any information about that in the provided documents" not in response_content:
            # Remove any existing references sections and generic source mentions
            response_content = re.sub(r'\n\n(?:References:|Source:).*?(?=\n\n|$)', '', response_content, flags=re.DOTALL)
            response_content = response_content.strip()
            
            # Extract which sources were actually used in the response
            response_lines = response_content.lower().split('\n')
            used_content_hashes = set()
            
            # Check each piece of content to see if it was used in the response
            for msg in message_content:
                if 'text' in msg:
                    text = msg['text'].lower()
                    words = set(text.split())
                    # Check if a significant portion of the content appears in the response
                    for line in response_lines:
                        if len(set(line.split()) & words) >= min(3, len(words)):
                            if 'content_hash' in msg:
                                used_content_hashes.add(msg['content_hash'])
                            break
            
            # Get the sources that were actually used
            actually_used_sources = {}
            for content_hash, (source, page_num) in source_content_map.items():
                if content_hash in used_content_hashes:
                    source_file = re.search(r'\[Source: ([^,]+)', source).group(1)
                    source_key = f"{source_file}_{page_num}"
                    if source_key not in actually_used_sources:
                        actually_used_sources[source_key] = (source, page_num, source_file)
            
            # Sort sources by filename first, then by page number
            sorted_sources = sorted(
                actually_used_sources.values(),
                key=lambda x: (x[2], x[1])  # Sort by filename first, then by page number
            )
            
            # Add single references section at the end
            references = "\n\n**References:**\n" + "\n".join(f"- {source[0]}" for source in sorted_sources)
            response_content += references

        return response_content
        
    except Exception as e:
        logger.error(f"Error invoking GPT-4: {str(e)}")
        return "I encountered an error while processing your request. Please try again."
    
def clear_vector_store():
    """Clear all stored vectors and caches"""
    try:
        if os.path.exists(VECTOR_STORE):
            import shutil
            shutil.rmtree(VECTOR_STORE)
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")

def clear_history():
    """Clear the query history and cached responses"""
    try:
        if os.path.exists(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE)):
            os.remove(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE))
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")

def check_openai_credentials():
    """Verify OpenAI API key is properly configured"""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return False
        return True
    except Exception as e:
        logger.error(f"OpenAI configuration error: {str(e)}")
        return False