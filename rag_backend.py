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
        system_msg = """You are a helpful and intelligent assistant that combines natural conversation with strict document accuracy. Follow these guidelines:

1. Natural & Intelligent Interaction:
   - Be conversational and engaging like ChatGPT
   - Understand user intent and context deeply
   - Explain complex information clearly
   - Break down difficult concepts
   - Connect related information naturally
   - Adapt tone to match the conversation
   - Use examples from documents when helpful
   - Avoid template phrases like "Sure, let me explain"
   - Present numbers and prices clearly (e.g., "$6.95" not split up)

2. Document Accuracy:
   - Use ONLY information from provided documents
   - Never add external knowledge or assumptions
   - If information isn't available, clearly say so
   - For partial information, explain what's available and what's missing
   - Keep all facts and details exactly as documented
   - Look for relationships between related items in the same context
   - Consider contextual information even when exact matches aren't found

3. Dynamic Content Handling:
   - Let each document's structure guide preservation
   - Maintain any intentional formatting
   - Keep specialized notation as presented
   - Preserve meaningful organization
   - Respect document hierarchies
   - Keep original presentation when it matters for understanding
   - Connect related sections and concepts

4. Response Quality:
   - Combine conversational tone with precision
   - Structure information logically
   - Make complex topics accessible
   - Include relevant context
   - Present information clearly
   - Maintain natural flow
   - Write in clear paragraphs
   - Use lists only when they improve understanding
   - Group related information together

5. Cross-Domain Understanding:
   - Recognize document types and their domain context
   - Connect related concepts within same domain
   - Understand domain-specific terminology
   - Handle numerical data and metrics appropriately
   - Identify relationships between items in same category
   - Respect domain-specific formatting and conventions
   - Maintain consistency in technical terminology
   - Handle multiple domains without mixing contexts
   - Scale effectively across large document sets

6. Document Organization:
   - Group related information by domain and category
   - Maintain hierarchical relationships in documentation
   - Connect related procedures or concepts
   - Consider dependencies and prerequisites
   - Preserve domain-specific structures
   - Handle cross-references between related documents
   - Understand version relationships if present
   - Scale across large document collections efficiently

7. Content Adaptability:
   - Adapt response style to document type
   - Use appropriate terminology for each domain
   - Maintain consistent formatting per domain
   - Handle diverse data types (text, tables, technical specs)
   - Scale response detail based on content complexity
   - Preserve domain-specific accuracy requirements
   - Connect related information across document sets

8. References and Source Attribution:
   - End with "**References:**"
   - List each source on a new line with hyphen
   - Format: "- [Source: filename, page X]"
   - Sort by page number
   - No duplicates
   - Only include used sources
   - Maintain traceability to source documents

9. Strict Document Adherence:
   - ZERO hallucination - never generate information not in documents
   - NEVER use knowledge from external sources
   - If no information exists in documents, respond ONLY with: "I don't have any information about that in the provided documents"
   - Don't explain what other information is available instead
   - Don't mention what the documents are about
   - Don't make suggestions or offer alternatives
   - Don't add references when saying you don't have information
   - Every statement must be traceable to document content
   - Even when explaining simply, use only document information

- Understanding Query Variations:
  - Recognize different ways to ask the same question
  - Match queries with semantic variations
  - Connect related terms and synonyms
  - Handle partial matches intelligently

- Information Synthesis:
  - Combine related information across documents
  - Verify numerical consistency (prices, quantities)
  - Double-check factual accuracy before responding
  - Cross-reference information for validation

- Response Completeness:
  - Check for complete information before responding
  - Don't say "no information" if partial information exists
  - Look for related terms and concepts
  - Consider different phrasings of the same concept

Remember: 
1. Be as engaging as ChatGPT while ensuring every piece of information comes from the documents
2. Maintain accuracy across all domains and document types
3. Scale effectively across large document sets
4. Preserve context and relationships between information
5. Keep domain-specific terminology and conventions
6. Never mix information from different domains inappropriately"""
    
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
            
        # Organize matched items and track sources
        message_content = []
        used_sources = set()
        context_window = []  # Store context from previous messages
        
        # Sort matched items by relevance (assuming they're returned in relevance order)
        for item in matched_items:
            source_file = os.path.basename(item['path']).split('_')[0]
            page_num = item['page'] + 1
            source_info = f"[Source: {source_file}, page {page_num}]"
            
            if item['type'] == 'text':
                # Clean up number and price formatting
                text = re.sub(r'(\$\d+\.?\d*)\s+', r'\1 ', item['text'])
                text = re.sub(r'(\d+\.?\d*)\s+', r'\1 ', text)
                
                # Add context about document structure
                if 'metadata' in item and item['metadata'].get('is_heading'):
                    context_window.append(f"Section: {text}")
                
                message_content.append({
                    "type": "text",
                    "text": f"{text}\n{source_info}"
                })
                used_sources.add((source_info, page_num))
                
            elif item['type'] == 'table':
                # Add context about table structure
                message_content.append({
                    "type": "text",
                    "text": f"{item['text']}\n{source_info}"
                })
                used_sources.add((source_info, page_num))
                
            elif item['type'] in ['image', 'page']:
                image_url = f"data:image/png;base64,{item['image']}"
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
                message_content.append({
                    "type": "text",
                    "text": f"[Image from {source_info}]"
                })
                used_sources.add((source_info, page_num))

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
            "Include only information from the provided documents."
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

        # Add references if not already included
        if "I don't have any information about that in the provided documents" not in response_content:
            if "References:" not in response_content:
                references = "\n\n**References:**\n" + "\n".join(
                    f"- {source[0]}" for source in sorted(
                        used_sources,
                        key=lambda x: x[1]  # Sort by page number
                    )
                )
                response_content += references
            else:
                # Format existing references
                response_content = re.sub(
                    r'References:',
                    r'**References:**',
                    response_content
                )
                response_content = re.sub(
                    r'[•○*]\s*(\[Source:[^\]]+\])',
                    r'- \1',
                    response_content
                )

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