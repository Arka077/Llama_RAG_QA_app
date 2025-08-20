import streamlit as st
import unsloth
import os
import faiss
import pickle
import nltk
import re
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from transformers import pipeline
from PyPDF2 import PdfReader
from keybert import KeyBERT
from collections import deque
from unsloth import FastLanguageModel
from rank_bm25 import BM25Okapi
import numpy as np
import torch
from peft import PeftModel
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.download('punkt_tab')
    except:
        pass

# ====== Config ======
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_HISTORY_TURNS = 5
TOP_K_SENTENCES = 50

# ====== RAG Prompt Template ======
rag_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# ====== Load models ======
@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    kw_model = KeyBERT()
    
    # Load fine-tuned Unsloth LLaMA model
    try:
        llama_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./base_model",
            max_seq_length=8192,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        
        # Then load the LoRA adapter
        llama_model = PeftModel.from_pretrained(
            llama_model, 
            "./lora_adapter",
            local_files_only=True
        )
        
        FastLanguageModel.for_inference(llama_model)
        st.success("Models loaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading LLaMA model: {e}")
        llama_model, tokenizer = None, None
    
    return embedder, cross_encoder, kw_model, llama_model, tokenizer

# ====== Table Processing Functions ======
class TableProcessor:
    def __init__(self):
        self.table_coordinates = []
        self.formatted_tables = []

    def extract_tables_with_coordinates(self, pdf_path, confidence_threshold=0.5):
        try:
            model_path = hf_hub_download(
                repo_id="foduucom/table-detection-and-extraction",
                filename="best.pt"
            )
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            return [], []

        doc = fitz.open(pdf_path)
        all_dataframes = []
        formatted_tables = []
        table_coordinates = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for page_num in range(len(doc)):
            progress = (page_num + 1) / len(doc)
            progress_bar.progress(progress)
            status_text.text(f"Processing tables on page {page_num + 1}/{len(doc)}...")

            page_rect = doc[page_num].rect
            page_width = page_rect.width
            page_height = page_rect.height

            pix = doc[page_num].get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")

            temp_img = f"temp_page_{page_num}.png"
            with open(temp_img, "wb") as f:
                f.write(img_bytes)

            try:
                results = model.predict(source=temp_img, conf=confidence_threshold, save=False)

                if results[0].boxes is not None:
                    img_width = pix.width
                    img_height = pix.height

                    with pdfplumber.open(pdf_path) as pdf:
                        page = pdf.pages[page_num]

                        for i in range(len(results[0].boxes)):
                            x1_img, y1_img, x2_img, y2_img = map(int, results[0].boxes.xyxy[i])

                            x1 = (x1_img / img_width) * page_width
                            y1 = (y1_img / img_height) * page_height
                            x2 = (x2_img / img_width) * page_width
                            y2 = (y2_img / img_height) * page_height

                            table_coordinates.append({
                                'page': page_num,
                                'bbox': (x1-5, y1-5, x2+5, y2+5)
                            })

                            try:
                                cropped_page = page.crop((x1-5, y1-5, x2+5, y2+5))
                                tables = cropped_page.extract_tables()

                                if tables and tables[0] and len(tables[0]) >= 2:
                                    df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
                                    df = df.dropna(how='all').loc[:, df.columns.notna()]

                                    if not df.empty:
                                        all_dataframes.append(df)
                                        formatted_table = self.format_dataframe_for_llama(df)
                                        formatted_tables.append(formatted_table)

                            except Exception as e:
                                continue

            except Exception as e:
                st.warning(f"Error processing page {page_num + 1}: {e}")

            if os.path.exists(temp_img):
                os.remove(temp_img)

        doc.close()
        progress_bar.progress(1.0)
        status_text.text(f"Extracted {len(formatted_tables)} tables")

        self.table_coordinates = table_coordinates
        self.formatted_tables = formatted_tables

        return formatted_tables, table_coordinates

    def format_dataframe_for_llama(self, df):
        if df.empty:
            return ""

        columns = df.columns.tolist()
        header_line = f"Table about {', '.join(columns)}"
        data_lines = []
        
        for _, row in df.iterrows():
            row_parts = []
            for col in columns:
                value = str(row[col]).strip() if pd.notna(row[col]) else ""
                if value:
                    row_parts.append(f"{col}: {value}")

            if row_parts:
                data_lines.append(", ".join(row_parts))

        if data_lines:
            formatted_text = header_line + "\n" + "\n".join(data_lines)
        else:
            formatted_text = header_line

        return formatted_text

# ====== Text Extraction ======
def extract_text_excluding_tables(pdf_path, table_coordinates, formatted_tables):
    """
    Extract text from PDF while excluding table regions to avoid duplication
    Uses both coordinate-based and content-based filtering
    """
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Get table coordinates for this specific page
            page_table_coords = [
                coord['bbox'] for coord in table_coordinates
                if coord['page'] == page_num
            ]
            
            # Extract text first
            text = page.extract_text()
            
            if not text or not text.strip():
                continue
                
            # If there are tables on this page, filter out table content
            if page_table_coords:
                try:
                    # Method 1: Coordinate-based filtering
                    chars = page.chars
                    filtered_chars = []
                    
                    for char in chars:
                        char_x = char['x0']
                        char_y = char['y0']
                        
                        # Check if character is inside any table bbox
                        inside_table = False
                        for bbox in page_table_coords:
                            x1, y1, x2, y2 = bbox
                            if x1 <= char_x <= x2 and y1 <= char_y <= y2:
                                inside_table = True
                                break
                        
                        if not inside_table:
                            filtered_chars.append(char)
                    
                    # Extract text from filtered characters
                    if filtered_chars:
                        filtered_page = page.within_bbox(page.bbox)
                        filtered_page.chars = filtered_chars
                        filtered_text = filtered_page.extract_text()
                        if filtered_text:
                            text = filtered_text
                        
                except Exception as e:
                    # Keep original text if coordinate filtering fails
                    pass
                
                # Method 2: Content-based filtering (remove table-like patterns)
                text = remove_table_content_from_text(text, formatted_tables)
            
            if text and text.strip():
                documents.append(text.strip())

    return documents

def remove_table_content_from_text(text, formatted_tables):
    """
    Remove table content from text using pattern matching
    """
    if not formatted_tables:
        return text
    
    # Extract key table data patterns to remove
    table_patterns_to_remove = []
    
    for table_text in formatted_tables:
        lines = table_text.split('\n')
        for line in lines[1:]:  # Skip header line
            if ':' in line and ',' in line:
                # Extract key data patterns like "Tree: Oak, Age: 250 years"
                parts = line.split(', ')
                if len(parts) >= 3:  # Ensure it's a substantial table row
                    # Create pattern to match this data in flowing text
                    # Look for the key values that appear together
                    values = []
                    for part in parts[:3]:  # Use first 3 parts as signature
                        if ':' in part:
                            value = part.split(':', 1)[1].strip()
                            if value and len(value) > 1:
                                values.append(value)
                    
                    if len(values) >= 2:
                        table_patterns_to_remove.extend(values)
    
    # Remove table-like content from text
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that look like table headers or data rows
        is_table_line = False
        
        # Check if line contains multiple table data patterns
        pattern_matches = 0
        for pattern in table_patterns_to_remove:
            if pattern.lower() in line.lower():
                pattern_matches += 1
        
        # If line contains many table patterns, it's likely a table row
        if pattern_matches >= 3:
            is_table_line = True
        
        # Also check for typical table formatting patterns
        if (line.count(' ') > 10 and 
            any(keyword in line.lower() for keyword in ['tree', 'age', 'height', 'leaf', 'years']) and
            pattern_matches >= 2):
            is_table_line = True
            
        # Check for lines that are mostly table-like structure
        words = line.split()
        if (len(words) > 6 and 
            pattern_matches >= 2 and
            any(w.endswith('m') or w.endswith('years') for w in words)):
            is_table_line = True
        
        if not is_table_line:
            filtered_lines.append(line)
    
    filtered_text = '\n'.join(filtered_lines)
    return filtered_text

# ====== Sentence Processing ======
class SentenceProcessor:
    def __init__(self):
        self.sentences = []
        self.sentence_metadata = []

    def split_text_to_sentences(self, text, doc_id=0, content_type="text"):
        sentences = nltk.sent_tokenize(text)
        cleaned_sentences = []
        
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) >= 10:
                cleaned_sentences.append(sent)
                self.sentence_metadata.append({
                    'doc_id': doc_id,
                    'sentence_id': i,
                    'original_text': sent,
                    'length': len(sent),
                    'content_type': content_type
                })

        return cleaned_sentences

    def process_text_documents(self, documents):
        text_sentences = []
        for doc_id, doc_text in enumerate(documents):
            doc_sentences = self.split_text_to_sentences(doc_text, doc_id, "text")
            text_sentences.extend(doc_sentences)
        return text_sentences

    def process_table_documents(self, table_texts):
        table_sentences = []
        for doc_id, table_text in enumerate(table_texts):
            doc_sentences = self.split_text_to_sentences(table_text, doc_id, "table")
            table_sentences.extend(doc_sentences)
        return table_sentences

    def process_all_documents(self, text_documents, table_texts):
        text_sentences = self.process_text_documents(text_documents)
        table_sentences = self.process_table_documents(table_texts)
        all_sentences = text_sentences + table_sentences
        self.sentences = all_sentences
        return all_sentences

# ====== Retrieval System ======
class SentenceRAGRetriever:
    def __init__(self, sentences, sentence_metadata, embedder, cross_encoder):
        self.sentences = sentences
        self.sentence_metadata = sentence_metadata
        self.embedder = embedder
        self.cross_encoder = cross_encoder
        self.setup_bm25_index()
        self.setup_dense_embeddings()

    def setup_bm25_index(self):
        tokenized_sentences = [sent.lower().split() for sent in self.sentences]
        self.bm25 = BM25Okapi(tokenized_sentences)

    def setup_dense_embeddings(self):
        self.dense_embeddings = self.embedder.encode(self.sentences, convert_to_tensor=True)

    def hybrid_sentence_retrieval(self, query, top_k=100):
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)

        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        if len(query_emb.shape) == 1:
            query_emb = query_emb.unsqueeze(0)
        cosine_scores = torch.nn.functional.cosine_similarity(
            self.dense_embeddings, query_emb, dim=1
        )

        sentence_scores = {}

        bm25_ranked = np.argsort(bm25_scores)[::-1]
        for rank, sent_idx in enumerate(bm25_ranked):
            sentence_scores[sent_idx] = sentence_scores.get(sent_idx, 0) + 1 / (60 + rank)

        dense_ranked = np.argsort(cosine_scores.cpu().numpy())[::-1]
        for rank, sent_idx in enumerate(dense_ranked):
            sentence_scores[sent_idx] = sentence_scores.get(sent_idx, 0) + 1 / (60 + rank)

        fused_ranking = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentence_indices = [idx for idx, score in fused_ranking[:top_k]]

        return top_sentence_indices, sentence_scores

    def rerank_sentences(self, query, sentence_indices, top_k=50):
        candidate_sentences = [self.sentences[i] for i in sentence_indices]
        pairs = [(query, sent) for sent in candidate_sentences]
        rerank_scores = self.cross_encoder.predict(pairs)

        reranked_with_scores = list(zip(rerank_scores, candidate_sentences, sentence_indices))
        reranked_with_scores.sort(key=lambda x: x[0], reverse=True)
        top_reranked = reranked_with_scores[:top_k]

        return [sent for score, sent, orig_idx in top_reranked]

# ====== Main Processing Pipeline ======
def process_pdf_with_tables_and_text(pdf_path):
    st.write("🚀 Starting PDF processing...")
    
    table_processor = TableProcessor()
    formatted_tables, table_coordinates = table_processor.extract_tables_with_coordinates(pdf_path)
    
    st.write("📄 Extracting text...")
    text_documents = extract_text_excluding_tables(pdf_path, table_coordinates, formatted_tables)
    
    st.write("📝 Processing sentences...")
    processor = SentenceProcessor()
    sentences = processor.process_all_documents(text_documents, formatted_tables)
    
    return sentences, processor.sentence_metadata, formatted_tables

# ====== QA Function ======
def answer_question_with_integrated_retrieval(
    user_question,
    retriever,
    llama_model,
    tokenizer,
    top_k_retrieve=200,
    top_k_sentences=50
):
    if llama_model is None or tokenizer is None:
        return "Model not loaded properly. Please check the model files.", []
    
    sentence_indices, scores = retriever.hybrid_sentence_retrieval(
        user_question, top_k=top_k_retrieve
    )
    
    final_sentences = retriever.rerank_sentences(
        user_question, sentence_indices, top_k=top_k_sentences
    )
    
    context_str = '\n'.join(final_sentences)
    
    instruction = "Answer the question using information provided only in the context. The context contains both text and table data. The answer should be abstractive and accurate."
    input_text = f"Context: {context_str}\n\nQuestion: {user_question}"
    
    formatted_prompt = rag_prompt.format(instruction, input_text, "")
    
    try:
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        outputs = llama_model.generate(**inputs, max_new_tokens=128, use_cache=True)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("### Response:")[-1].strip()
    except Exception as e:
        answer = f"Error generating answer: {e}"
    
    return answer, final_sentences

# ====== Streamlit App ======
def main():
    st.title("📚 PDF RAG System")
    st.write("Upload a PDF and ask questions about its content (text and tables)")
    
    # Initialize NLTK
    download_nltk_data()
    
    # Load models
    embedder, cross_encoder, kw_model, llama_model, tokenizer = load_models()
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process PDF
        if 'processed_pdf' not in st.session_state or st.session_state.get('last_pdf') != uploaded_file.name:
            st.session_state.last_pdf = uploaded_file.name
            
            with st.spinner("Processing PDF..."):
                sentences, sentence_metadata, formatted_tables = process_pdf_with_tables_and_text(tmp_path)
                retriever = SentenceRAGRetriever(sentences, sentence_metadata, embedder, cross_encoder)
                
                st.session_state.processed_pdf = True
                st.session_state.retriever = retriever
                st.session_state.sentences = sentences
                st.session_state.formatted_tables = formatted_tables
            
            st.success(f"✅ PDF processed! Found {len(formatted_tables)} tables and {len(sentences)} sentences")
            
            # Show summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sentences", len(sentences))
            with col2:
                st.metric("Tables Found", len(formatted_tables))
            with col3:
                text_sentences = sum(1 for meta in sentence_metadata if meta.get('content_type') == 'text')
                st.metric("Text Sentences", text_sentences)
        
        # Question answering interface
        if st.session_state.get('processed_pdf'):
            st.subheader("Ask Questions")
            
            user_question = st.text_input("Enter your question:")
            
            if st.button("Get Answer") and user_question:
                with st.spinner("Generating answer..."):
                    answer, retrieved_sentences = answer_question_with_integrated_retrieval(
                        user_question,
                        st.session_state.retriever,
                        llama_model,
                        tokenizer
                    )
                
                st.subheader("Answer:")
                st.write(answer)
                
                # Show retrieved context
                with st.expander("View Retrieved Context"):
                    for i, sent in enumerate(retrieved_sentences[:5], 1):
                        st.write(f"**Context {i}:** {sent}")
        
        # Clean up temp file
        os.unlink(tmp_path)

if __name__ == "__main__":
    main()