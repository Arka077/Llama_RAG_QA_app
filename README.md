# PDF RAG System with LLaMA 📚

A sophisticated Retrieval-Augmented Generation (RAG) system that enables intelligent question answering over PDF documents using fine-tuned LLaMA models, hybrid retrieval strategies, and advanced table extraction capabilities.

## 🌟 Overview

This application combines state-of-the-art NLP techniques to provide accurate answers to questions about PDF content. It intelligently processes both textual content and tabular data, using a hybrid retrieval approach that ensures high-quality context extraction for the language model.

## ✨ Key Features

### 🔍 Advanced Retrieval System
- **Hybrid Retrieval**: Combines BM25 (sparse) and dense embeddings for optimal recall
- **Cross-Encoder Reranking**: Uses a trained cross-encoder to rerank retrieved passages for improved relevance
- **Sentence-Level Processing**: Operates at the sentence level for granular context extraction

### 📊 Table Extraction & Processing
- **YOLO-Based Detection**: Automatically detects tables in PDFs using a specialized YOLO model
- **Coordinate-Based Filtering**: Separates table content from regular text to avoid duplication
- **Structured Formatting**: Converts tables into LLM-friendly text format with clear structure

### 🤖 LLaMA Integration
- **Fine-Tuned Model**: Utilizes a fine-tuned LLaMA model with LoRA adapters
- **4-bit Quantization**: Efficient inference using 4-bit quantization with `unsloth`
- **RAG-Optimized Prompting**: Custom prompt templates designed for accurate abstractive answers

### 🎨 User Interface
- **Streamlit Dashboard**: Clean, intuitive web interface for document upload and Q&A
- **Real-Time Processing**: Progress indicators for PDF processing and table extraction
- **Context Visualization**: View retrieved context passages for transparency

## 🏗️ Architecture

```
PDF Upload
    ↓
Table Detection (YOLO) + Text Extraction (PyMuPDF/PDFPlumber)
    ↓
Sentence Segmentation (NLTK)
    ↓
Embedding Generation (Sentence Transformers)
    ↓
Hybrid Retrieval (BM25 + Dense Embeddings)
    ↓
Cross-Encoder Reranking
    ↓
LLaMA Generation (Fine-tuned with LoRA)
    ↓
Answer Display
```

### Retrieval Pipeline

1. **Document Processing**
   - Extract tables using YOLO model from `foduucom/table-detection-and-extraction`
   - Extract text content excluding table regions to prevent duplication
   - Segment both text and table content into sentences

2. **Indexing**
   - Create BM25 index for sparse retrieval
   - Generate dense embeddings using `all-MiniLM-L6-v2`
   - Store sentence metadata for context tracking

3. **Query Processing**
   - Retrieve top-k candidates using hybrid approach (reciprocal rank fusion)
   - Rerank candidates using `ms-marco-MiniLM-L-6-v2` cross-encoder
   - Select top sentences for context

4. **Answer Generation**
   - Format context and question using RAG prompt template
   - Generate answer using fine-tuned LLaMA model
   - Extract and display the response

## 📋 Prerequisites

### System Requirements
- **GPU**: CUDA-compatible GPU recommended (model runs in 4-bit mode)
- **RAM**: Minimum 16GB recommended
- **Storage**: ~10GB for models and dependencies

### Model Files Required

Before running the application, you need to have the following model files:

1. **Base LLaMA Model** (`./base_model/`)
   - Download or train a base LLaMA model compatible with `unsloth`
   - Place in `./base_model/` directory

2. **LoRA Adapter** (`./lora_adapter/`)
   - Fine-tuned LoRA adapter for RAG tasks
   - Place in `./lora_adapter/` directory

**Note**: The application expects these directories to exist locally. You'll need to prepare or train these models separately.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Arka077/Llama_RAG_QA_app.git
cd Llama_RAG_QA_app
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` includes PyTorch with CUDA 12.6 support. Adjust the PyTorch version if you have a different CUDA version:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

### 4. Download NLTK Data

The application will automatically download required NLTK data on first run, but you can pre-download:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 5. Set Up Model Files

Ensure you have the required model files in place:
- `./base_model/` - Base LLaMA model
- `./lora_adapter/` - Fine-tuned LoRA adapter

## 💻 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Using the Interface

1. **Upload PDF**: Click "Choose a PDF file" and select your document
2. **Wait for Processing**: The system will:
   - Extract tables using YOLO detection
   - Extract text content
   - Process sentences and create embeddings
3. **Ask Questions**: Enter your question in the text input
4. **View Results**: Get the answer along with retrieved context passages

### Example Questions

Based on a PDF about trees:
- "What is the height of the Oak tree?"
- "Which tree has the most common leaf shape?"
- "Compare the ages of different trees in the document"

## 🔧 Configuration

### Model Parameters

Edit the configuration in `app.py`:

```python
# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cross-encoder for reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval parameters
TOP_K_SENTENCES = 50  # Number of sentences to use as context

# LLaMA model settings
max_seq_length = 8192  # Maximum sequence length
load_in_4bit = True    # Use 4-bit quantization
```

### Retrieval Tuning

Adjust retrieval parameters in the `answer_question_with_integrated_retrieval` function:

```python
top_k_retrieve=200,    # Initial retrieval pool size
top_k_sentences=50     # Final context size after reranking
```

## 📦 Key Dependencies

- **streamlit**: Web application framework
- **unsloth**: Efficient LLaMA model inference with 4-bit quantization
- **sentence-transformers**: Dense embeddings and cross-encoder
- **rank-bm25**: BM25 sparse retrieval
- **ultralytics (YOLO)**: Table detection
- **pdfplumber/PyMuPDF**: PDF processing
- **faiss-cpu**: Efficient similarity search
- **transformers**: HuggingFace transformers library
- **peft**: Parameter-Efficient Fine-Tuning (LoRA)

## 🛠️ Technical Details

### Table Extraction Process

1. **Detection**: YOLO model identifies table regions in PDF pages
2. **Coordinate Mapping**: Image coordinates are mapped to PDF coordinates
3. **Content Extraction**: PDFPlumber extracts table data from identified regions
4. **Formatting**: Tables are converted to structured text format
5. **Filtering**: Table content is removed from text passages to prevent duplication

### Hybrid Retrieval Strategy

The system uses **Reciprocal Rank Fusion (RRF)** to combine:
- **BM25 scores**: Captures exact keyword matches
- **Dense similarity**: Captures semantic similarity

Formula: `score = 1/(60 + rank_bm25) + 1/(60 + rank_dense)`

### Context Window Management

- Maximum sequence length: 8192 tokens
- Context is constructed from top-k reranked sentences
- Prompt template includes instruction, context, and question

## 🐛 Troubleshooting

### Model Loading Errors

**Problem**: "Error loading LLaMA model"
**Solution**: 
- Ensure `./base_model/` and `./lora_adapter/` directories exist
- Verify model files are compatible with `unsloth`
- Check CUDA is properly installed for GPU usage

### Memory Issues

**Problem**: Out of memory errors
**Solution**:
- Reduce `max_seq_length` in model loading
- Decrease `top_k_sentences` to use less context
- Ensure 4-bit quantization is enabled (`load_in_4bit=True`)

### Table Detection Failures

**Problem**: Tables not being detected
**Solution**:
- Adjust `confidence_threshold` in `extract_tables_with_coordinates`
- Ensure PDF has clear table boundaries
- Check internet connection for YOLO model download

### Slow Performance

**Problem**: Processing or inference is slow
**Solution**:
- Use GPU if available
- Reduce PDF resolution in table extraction
- Decrease retrieval pool size (`top_k_retrieve`)
- Consider using smaller embedding models

## 📊 Performance Considerations

- **First Run**: Initial model loading takes 1-2 minutes
- **PDF Processing**: ~5-15 seconds per page (depends on table count)
- **Query Response**: ~2-5 seconds per question
- **Memory Usage**: ~8-12GB GPU memory with 4-bit quantization

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for more document formats
- Additional retrieval strategies
- Better table formatting
- Multi-document Q&A
- Conversation history support

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Unsloth**: For efficient LLaMA inference
- **HuggingFace**: For model hosting and transformers library
- **Foduucom**: For the table detection YOLO model
- **Sentence Transformers**: For embedding models
- **LangChain**: For retrieval utilities

## 📧 Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/Arka077/Llama_RAG_QA_app).

---

**Built with ❤️ using LLaMA, Streamlit, and state-of-the-art NLP techniques**