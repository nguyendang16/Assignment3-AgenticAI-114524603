# Financial Document RAG Agent with LangGraph

A Retrieval-Augmented Generation (RAG) agent for analyzing Apple and Tesla 10-K financial reports using LangGraph workflow.

## Features

- **Intelligent Query Routing**: Automatically routes questions to Apple, Tesla, or both data sources
- **Document Grading**: Filters irrelevant documents before generation
- **Answer Generation**: Produces English answers with proper citations
- **Query Rewriting**: Refines search queries when initial retrieval fails
- **Multi-Provider LLM Support**: Compatible with Google Gemini, OpenAI, and Anthropic
- **Legacy Agent**: Includes ReAct-based agent for comparison

## Architecture

```
Question → Retrieve → Grade Documents → Generate Answer
                ↑            ↓
                ←── Rewrite ←┘ (if irrelevant)
```

## Prerequisites

- Python 3.11 (Required)
- API Key for one of: Google Gemini, OpenAI, or Anthropic

## Setup

### 1. Virtual Environment

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Copy `.env_example` to `.env` and configure your API key:

```bash
cp .env_example .env
```

Edit `.env` to set your preferred LLM provider:
```
LLM_PROVIDER=google  # Options: google, openai, anthropic
GOOGLE_API_KEY=your_api_key_here
```

### 4. Add Financial Reports

Place PDF files in the `data/` folder:
- `FY24_Q4_Consolidated_Financial_Statements.pdf` (Apple 10-K)
- `tsla-20241231-gen.pdf` (Tesla 10-K)

## Usage

### Step 1: Build Vector Database

```bash
python build_rag.py
```

This creates ChromaDB vector stores from the PDF documents.

### Step 2: Run Evaluation

```bash
python evaluator.py
```

Runs 14 test cases covering:
- Single company queries (Apple/Tesla revenue, R&D, costs)
- Cross-company comparisons
- Trap questions (unknown/future information)

## Project Structure

```
├── langgraph_agent.py    # Main LangGraph agent with workflow nodes
├── build_rag.py          # PDF ingestion and vector DB builder
├── evaluator.py          # Benchmark testing with LLM-as-Judge
├── config.py             # LLM and embedding configuration
├── data/                 # PDF financial reports
├── chroma_db/            # Vector database storage
└── eval_output/          # Evaluation results
```

## Key Components

### `langgraph_agent.py`
- `retrieve_node`: Routes queries and retrieves relevant documents
- `grade_documents_node`: Assesses document relevance
- `generate_node`: Generates final answers with citations
- `rewrite_node`: Reformulates queries for better retrieval
- `run_legacy_agent`: ReAct-based agent for baseline comparison

### `config.py`
- Multi-provider LLM factory (Google, OpenAI, Anthropic)
- Local HuggingFace embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)

## Configuration Options

In `evaluator.py`, change `TEST_MODE` to switch between agents:
```python
TEST_MODE = "GRAPH"   # LangGraph workflow agent
TEST_MODE = "LEGACY"  # ReAct-based agent
```

## Evaluation Results

Latest benchmark score: **10/14** test cases passed

| Category | Status |
|----------|--------|
| Apple Revenue | ✅ Pass |
| Apple R&D | ✅ Pass |
| Apple Services Cost | ✅ Pass |
| Tesla CEO Identity | ✅ Pass |
| R&D Comparison | ✅ Pass |
| Gross Margin Analysis | ✅ Pass |
| Unknown Info (Trap) | ✅ Pass |

## Tech Stack

- **Framework**: LangChain, LangGraph
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace Sentence Transformers
- **LLMs**: Google Gemini / OpenAI GPT / Anthropic Claude
- **PDF Processing**: PyMuPDF  
