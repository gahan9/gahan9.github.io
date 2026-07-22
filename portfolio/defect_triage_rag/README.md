# Intelligent Defect Triage System using RAG & LangGraph

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.40+-green.svg)](https://langchain-ai.github.io/langgraph/)

AI-powered defect analysis and real-time triage prediction for enterprise HPC validation workflows.

## Overview

This system leverages **Retrieval-Augmented Generation (RAG)** and **LangGraph** to provide intelligent, context-aware triage predictions for JIRA defects in HPC validation environments.

### Key Features

- 🔍 **RAG Pipeline**: Semantic search over historical defects using PyTorch embeddings
- 🧠 **LangGraph Orchestration**: Multi-step workflow for robust defect analysis
- 📊 **Trend Analysis**: Pattern detection and root cause clustering
- 🎯 **Real-Time Prediction**: Context-aware triage with confidence scoring
- 🏢 **Enterprise Ready**: Scalable architecture for large validation datasets

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    LangGraph Triage Workflow                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────┐    ┌─────────────┐    ┌──────────────┐             │
│   │  START  │───▶│ Embed Defect│───▶│ Retrieve     │             │
│   └─────────┘    │  (PyTorch)  │    │ Context (RAG)│             │
│                  └─────────────┘    └──────┬───────┘             │
│                                            │                      │
│   ┌─────────┐    ┌─────────────┐    ┌──────▼───────┐             │
│   │   END   │◀───│  Generate   │◀───│  Classify    │             │
│   └─────────┘    │  Prediction │    │  Domain      │             │
│                  │    (LLM)    │    └──────┬───────┘             │
│                  └─────────────┘           │                      │
│                        ▲            ┌──────▼───────┐             │
│                        └────────────│  Analyze     │             │
│                                     │  Trends      │             │
│                                     └──────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Embedding Generator (`embeddings.py`)
- Uses **sentence-transformers** with PyTorch backend
- Generates semantic embeddings for defect texts
- Preliminary domain classification via keyword analysis

### 2. Vector Store (`vector_store.py`)
- **FAISS**-based vector storage for efficient similarity search
- Supports domain filtering and batch operations
- Persistence for enterprise deployments

### 3. Triage Graph (`triage_graph.py`)
- **LangGraph** workflow orchestration
- Multi-node pipeline: Embed → Retrieve → Classify → Analyze → Predict
- LLM integration for context-aware predictions

### 4. Data Models (`models.py`)
- Pydantic models for type safety
- Domain classifications, rejection reasons, and priorities
- Structured prediction outputs

## Installation

```bash
# Clone and navigate
cd defect_triage_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (for LLM node)
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Demo Mode
```bash
python -m defect_triage_rag.main --demo
```

### Triage from JSON
```bash
python -m defect_triage_rag.main --defect-json path/to/defect.json
```

### Programmatic Usage
```python
from defect_triage_rag.embeddings import DefectEmbeddingGenerator
from defect_triage_rag.vector_store import DefectVectorStore
from defect_triage_rag.triage_graph import DefectTriageGraph
from defect_triage_rag.models import JiraDefect

# Initialize components
embedding_gen = DefectEmbeddingGenerator()
vector_store = DefectVectorStore()
triage_graph = DefectTriageGraph(embedding_gen, vector_store)

# Create defect
defect = JiraDefect(
    id="HPC-1234",
    summary="GPU memory error during training",
    description="PyTorch distributed training fails with OOM...",
    component="ROCm-Memory",
    platform="MI300X",
)

# Run triage
prediction = triage_graph.triage(defect)

print(f"Priority: {prediction.predicted_priority.value}")
print(f"Domain: {prediction.predicted_domain.value}")
print(f"Confidence: {prediction.confidence_score:.2%}")
```

## Domain Classifications

| Domain | Description |
|--------|-------------|
| `infrastructure` | Setup, networking, cluster issues |
| `memory` | GPU/HIP memory allocation failures |
| `system_call` | Kernel, driver, syscall errors |
| `performance` | Regressions, benchmarking issues |
| `firmware` | BIOS, UEFI, RAS related |
| `driver` | AMD GPU driver issues |
| `configuration` | Config, environment issues |

## Triage Output

```json
{
  "defect_id": "HPC-5678",
  "predicted_domain": "memory",
  "predicted_priority": "P1",
  "predicted_rejection_reason": null,
  "confidence_score": 0.87,
  "similar_defects": ["HPC-1234", "HPC-2345"],
  "suggested_resolution": "Increase GPU memory pool allocation...",
  "context_summary": "Memory allocation failure during distributed training...",
  "requires_manual_review": false
}
```

## Performance

- **Embedding Generation**: ~50ms per defect (GPU accelerated)
- **Vector Search**: <10ms for top-5 retrieval (FAISS)
- **Full Pipeline**: ~2-3s (including LLM call)

## License

MIT License - See LICENSE for details.

---

**Author**: Gahan Saraiya  
**Project**: Intelligent Defect Triage for HPC Validation
