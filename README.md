# AI-Powered Log Processing
## Overview
This repository contains examples for efficiently processing logs using a **locally hosted Mistral LLM** with **LangChain**. 
 
Logs can be intelligently processed by:

- Tokenizing and Structuring Logs
- Processing Logs via LangChain Pipelines
- Efficiently Storing and Retrieving Context with FAISS
- Optimizing Performance with Incremental Log Submission
- Supporting Any LLM Backend (Mistral, GPT, Llama, etc.)

By leveraging LangChain, FAISS, and local LLMs like Mistral (via Ollama), this project ensures fast, secure, and scalable log analysis without sending data outside your environment.
## Setting Up Ollama and Mistral on a Local Machine
## Setup Instructions

### 1. **System Requirements**
- Python 3.10+
- Mistral LLM (running locally)
- Virtual Environment (recommended)
- Dependencies: `LangChain`, `Ollama`, `FastAPI` (optional for API-based usage)

### 2. **Installation**
#### a) **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
#### b) **Install dependencies**
```bash
pip install langchain langchain-community langchain-ollama faiss-cpu numpy
```

#### c) **Ensure Mistral LLM is running locally**
If you're using `ollama`, start the model:
```bash
ollama run mistral
```
Or if using a raw model file:
```bash
MODEL_PATH="/path/to/mistral-model.bin"
```

---

```python
import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "mistral",
    "prompt": "Hello, how are you?",
    "stream": False
})
print(response.json())
```

## Optimization Approach
### **1. Initial Issue: Slow Processing**
- Directly sending large logs to Mistral took **300+ seconds**.
- Single-threaded processing led to **timeouts**.

### **2. Successive Optimizations:**
#### ✅ **Chunk-Based Processing**
- Split logs into **smaller batches** and process them in **parallel**.
#### ✅ **Contextual Prompting**
- Structured prompts into steps:
  1. Context: Describe log format and purpose.
  2. Schema: Define JSON structure.
  3. Example Logs: Provide sample mappings.
  4. Log Processing: Send batch logs for processing.
#### ✅ **LangChain Integration**
- Used `LLMChain` to manage multi-step execution.
- Parallel execution reduced processing time drastically.

---

## Key Insights
- **Batching reduces execution time** significantly.
- **Well-structured prompts improve LLM understanding.**
- **Parallel processing prevents timeouts and bottlenecks.**
---
