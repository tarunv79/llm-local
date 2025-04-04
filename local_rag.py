import json
import time
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

LOCAL_MODEL_PATH = "Projects/model/sentence-transformers/all-MiniLM-L6-v2"
MISTRAL_MODEL = "mistral"
CHUNK_SIZE = 5  # Logs per batch
TIMEOUT = 60  # Timeout per LLM call

FAISS_INDEX_FILE = "faiss_index"

# ---------------------------------
# Sample Schema and Examples
# ---------------------------------
schemas = [
    {
        "name": "Server Logs",
        "content": """
        Schema:
        {
            "timestamp": "ISO 8601 format",
            "server": {
                "cpu": "CPU model",
                "memory": "RAM size",
                "disk": "Storage type"
            },
            "status": {
                "state": "Running | Idle | Down | Overload",
                "temperature": "Numeric with °C",
                "alert_level": "None | Low | Medium | High | Critical"
            }
        }
        Examples:
        Log: "2025-03-20 15:30:45 Server CPU: Intel Xeon E5-2670, Memory: 64GB DDR4, Disk: 512GB SSD. Status: Running, Temperature: 45°C, Alert: None."
        Output:
        {
            "timestamp": "2025-03-20 15:30:45",
            "server": {
                "cpu": "Intel Xeon E5-2670",
                "memory": "64GB DDR4",
                "disk": "512GB SSD"
            },
            "status": {
                "state": "Running",
                "temperature": "45°C",
                "alert_level": "None"
            }
        }
        """
    }
]

# Sample Logs
logs = [
    # "2025-03-20 15:30:45 Server CPU: Intel Xeon E5-2670, Memory: 64GB DDR4, Disk: 512GB SSD. Status: Running, Temperature: 45°C, Alert: None.",
    # "2025-03-20 15:35:22 Server CPU: AMD EPYC 7742, Memory: 128GB DDR4, Disk: 1TB NVMe. Status: Idle, Temperature: 40°C, Alert: None.",
    "2025-03-20 15:40:10 Server CPU: Intel Core i9-9900K, Memory: 32GB DDR4, Disk: 256GB NVMe. Status: Down, Temperature: 80°C, Alert: High.",
    "2025-03-20 15:45:55 Server CPU: AMD Ryzen 9 5950X, Memory: 64GB DDR4, Disk: 2TB SSD. Status: Overload, Temperature: 90°C, Alert: High."
]


# ---------------------------------
# Local Embedding Model + FAISS Setup
# ---------------------------------
def create_faiss_index(schemas, index_file=FAISS_INDEX_FILE):
    """Index schemas in FAISS and save to disk."""

    # Use local embeddings
    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)

    if os.path.exists(index_file):
        # Load FAISS index from disk with deserialization enabled
        print(f"🔥 Loading FAISS index from {index_file}")
        db = FAISS.load_local(index_file, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🚀 Indexing schemas...")
        docs = [Document(page_content=schema["content"]) for schema in schemas]

        # Create new FAISS index
        db = FAISS.from_documents(docs, embeddings)

        # Save FAISS index to disk
        db.save_local(index_file)
        print(f"✅ Index saved to {index_file}")

    return db


# ---------------------------------
# FAISS Retrieval
# ---------------------------------
def retrieve_context(faiss_index, query, k=2):
    """Retrieve relevant schemas/examples from FAISS."""
    results = faiss_index.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])


# ---------------------------------
# Mistral LLM Execution
# ---------------------------------
def process_logs_with_rag(logs, faiss_index):
    """Process logs with RAG (FAISS + Mistral LLM)."""
    llm = ChatOllama(model=MISTRAL_MODEL, temperature=0.2, timeout=TIMEOUT)

    mapped_results = []

    for log in logs:
        print(f"🚀 Processing Log: {log}")

        # Retrieve relevant schema and examples from FAISS
        context = retrieve_context(faiss_index, log)

        prompt = f"""
        Context:
        {context}

        Log:
        {log}

        Task: Map the log to the schema and output as JSON.
        """

        template = PromptTemplate(input_variables=["prompt"], template="{prompt}")
        chain = template | llm

        try:
            start_time = time.time()
            response = chain.invoke({"prompt": prompt})
            duration = time.time() - start_time

            # Parse JSON response
            result = json.loads(response.content)
            mapped_results.append(result)

            print(f"✅ Processed in {duration:.2f} seconds.")
        except Exception as e:
            print(f"❌ Exception: {e}")

    return mapped_results


# ---------------------------------
# Main Execution
# ---------------------------------
if __name__ == "__main__":
    start = time.time()

    # Step 1: Create FAISS index
    faiss_index = create_faiss_index(schemas)

    # Step 2: Process logs using RAG pipeline
    mapped_logs = process_logs_with_rag(logs, faiss_index)

    end = time.time()

    # Results
    print("\n🔥 Final Mapped Logs:")
    print(json.dumps(mapped_logs, indent=4))

    print(f"\n🚀 Processed {len(logs)} logs in {end - start:.2f} seconds.")

"""
🚀 Indexing schemas...
✅ Index saved to faiss_index
🚀 Processing Log: 2025-03-20 15:40:10 Server CPU: Intel Core i9-9900K, Memory: 32GB DDR4, Disk: 256GB NVMe. Status: Down, Temperature: 80°C, Alert: High.
✅ Processed in 52.15 seconds.
🚀 Processing Log: 2025-03-20 15:45:55 Server CPU: AMD Ryzen 9 5950X, Memory: 64GB DDR4, Disk: 2TB SSD. Status: Overload, Temperature: 90°C, Alert: High.
✅ Processed in 26.51 seconds.

🔥 Final Mapped Logs:
[
    {
        "timestamp": "2025-03-20 15:40:10",
        "server": {
            "cpu": "Intel Core i9-9900K",
            "memory": "32GB DDR4",
            "disk": "256GB NVMe"
        },
        "status": {
            "state": "Down",
            "temperature": "80\u00b0C",
            "alert_level": "High"
        }
    },
    {
        "timestamp": "2025-03-20 15:45:55",
        "server": {
            "cpu": "AMD Ryzen 9 5950X",
            "memory": "64GB DDR4",
            "disk": "2TB SSD"
        },
        "status": {
            "state": "Overload",
            "temperature": "90\u00b0C",
            "alert_level": "High"
        }
    }
]

🚀 Processed 2 logs in 107.18 seconds.

----------------------------------------

 Loading FAISS index from faiss_index
🚀 Processing Log: 2025-03-20 15:40:10 Server CPU: Intel Core i9-9900K, Memory: 32GB DDR4, Disk: 256GB NVMe. Status: Down, Temperature: 80°C, Alert: High.
✅ Processed in 25.79 seconds.
🚀 Processing Log: 2025-03-20 15:45:55 Server CPU: AMD Ryzen 9 5950X, Memory: 64GB DDR4, Disk: 2TB SSD. Status: Overload, Temperature: 90°C, Alert: High.
✅ Processed in 30.59 seconds.

🔥 Final Mapped Logs:
[
    {
        "timestamp": "2025-03-20 15:40:10",
        "server": {
            "cpu": "Intel Core i9-9900K",
            "memory": "32GB DDR4",
            "disk": "256GB NVMe"
        },
        "status": {
            "state": "Down",
            "temperature": "80\u00b0C",
            "alert_level": "High"
        }
    },
    {
        "timestamp": "2025-03-20 15:45:55",
        "server": {
            "cpu": "AMD Ryzen 9 5950X",
            "memory": "64GB DDR4",
            "disk": "2TB SSD"
        },
        "status": {
            "state": "Overload",
            "temperature": "90\u00b0C",
            "alert_level": "High"
        }
    }
]

🚀 Processed 2 logs in 71.59 seconds.

Process finished with exit code 0


"""
