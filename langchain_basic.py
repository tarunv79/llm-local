import json
import time
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count


MODEL_NAME = "mistral"
CHUNK_SIZE = 5  # Logs per batch
TIMEOUT = 60  # Timeout per LLM call


logs = [
    "2025-03-20 15:30:45 Server CPU: Intel Xeon E5-2670, Memory: 64GB DDR4, Disk: 512GB SSD. Status: Running, Temperature: 45°C, Alert: None.",
    "2025-03-20 15:35:22 Server CPU: AMD EPYC 7742, Memory: 128GB DDR4, Disk: 1TB NVMe. Status: Idle, Temperature: 40°C, Alert: None.",
    "2025-03-20 15:40:10 Server CPU: Intel Core i9-9900K, Memory: 32GB DDR4, Disk: 256GB NVMe. Status: Down, Temperature: 80°C, Alert: High.",
    "2025-03-20 15:45:55 Server CPU: AMD Ryzen 9 5950X, Memory: 64GB DDR4, Disk: 2TB SSD. Status: Overload, Temperature: 90°C, Alert: High."
]


def get_llm():
    """Initialize the local Mistral model."""
    return ChatOllama(model=MODEL_NAME, temperature=0.2, timeout=TIMEOUT)


def build_prompt(log_chunk):
    """Create multi-step prompt for better context."""

    context = """
    You are an expert log parser. Your task is to map server logs to a given JSON schema.
    The logs contain information about server hardware, status, and alerts.
    Extract the relevant attributes and map them to the schema accurately.
    """

    schema = """
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
    """

    examples = """
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

    logs_str = "\n".join(log_chunk)

    request = f"""
    {context}

    {schema}

    {examples}

    Logs to Process:
    {logs_str}

    Now, process the logs and map them to the schema using the same format as the examples.
    Output the result as a JSON array.
    """

    return request


def process_chunk(chunk, llm):
    """Processes a batch of logs using multi-step prompting."""

    prompt = build_prompt(chunk)

    template = PromptTemplate(input_variables=["prompt"], template="{prompt}")

    chain = template | llm

    try:
        start_time = time.time()

        result = chain.invoke({"prompt": prompt})

        duration = time.time() - start_time
        print(f"✅ Batch processed in {duration:.2f} seconds.")

        output_text = result.content if hasattr(result, "content") else str(result)

        # Parse the output JSON safely
        try:
            parsed_output = json.loads(output_text)
        except json.JSONDecodeError:
            print("❌ Invalid JSON format, skipping batch.")
            parsed_output = []

        return parsed_output

    except Exception as e:
        print(f"❌ Exception during LLM processing: {e}")
        return []


def batch_process_logs(logs, chunk_size=CHUNK_SIZE):
    """Batch processes logs using parallel execution."""
    llm = get_llm()
    batches = [logs[i:i + chunk_size] for i in range(0, len(logs), chunk_size)]

    results = []

    max_workers = max(1, cpu_count() // 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, batch, llm) for batch in batches]

        for future in futures:
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"❌ Failed to process batch: {e}")

    return results


# ✅ Execution
start = time.time()
mapped_logs = batch_process_logs(logs)
end = time.time()

print("\n🔥 Mapped Logs to Schema:")
print(json.dumps(mapped_logs, indent=4))
print(f"\n🚀 Processed {len(logs)} logs in {end - start:.2f} seconds.")


"""
✅ Batch processed in 98.61 seconds.

🔥 Mapped Logs to Schema:
[
    {
        "timestamp": "2025-03-20 15:30:45",
        "server": {
            "cpu": "Intel Xeon E5-2670",
            "memory": "64GB DDR4",
            "disk": "512GB SSD"
        },
        "status": {
            "state": "Running",
            "temperature": "45\u00b0C",
            "alert_level": "None"
        }
    },
    {
        "timestamp": "2025-03-20 15:35:22",
        "server": {
            "cpu": "AMD EPYC 7742",
            "memory": "128GB DDR4",
            "disk": "1TB NVMe"
        },
        "status": {
            "state": "Idle",
            "temperature": "40\u00b0C",
            "alert_level": "None"
        }
    },
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

🚀 Processed 4 logs in 98.64 seconds.

Process finished with exit code 0

✅ Possible Optimizations to Improve Performance
- Persistent LLM Instance with Pooled Execution:
- Instead of creating a new ChatOllama() instance for each batch, create a single LLM instance and reuse it.
This reduces model initialization overhead significantly.
- Batch Larger Chunks:
- Reduce the number of LLM calls by increasing the batch size.
- Instead of processing 4 logs as 4 individual batches, process them in a single LLM call.
- Use ThreadPoolExecutor with max_workers equal to the number of CPU cores. This balances parallel execution 
without overwhelming machine.

"""