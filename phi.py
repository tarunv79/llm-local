import requests
import json
import time
from io import StringIO


def extract_attributes(log_text, model_name="phi", temperature=0.1, max_tokens=1024):
    """Extract attributes using Phi with strict JSON enforcement."""

    start_time = time.time()

    # Improved strict prompt
    prompt = (
        "Extract attributes and their values from the following logs. "
        "Output as a compact, valid JSON object with no extra formatting or explanation:\n\n"
        f"{log_text}"
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as response:
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return None, 0.0

        buffer = StringIO()

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                try:
                    chunk_json = json.loads(chunk)
                    buffer.write(chunk_json.get("response", ""))
                except json.JSONDecodeError:
                    continue

        elapsed_time = time.time() - start_time

        full_response = buffer.getvalue().strip()

        try:
            # Locate the first and last valid JSON objects in case of extra content
            start_idx = full_response.find("{")
            end_idx = full_response.rfind("}") + 1

            if start_idx != -1 and end_idx != -1:
                valid_json = full_response[start_idx:end_idx]
                result = json.loads(valid_json)
            else:
                result = {"error": "Invalid JSON format"}

        except json.JSONDecodeError:
            result = {"error": "Failed to parse JSON"}

        return result, elapsed_time


# Sample log data
log_data = """
Timestamp: 2025-03-20 15:30:45
CPU: Intel Xeon E5-2670
Memory: 64GB DDR4
Status: Running
Disk: 512GB SSD
Temperature: 45°C
"""

# Extract attributes
output, processing_time = extract_attributes(log_data)

# Display results
if output:
    print(json.dumps(output, indent=4))  # Clean JSON output
print(f"\n️ {processing_time:.2f} seconds")


"""
1. phi

 {
    "timestamp": "2025-03-20 15:30:45",
    "cpu": "Intel Xeon E5-2670",
    "memory": "64GB DDR4",
    "status": "Running",
    "disk": "512GB SSD",
    "temperature": "45\u00b0C"
}

 4.51 seconds
===================================

2. mistral
{
    "Timestamp": "2025-03-20 15:30:45",
    "CPU": "Intel Xeon E5-2670",
    "Memory": "64GB DDR4",
    "Status": "Running",
    "Disk": "512GB SSD",
    "Temperature": "45\u00b0C"
}

 27.76 seconds
"""