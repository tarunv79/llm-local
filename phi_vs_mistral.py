import requests
import json
import time
from io import StringIO


def extract_attributes_with_model(log_text, model_name="mistral"):
    """Extract attributes using the specified model and measure processing time."""
    start_time = time.time()

    prompt = (
        "Extract attributes and their values from the following logs. "
        "Output as a compact, valid JSON object with no extra formatting or explanation:\n\n"
        f"{log_text}"
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.1,
        "max_tokens": 1024
    }

    with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as response:
        if response.status_code != 200:
            print(f"❌ Error with {model_name}: {response.status_code}")
            return None, 0.0

        buffer = StringIO()

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                try:
                    chunk_json = json.loads(chunk)
                    buffer.write(chunk_json.get("response", ""))
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid chunk for {model_name}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        full_response = buffer.getvalue()

        # parse JSON output
        try:
            result = json.loads(full_response)
        except json.JSONDecodeError:
            result = full_response  # Return raw output if JSON parsing fails

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

# Run with both models
mistral_output, mistral_time = extract_attributes_with_model(log_data, "mistral")
phi_output, phi_time = extract_attributes_with_model(log_data, "phi")

# Display the results
print("\n🚀 **Comparison Results:**")
print("\n🔹 Mistral Output:")
print(json.dumps(mistral_output, indent=4))
print(f"⏱️ Mistral Processing Time: {mistral_time:.2f} seconds")

print("\n🔹 Phi Output:")
print(json.dumps(phi_output, indent=4))
print(f"⏱️ Phi Processing Time: {phi_time:.2f} seconds")

# Compare processing time
if mistral_time < phi_time:
    print("\n✅ **Mistral was faster.**")
elif phi_time < mistral_time:
    print("\n✅ **Phi was faster.**")
else:
    print("\n⚖️ **Both models took the same time.**")


"""
⚠️ Skipping invalid chunk for mistral
⚠️ Skipping invalid chunk for mistral
⚠️ Skipping invalid chunk for phi
⚠️ Skipping invalid chunk for phi

🚀 **Comparison Results:**

🔹 Mistral Output:
{
    "Timestamp": "2025-03-20 15:30:45",
    "CPU": "Intel Xeon E5-2670",
    "Memory": "64GB DDR4",
    "Status": "Running",
    "Disk": "512GB SSD",
    "Temperature": "45\u00b0C"
}
⏱️ Mistral Processing Time: 16.37 seconds

🔹 Phi Output:
{
    "timestamp": "2025-03-20 15:30:45",
    "cpu": "Intel Xeon E5-26770",
    "memory": "64GB DDR4",
    "status": "Running",
    "disk": "512GB SSD",
    "temperature": "45\u00b0C"
}
⏱️ Phi Processing Time: 7.61 seconds

✅ **Phi was faster.**

Process finished with exit code 0

"""