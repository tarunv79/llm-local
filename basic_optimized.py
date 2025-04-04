import requests
import json
import time
from io import StringIO

"""Optimized extraction with enhanced prompt, lower temperature, and better streaming handling."""


def extract_attributes_optimized(log_text):
    start_time = time.time()

    # Enhanced prompt with specific instructions
    prompt = (
        "Extract attributes and their values from the following logs. "
        "Output as a compact, valid JSON object with no extra formatting or explanation:\n\n"
        f"{log_text}"
    )

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "temperature": 0.1,  # More deterministic output
        "max_tokens": 500    # Limit output length
    }

    # Use streaming for efficient processing
    with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as response:
        if response.status_code != 200:
            print("❌ Error:", response.status_code, response.text)
            return None

        # Use StringIO for faster concatenation
        buffer = StringIO()

        # Iterate over streaming content
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                try:
                    # Parse each chunk as JSON
                    chunk_json = json.loads(chunk)
                    buffer.write(chunk_json.get("response", ""))
                except json.JSONDecodeError:
                    print("⚠️ Skipping invalid chunk:", chunk)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Final assembled output
        full_response = buffer.getvalue()

        print("\n🔹 Full Combined Response:\n", full_response)
        print(f"\n⏱️ Processing Time: {elapsed_time:.2f} seconds")

        # Attempt to parse the final JSON
        try:
            result = json.loads(full_response)
            print("\n✅ Extracted JSON:", json.dumps(result, indent=4))
            return result, elapsed_time
        except json.JSONDecodeError:
            print("\n❌ Failed to parse JSON. Returning raw output.")
            return full_response, elapsed_time


# ✅ Test with sample log data
log_data = """
Timestamp: 2025-03-19 10:15:30
CPU=Intel Xeon
Memory: 16GB
Status=Running
"""

# Execute extraction and capture the processing time
result, processing_time = extract_attributes_optimized(log_data)

# Display the result and time
print("\n🚀 Final Output:")
print("Processing Time:", processing_time, "seconds")
print("Result:", result)


"""
PROGRAM OUTPUT:


🔹 Full Combined Response:
  {
      "Timestamp": "2025-03-19 10:15:30",
      "CPU": "Intel Xeon",
      "Memory": "16GB",
      "Status": "Running"
   }

⏱️ Processing Time: 14.58 seconds

✅ Extracted JSON: {
    "Timestamp": "2025-03-19 10:15:30",
    "CPU": "Intel Xeon",
    "Memory": "16GB",
    "Status": "Running"
}

🚀 Final Output:
Processing Time: 14.583279132843018 seconds
Result: {'Timestamp': '2025-03-19 10:15:30', 'CPU': 'Intel Xeon', 'Memory': '16GB', 'Status': 'Running'}

"""