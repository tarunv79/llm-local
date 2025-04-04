import requests
import json
import time

"""Extract attributes from logs by streaming Mistral's response with processing time measurement."""


def extract_attributes(log_text):
    start_time = time.time()

    prompt = f"Extract all key-value pairs from the following text and output them as JSON, " \
             f"only output the json with no extra text:\n\n{log_text}"
    payload = {"model": "mistral", "prompt": prompt}

    # Use streaming to handle the chunked response
    with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as response:
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            return None

        full_response = ""

        # Iterate over the streamed chunks
        for line in response.iter_lines():
            if line:
                try:
                    # Parse each JSON fragment
                    chunk = json.loads(line)
                    # Extract the "response" field
                    full_response += chunk.get("response", "")
                except json.JSONDecodeError:
                    print("Skipping invalid chunk:", line)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\nFull Combined Response:\n", full_response)
        print(f"\nProcessing Time: {elapsed_time:.2f} seconds")

        try:
            result = json.loads(full_response)
            print("\nExtracted JSON:", json.dumps(result, indent=4))
            return result, elapsed_time
        except json.JSONDecodeError:
            print("\nFailed to parse JSON. Returning raw output.")
            return full_response, elapsed_time


log_data = """
Timestamp: 2025-03-19 10:15:30
CPU=Intel Xeon
Memory: 16GB
Status=Running
"""

result, processing_time = extract_attributes(log_data)

print("\nFinal Output:")
print("Processing Time:", processing_time, "seconds")
print("Result:", result)

"""
RUN OUTPUT::

 Full Combined Response:
 {"Timestamp": "2025-03-19 10:15:30", "CPU": "Intel Xeon", "Memory": "16GB", "Status": "Running"}

️ Processing Time: 8.22 seconds

 Extracted JSON: {
    "Timestamp": "2025-03-19 10:15:30",
    "CPU": "Intel Xeon",
    "Memory": "16GB",
    "Status": "Running"
}

 Final Output:
Processing Time: 8.218227863311768 seconds
Result: {'Timestamp': '2025-03-19 10:15:30', 'CPU': 'Intel Xeon', 'Memory': '16GB', 'Status': 'Running'}

Process finished with exit code 0

"""
