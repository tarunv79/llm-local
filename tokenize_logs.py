import json
import time
import tiktoken
import requests
import re


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral"

yaml_schema = """
server:
  timestamp: string
  hardware:
    cpu: string
    memory: string
    disk: string
  status:
    state: string
    temperature: string
"""

log_text = """
[2025-03-20 15:30:45] CPU: Intel Xeon E5-2670, Memory: 64GB DDR4, Status: Running, Disk: 512GB SSD, Temperature: 45°C
[2025-03-20 15:35:22] CPU: AMD EPYC 7742, Memory: 128GB DDR4, Status: Idle, Disk: 1TB NVMe, Temperature: 40°C
"""


# Tokenizer function
def tokenize_text(text):
    """Tokenize the logs using Tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return tokens


# Tokenize the logs
log_tokens = tokenize_text(log_text)
print("\n🔹 Log Tokens:", log_tokens)
print("\n🔹 Log Tokens Count:", len(log_tokens))


prompt = f"""
You are a log parsing AI.
Your task is to extract attributes from tokenized logs according to the given schema.

# Schema:
{yaml_schema}

# Tokenized Logs:
{log_text}

# Instructions:
- Extract attributes and format them into JSON according to the schema.
- Ensure valid JSON formatting without explanations.
"""


def send_request(prompt):
    """Send prompt to the local Mistral model and get the response."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

    return response.json()


def extract_json(raw_output):
    """Extract valid JSON from the model's raw output and decode UTF-8 properly."""
    if not raw_output or "message" not in raw_output:
        return None

    content = raw_output["message"]["content"]

    content = content.replace('```json', '').replace('```', '').strip()

    # Use looser regex matching for both raw and wrapped JSON
    json_match = re.search(r'(\{.*?\}|\[.*?\])', content, re.DOTALL)

    if json_match:
        try:
            # Double Unicode decoding process
            clean_json = json_match.group(0)

            # Step 1: Decode Unicode escape sequences
            clean_json = clean_json.encode().decode('unicode_escape')

            # Step 2: Decode it again as UTF-8 to handle double encoding
            clean_json = clean_json.encode('latin1').decode('utf-8')

            # Validate and parse the JSON
            json_output = json.loads(clean_json)

            # Pretty-print the final decoded JSON
            return json.dumps(json_output, indent=4, ensure_ascii=False)

        except json.JSONDecodeError as e:
            print(f"\n❌ JSON Decode Error: {e}")
            return None

    return None


# Main Execution
start_time = time.time()

print("\n⏱️ Sending request...")
raw_output = send_request(prompt)

if raw_output:
    print("\n✅ Raw Output from Model:")
    print(json.dumps(raw_output, indent=4))

    parsed_output = extract_json(raw_output)

    if parsed_output:
        print("\n✅ Extracted JSON Output:\n", parsed_output)
    else:
        print("\n❌ Failed to extract JSON output.")
else:
    print("\n❌ No valid response received.")

end_time = time.time()
print(f"\n⏱️ Response Time: {end_time - start_time:.2f} seconds")


"""

🔹 Log Tokens: [198, 58, 2366, 20, 12, 2839, 12, 508, 220, 868, 25, 966, 25, 1774, 60, 14266, 25, 15984, 1630, 66130, 469, 20, 12, 16567, 15, 11, 14171, 25, 220, 1227, 5494, 44860, 19, 11, 8266, 25, 29125, 11, 39968, 25, 220, 8358, 5494, 37462, 11, 38122, 25, 220, 1774, 32037, 198, 58, 2366, 20, 12, 2839, 12, 508, 220, 868, 25, 1758, 25, 1313, 60, 14266, 25, 25300, 19613, 93944, 220, 24472, 17, 11, 14171, 25, 220, 4386, 5494, 44860, 19, 11, 8266, 25, 71400, 11, 39968, 25, 220, 16, 32260, 25464, 7979, 11, 38122, 25, 220, 1272, 32037, 198]

🔹 Log Tokens Count: 100

⏱️ Sending request...

✅ Raw Output from Model:
{
    "model": "mistral",
    "created_at": "2025-04-01T13:17:35.746612Z",
    "message": {
        "role": "assistant",
        "content": " ```json\n{\n  \"server\": {\n    \"timestamp\": \"198-58-2366-20-12\",\n    \"hardware\": {\n      \"cpu\": \"2839\",\n      \"memory\": \"12\",\n      \"disk\": \"220\"\n    },\n    \"status\": {\n      \"state\": \"25\",\n      \"temperature\": \"66130\"\n    }\n  }\n}\n```"
    },
    "done_reason": "stop",
    "done": true,
    "total_duration": 62301523378,
    "load_duration": 13146477,
    "prompt_eval_count": 649,
    "prompt_eval_duration": 40409000000,
    "eval_count": 114,
    "eval_duration": 21878000000
}

❌ JSON Decode Error: Expecting ',' delimiter: line 8 column 6 (char 140)

❌ Failed to extract JSON output.

⏱️ Response Time: 62.32 seconds

==================================
## RUN 2 without tokens::

🔹 Log Tokens: [198, 58, 2366, 20, 12, 2839, 12, 508, 220, 868, 25, 966, 25, 1774, 60, 14266, 25, 15984, 1630, 66130, 469, 20, 12, 16567, 15, 11, 14171, 25, 220, 1227, 5494, 44860, 19, 11, 8266, 25, 29125, 11, 39968, 25, 220, 8358, 5494, 37462, 11, 38122, 25, 220, 1774, 32037, 198, 58, 2366, 20, 12, 2839, 12, 508, 220, 868, 25, 1758, 25, 1313, 60, 14266, 25, 25300, 19613, 93944, 220, 24472, 17, 11, 14171, 25, 220, 4386, 5494, 44860, 19, 11, 8266, 25, 71400, 11, 39968, 25, 220, 16, 32260, 25464, 7979, 11, 38122, 25, 220, 1272, 32037, 198]

🔹 Log Tokens Count: 100

⏱️ Sending request...

✅ Raw Output from Model:
{
    "model": "mistral",
    "created_at": "2025-04-01T13:12:41.721506Z",
    "message": {
        "role": "assistant",
        "content": " [\n    {\n      \"timestamp\": \"2025-03-20 15:30:45\",\n      \"hardware\": {\n        \"cpu\": \"Intel Xeon E5-2670\",\n        \"memory\": \"64GB DDR4\",\n        \"disk\": \"512GB SSD\"\n      },\n      \"status\": {\n        \"state\": \"Running\",\n        \"temperature\": \"45\u00b0C\"\n      }\n    },\n    {\n      \"timestamp\": \"2025-03-20 15:35:22\",\n      \"hardware\": {\n        \"cpu\": \"AMD EPYC 7742\",\n        \"memory\": \"128GB DDR4\",\n        \"disk\": \"1TB NVMe\"\n      },\n      \"status\": {\n        \"state\": \"Idle\",\n        \"temperature\": \"40\u00b0C\"\n      }\n    }\n   ]"
    },
    "done_reason": "stop",
    "done": true,
    "total_duration": 42554163757,
    "load_duration": 10455331,
    "prompt_eval_count": 257,
    "prompt_eval_duration": 162000000,
    "eval_count": 236,
    "eval_duration": 42380000000
}

✅ Extracted JSON Output:
 [
    {
        "timestamp": "2025-03-20 15:30:45",
        "hardware": {
            "cpu": "Intel Xeon E5-2670",
            "memory": "64GB DDR4",
            "disk": "512GB SSD"
        },
        "status": {
            "state": "Running",
            "temperature": "45°C"
        }
    },
    {
        "timestamp": "2025-03-20 15:35:22",
        "hardware": {
            "cpu": "AMD EPYC 7742",
            "memory": "128GB DDR4",
            "disk": "1TB NVMe"
        },
        "status": {
            "state": "Idle",
            "temperature": "40°C"
        }
    }
]

⏱️ Response Time: 42.57 seconds

Process finished with exit code 0
"""