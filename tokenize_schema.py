import json
import requests
import tiktoken
import time

MODEL_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "mistral"

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


def build_prompt_with_schema_and_logs(schema_tokens, log_tokens):
    return (
        "Extract attributes and their values from the following logs according to the provided schema.\n"
        "Strictly output as a valid, compact JSON array with no explanation or extra formatting.\n"
        "If the JSON cannot be constructed, return an empty JSON array []\n\n"
        "# Schema Tokens:\n"
        f"{schema_tokens}\n\n"
        "# Logs:\n"
        f"{log_tokens}\n\n"
        "# Output format example:\n"
        "[\n"
        "  {\"timestamp\": \"2025-03-20 15:30:45\", \"cpu\": \"Intel Xeon E5-2670\", \"memory\": \"64GB DDR4\", "
        "\"status\": \"Running\", \"disk\": \"512GB SSD\", \"temperature\": \"45°C\"},\n"
        "  {\"timestamp\": \"2025-03-20 15:35:22\", \"cpu\": \"AMD EPYC 7742\", \"memory\": \"128GB DDR4\", "
        "\"status\": \"Idle\", \"disk\": \"1TB NVMe\", \"temperature\": \"40°C\"}\n"
        "]\n\n"
        "# Important:\n"
        "- Only return valid JSON. Do not describe or explain anything.\n"
        "- No markdown, code blocks, or comments.\n"
    )


# Pre-tokenizes the schema using the LLM tokenizer.
def pre_tokenize(schema):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(schema)
    print("Schema Tokens :: ")
    print(tokens)
    return tokens


pre_tokenized_schema = pre_tokenize(yaml_schema)


def safe_json_parse(response):
    """
    Safely extract and parses JSON from the LLM response.
    Handle cases where the JSON is surrounded by extra text or in code blocks.
    """
    response = response.strip()

    # Remove markdown code block symbols if present
    if response.startswith("```json") and response.endswith("```"):
        response = response[7:-3].strip()

    print("\n🔍 Raw Output from Model:\n", response)

    # Attempt to parse JSON
    try:
        parsed = json.loads(response)
        print("\n✅ Valid JSON Output:\n", json.dumps(parsed, indent=4))
        return parsed
    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to parse JSON: {e}")
        return None


def parse_logs_with_schema(logs, schema_tokens):
    """Send logs with pre-tokenized schema for faster processing."""

    prompt = build_prompt_with_schema_and_logs(schema_tokens, logs)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a log parsing assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    }

    start_time = time.time()

    response = requests.post(MODEL_URL, json=payload)

    duration = time.time() - start_time

    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]["content"]
        parsed_output = safe_json_parse(result)
        if parsed_output:
            print("\n✅ Parsed JSON Output:\n", json.dumps(parsed_output, indent=4))
        else:
            print("\n❌ Failed to extract attributes.")

        print("\n✅ Parsed JSON Output:\n", json.dumps(parsed_output, indent=4))
        print(f"\n⏱️ Response Time: {duration:.2f} seconds")
    else:
        print(f"❌ Failed with status {response.status_code}: {response.text}")


parse_logs_with_schema(log_text, pre_tokenized_schema)

"""
Schema Tokens :: 
[198, 4120, 512, 220, 11695, 25, 925, 198, 220, 12035, 512, 262, 17769, 25, 925, 198, 262, 5044, 25, 925, 198, 262, 13668, 25, 925, 198, 220, 2704, 512, 262, 1614, 25, 925, 198, 262, 9499, 25, 925, 198]

🔍 Raw Output from Model:
 [{"timestamp": "2025-03-20 15:30:45", "cpu": "Intel Xeon E5-2670", "memory": "64GB DDR4", "status": "Running", "disk": "512GB SSD", "temperature": "45°C"}, {"timestamp": "2025-03-20 15:35:22", "cpu": "AMD EPYC 7742", "memory": "128GB DDR4", "status": "Idle", "disk": "1TB NVMe", "temperature": "40°C"}]

✅ Valid JSON Output:
 [
    {
        "timestamp": "2025-03-20 15:30:45",
        "cpu": "Intel Xeon E5-2670",
        "memory": "64GB DDR4",
        "status": "Running",
        "disk": "512GB SSD",
        "temperature": "45\u00b0C"
    },
    {
        "timestamp": "2025-03-20 15:35:22",
        "cpu": "AMD EPYC 7742",
        "memory": "128GB DDR4",
        "status": "Idle",
        "disk": "1TB NVMe",
        "temperature": "40\u00b0C"
    }
]

✅ Parsed JSON Output:
 [
    {
        "timestamp": "2025-03-20 15:30:45",
        "cpu": "Intel Xeon E5-2670",
        "memory": "64GB DDR4",
        "status": "Running",
        "disk": "512GB SSD",
        "temperature": "45\u00b0C"
    },
    {
        "timestamp": "2025-03-20 15:35:22",
        "cpu": "AMD EPYC 7742",
        "memory": "128GB DDR4",
        "status": "Idle",
        "disk": "1TB NVMe",
        "temperature": "40\u00b0C"
    }
]

✅ Parsed JSON Output:
 [
    {
        "timestamp": "2025-03-20 15:30:45",
        "cpu": "Intel Xeon E5-2670",
        "memory": "64GB DDR4",
        "status": "Running",
        "disk": "512GB SSD",
        "temperature": "45\u00b0C"
    },
    {
        "timestamp": "2025-03-20 15:35:22",
        "cpu": "AMD EPYC 7742",
        "memory": "128GB DDR4",
        "status": "Idle",
        "disk": "1TB NVMe",
        "temperature": "40\u00b0C"
    }
]

⏱️ Response Time: 46.37 seconds

Process finished with exit code 0

"""
