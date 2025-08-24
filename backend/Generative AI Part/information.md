## paylods parameters

```bash
payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write a Python function to reverse a string."}
    ],
    "temperature": 0.5,
    "max_tokens": 150,
    "top_p": 0.9,
    "stop": ["\nUser:"],
    "frequency_penalty": 0.2,
    "presence_penalty": 0.1,
    "stream": False
}

```