import requests

url = "https://api.perplexity.ai/chat/completions"

payload = {
    "model": "sonar-deep-research",
    "messages": [
        {"role": "user", "content": "Provide an in-depth analysis of the impact of AI on global job markets over the next decade."}
    ],
    "max_tokens": 500
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())


