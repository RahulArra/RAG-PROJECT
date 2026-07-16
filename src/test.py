# from generation.llm import LLM

# llm = LLM()

# answer = llm.generate(
#     "What is Java?"
# )

# print(answer)




import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:8b",
        "prompt": "What is Java?",
        "stream": False
    }
)

answer = response.json()["response"]

print(answer)