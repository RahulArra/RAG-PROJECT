import requests


class LLM:

    def __init__(self):

        self.url = "http://localhost:11434/api/generate"

        self.model = "qwen3:8b"

    def generate_stream(self, prompt):
        import json
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")