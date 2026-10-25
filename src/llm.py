import os

from groq import Groq

from .settings import required_env


class ChatClient:
    def __init__(self):
        self.api_key = required_env("GROQ_API_KEY")
        self.model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.client = Groq(api_key=self.api_key)

    def complete(self, messages, tools=None, temperature=0.1):
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status:
                raise RuntimeError(f"Groq API returned HTTP {status}: {error}") from error
            raise RuntimeError(f"Groq API request failed: {error}") from error
        return response.choices[0].message.model_dump(exclude_none=True)
