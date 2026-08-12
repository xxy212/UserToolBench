import json
import os
import time

try:
    from openai import OpenAI, RateLimitError
except ModuleNotFoundError:
    class RateLimitError(Exception):
        pass

    class OpenAI:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("openai package is required for real API calls")
from utils.openai_utils import retry_with_backoff


class QWENAPIHandler:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
        base_url = os.getenv('LOCAL_QWEN_BASE_URL')
        self.client = OpenAI(
            base_url=base_url  
        )

    @retry_with_backoff(RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    def request_model(self, messages, max_tokens=None):
        kwargs = {
            "messages": messages,
            "timeout": 300,
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        api_response, latency = self.generate_with_backoff(**kwargs)
        api_response = json.loads(api_response.json())
        choice = api_response["choices"][0]
        message = choice["message"]
        text = message["content"]
                 
        if "</think>" in text:
            text = text.split("</think>", 1)[1].lstrip()
        return text


def main():
    model_name = "qwen3-32b"
    temperature = 0.0
    from constant import DOTENV_PATH
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)                      
    handle = QWENAPIHandler(model_name, temperature)
    messages = [
        {
            "role": "user",
            "content": "Hello, who are you?"
        }
    ]
    print(json.dumps(messages, ensure_ascii=False, indent=4))
    print("---")
    result = handle.request_model(messages)
    print(result)


if __name__ == "__main__":
    main()
