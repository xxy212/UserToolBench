import json
import os
import time

from wtb.model_handler.base_handler import BaseHandler
from wtb.model_handler.utils import retry_with_backoff
from openai import OpenAI, RateLimitError


class GLMHandler(BaseHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.client = OpenAI(
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )

    @retry_with_backoff(RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    def _request_tool_call(self, inference_data):
        messages = inference_data["messages"]
        tools = inference_data["tools"]
        api_response, latency = self.generate_with_backoff(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            tools=tools
        )

        return api_response, latency

    def _parse_api_response(self, api_response):
        api_response = json.loads(api_response.json())
        choice = api_response["choices"][0]
        message = choice["message"]
        reasoning_content = message.get("reasoning_content", None)
        content = message["content"]
        tool_calls = message.get("tool_calls", None)
        input_token = api_response["usage"]["prompt_tokens"]
        output_token = api_response["usage"]["completion_tokens"]

        return {
            "reasoning_content": reasoning_content,
            "content": content,
            "tool_calls": tool_calls,
            "input_token": input_token,
            "output_token": output_token
        }
