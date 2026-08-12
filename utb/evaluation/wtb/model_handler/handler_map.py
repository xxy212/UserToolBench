from .api_inference.oai import OpenAIHandler
from .api_inference.deepseek import DeepSeekAPIHandler
from .api_inference.hunyuan import HunYuanAPIHandler
from .api_inference.kimi import KIMIHandler
from .api_inference.claude import ClaudeHandler
from .api_inference.qwen import QwenHandler
from .api_inference.glm import GLMHandler

api_inference_handler_map = {
    "gpt-4o-2024-11-20": OpenAIHandler,
    "deepseek-chat": DeepSeekAPIHandler,
    "hunyuan-2.0-thinking-20251109": HunYuanAPIHandler,
    "hunyuan-2.0-instruct-20251111": HunYuanAPIHandler,
    "Kimi-K2.6": KIMIHandler ,
    "deepseek-v4-flash": DeepSeekAPIHandler,
    'gpt-5.4': OpenAIHandler,
    'claude-sonnet-4-6': ClaudeHandler,
    'qwen3.6-plus' : QwenHandler ,
    'glm-4.7' : GLMHandler

}

HANDLER_MAP = {**api_inference_handler_map}
