# -*- coding: utf-8 -*-
"""LLM 调用辅助模块。"""

import asyncio

import yaml

try:
    from ..config.llm_config import LLMConfig
    from .fallback_openai_client import AsyncFallbackOpenAIClient
except ImportError:
    from config.llm_config import LLMConfig
    from utils.fallback_openai_client import AsyncFallbackOpenAIClient


class LLMHelper:
    """OpenAI 兼容接口调用封装。"""

    def __init__(self, config: LLMConfig = None):
        self.config = config
        self.client = AsyncFallbackOpenAIClient(
            primary_api_key=config.api_key,
            primary_base_url=config.base_url,
            primary_model_name=config.model,
        )

    async def async_call(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """异步调用 LLM。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        try:
            response = await self.client.chat_completions_create(
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM 调用失败: {e}") from e

    def call(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """同步调用 LLM。"""
        return asyncio.run(self.async_call(prompt, system_prompt, max_tokens, temperature))

    def parse_yaml_response(self, response: str) -> dict:
        """解析 YAML 格式响应。"""
        try:
            if "```yaml" in response:
                start = response.find("```yaml") + 7
                end = response.find("```", start)
                yaml_content = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                yaml_content = response[start:end].strip()
            else:
                yaml_content = response.strip()

            parsed = yaml.safe_load(yaml_content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as e:
            print(f"YAML 解析失败: {e}")
            print(f"原始响应: {response}")
            return {}

    async def close(self):
        """关闭客户端连接。"""
        await self.client.close()
