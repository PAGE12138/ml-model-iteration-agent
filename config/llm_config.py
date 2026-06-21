# -*- coding: utf-8 -*-
"""大模型接口配置。"""

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """OpenAI 兼容大模型配置。"""

    provider: str = "openai"
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")
    temperature: float = 0.1
    max_tokens: int = 16384

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """从字典创建配置。"""
        return cls(**data)

    def validate(self) -> bool:
        """验证配置是否完整。"""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL is required")
        if not self.model:
            raise ValueError("OPENAI_MODEL is required")
        return True
