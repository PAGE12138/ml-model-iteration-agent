# -*- coding: utf-8 -*-
"""机器学习模型迭代智能体包。"""

try:
    from .ml_model_agent import MLModelAgent, quick_ml_modeling
    from .config.llm_config import LLMConfig
    from .utils.code_executor import CodeExecutor
except ImportError:
    from ml_model_agent import MLModelAgent, quick_ml_modeling
    from config.llm_config import LLMConfig
    from utils.code_executor import CodeExecutor

__version__ = "1.0.0"
__author__ = "ML Model Iteration Agent Team"

__all__ = [
    "MLModelAgent",
    "LLMConfig",
    "CodeExecutor",
    "create_ml_agent",
    "quick_ml_modeling",
]


def create_ml_agent(config=None, output_dir="outputs", max_rounds=20):
    """创建机器学习模型迭代智能体实例。"""
    if config is None:
        config = LLMConfig()
    return MLModelAgent(llm_config=config, output_dir=output_dir, max_rounds=max_rounds)
