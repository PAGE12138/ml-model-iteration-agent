# -*- coding: utf-8 -*-
"""机器学习模型迭代智能体命令行示例。"""

from config.llm_config import LLMConfig
from ml_model_agent import MLModelAgent


def main():
    llm_config = LLMConfig()
    agent = MLModelAgent(llm_config=llm_config, max_rounds=8)

    result = agent.build_model(
        user_input=(
            "请根据给定数据训练一个机器学习模型。先理解数据结构和任务类型，"
            "再建立可靠基线模型，并根据验证结果持续迭代优化，最后保存最佳模型和报告。"
        ),
        files=["./data.csv"],
    )

    print(result["final_report"])


if __name__ == "__main__":
    main()
