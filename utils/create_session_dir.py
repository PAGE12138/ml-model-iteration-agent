import os
import uuid


def create_session_output_dir(base_output_dir, user_input: str) -> str:
    """为一次建模任务创建独立输出目录。"""
    session_id = uuid.uuid4().hex
    session_dir = os.path.join(base_output_dir, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir
