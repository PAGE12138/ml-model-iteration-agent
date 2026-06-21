# -*- coding: utf-8 -*-
"""机器学习模型迭代智能体。"""

import os
from typing import Any, Callable, Dict, List

try:
    from .config.llm_config import LLMConfig
    from .prompts import ml_model_system_prompt
    from .utils.code_executor import CodeExecutor
    from .utils.create_session_dir import create_session_output_dir
    from .utils.extract_code import extract_code_from_response
    from .utils.format_execution_result import format_execution_result
    from .utils.llm_helper import LLMHelper
except ImportError:
    from config.llm_config import LLMConfig
    from prompts import ml_model_system_prompt
    from utils.code_executor import CodeExecutor
    from utils.create_session_dir import create_session_output_dir
    from utils.extract_code import extract_code_from_response
    from utils.format_execution_result import format_execution_result
    from utils.llm_helper import LLMHelper


class MLModelAgent:
    """根据用户需求生成、训练、评估并迭代机器学习模型。"""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        output_dir: str = "outputs",
        max_rounds: int = 20,
    ):
        self.config = llm_config or LLMConfig()
        self.llm = LLMHelper(self.config)
        self.base_output_dir = output_dir
        self.max_rounds = max_rounds
        self.conversation_history = []
        self.iteration_results = []
        self.current_round = 0
        self.session_output_dir = None
        self.executor = None

    def build_model(
        self,
        user_input: str,
        files: List[str] = None,
        target: str = None,
        metric: str = None,
        expected_performance: str = None,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
        should_stop: Callable[[], bool] = None,
    ) -> Dict[str, Any]:
        """开始一次机器学习模型迭代任务。"""
        self._reset_run_state()
        self.session_output_dir = create_session_output_dir(
            self.base_output_dir,
            user_input,
        )
        self.executor = CodeExecutor(self.session_output_dir)
        self.executor.set_variable("session_output_dir", self.session_output_dir)

        initial_prompt = self._build_initial_prompt(
            user_input=user_input,
            files=files,
            target=target,
            metric=metric,
            expected_performance=expected_performance,
        )

        print("开始机器学习模型迭代任务")
        print(f"用户需求: {user_input}")
        if files:
            print(f"数据文件: {', '.join(files)}")
        if target:
            print(f"目标列: {target}")
        if metric:
            print(f"评价指标: {metric}")
        if expected_performance:
            print(f"预期性能: {expected_performance}")
        print(f"输出目录: {self.session_output_dir}")
        print(f"最大轮数: {self.max_rounds}")
        print("=" * 60)

        self.conversation_history.append({"role": "user", "content": initial_prompt})
        self._emit_progress(progress_callback, {
            "type": "started",
            "round": 0,
            "message": "机器学习模型迭代任务已启动。",
            "session_output_dir": self.session_output_dir,
        })

        final_report = ""
        while self.current_round < self.max_rounds:
            if should_stop and should_stop():
                final_report = "任务已由用户停止。"
                self._emit_progress(progress_callback, {
                    "type": "stopped",
                    "round": self.current_round,
                    "message": "任务已停止。",
                    "final_report": final_report,
                })
                break

            self.current_round += 1
            print(f"\n第 {self.current_round} 轮建模")
            self._emit_progress(progress_callback, {
                "type": "round_started",
                "round": self.current_round,
                "message": f"第 {self.current_round} 轮建模开始。",
            })

            try:
                notebook_variables = self.executor.get_environment_info()
                formatted_system_prompt = ml_model_system_prompt.format(
                    notebook_variables=notebook_variables,
                )
                response = self.llm.call(
                    prompt=self._build_conversation_prompt(),
                    system_prompt=formatted_system_prompt,
                )
                if not response or not response.strip():
                    raise RuntimeError(
                        "LLM 返回了空内容，请检查 API Key、Base URL、模型名和 max_tokens 配置。"
                    )
                if should_stop and should_stop():
                    final_report = "任务已由用户停止。"
                    self._emit_progress(progress_callback, {
                        "type": "stopped",
                        "round": self.current_round,
                        "message": "任务已停止。",
                        "final_report": final_report,
                    })
                    break

                print(f"助手响应:\n{response}")
                process_result = self._process_response(response)
                progress_payload = self._build_progress_payload(process_result)
                progress_payload.update({
                    "type": "round_finished",
                    "round": self.current_round,
                    "assistant_response": response,
                })
                self._emit_progress(progress_callback, progress_payload)

                if not process_result.get("continue", True):
                    final_report = process_result.get("final_report", "")
                    print("\n模型迭代完成。")
                    break

                self.conversation_history.append({"role": "assistant", "content": response})

                if process_result["action"] == "generate_code":
                    feedback = process_result.get("feedback", "")
                    self.conversation_history.append({
                        "role": "user",
                        "content": f"代码执行反馈:\n{feedback}",
                    })
                    self.iteration_results.append({
                        "round": self.current_round,
                        "code": process_result.get("code", ""),
                        "result": process_result.get("result", {}),
                        "response": response,
                    })
                elif process_result["action"] == "invalid_response":
                    self.conversation_history.append({
                        "role": "user",
                        "content": "响应中缺少可执行代码，请严格按 YAML 格式重新生成。",
                    })

            except Exception as e:
                error_msg = f"LLM 调用或建模流程错误: {str(e)}"
                print(error_msg)
                self._emit_progress(progress_callback, {
                    "type": "round_error",
                    "round": self.current_round,
                    "message": error_msg,
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": f"发生错误: {error_msg}。请基于错误重新生成代码。",
                })

        if not final_report:
            if self.current_round >= self.max_rounds:
                print(f"\n已达到最大轮数 {self.max_rounds}，生成当前最佳结果报告。")
            final_report = self._generate_final_report()

        report_file_path = self._save_final_report(final_report)
        self._emit_progress(progress_callback, {
            "type": "finished",
            "round": self.current_round,
            "message": "机器学习模型迭代任务已完成。",
            "final_report": final_report,
            "report_file_path": report_file_path,
        })

        return {
            "session_output_dir": self.session_output_dir,
            "total_rounds": self.current_round,
            "iteration_results": self.iteration_results,
            "conversation_history": self.conversation_history,
            "final_report": final_report,
            "report_file_path": report_file_path,
        }

    def _process_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 响应并执行对应动作。"""
        try:
            yaml_data = self.llm.parse_yaml_response(response)
            action = yaml_data.get("action", "generate_code") if yaml_data else "generate_code"
            print(f"检测到动作: {action}")

            if action == "model_complete":
                return {
                    "action": "model_complete",
                    "final_report": yaml_data.get("final_report", "模型迭代完成，但未返回最终报告。"),
                    "response": response,
                    "continue": False,
                }
            if action == "generate_code":
                return self._handle_generate_code(response, yaml_data)

            print(f"未知动作类型: {action}，按 generate_code 处理。")
            return self._handle_generate_code(response, yaml_data)
        except Exception as e:
            print(f"解析响应失败: {str(e)}，按 generate_code 处理。")
            return self._handle_generate_code(response, {})

    def _handle_generate_code(self, response: str, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理代码生成动作。"""
        code = yaml_data.get("code", "") if yaml_data else ""
        if not code:
            code = extract_code_from_response(response)

        if not code:
            print("未从响应中提取到可执行代码，要求 LLM 重新生成。")
            return {
                "action": "invalid_response",
                "error": "响应中缺少可执行代码",
                "assistant_response": response,
                "response": response,
                "continue": True,
            }

        print(f"执行代码:\n{code}")
        print("-" * 40)
        result = self.executor.execute_code(code)
        feedback = format_execution_result(result)
        print(f"执行反馈:\n{feedback}")

        return {
            "action": "generate_code",
            "code": code,
            "result": result,
            "feedback": feedback,
            "response": response,
            "continue": True,
        }

    def _build_progress_payload(self, process_result: Dict[str, Any]) -> Dict[str, Any]:
        """把内部结果转换为前端可消费的进度数据。"""
        result = process_result.get("result", {}) or {}
        return {
            "action": process_result.get("action"),
            "code": process_result.get("code", ""),
            "success": result.get("success"),
            "output": result.get("output", ""),
            "error": result.get("error", ""),
            "variables": result.get("variables", {}),
            "feedback": process_result.get("feedback", ""),
            "final_report": process_result.get("final_report", ""),
            "assistant_response": process_result.get("assistant_response")
            or process_result.get("response", ""),
        }

    def _emit_progress(
        self,
        progress_callback: Callable[[Dict[str, Any]], None],
        payload: Dict[str, Any],
    ):
        """向外部任务管理器发送进度事件。"""
        if not progress_callback:
            return
        try:
            progress_callback(payload)
        except Exception as e:
            print(f"进度回调失败: {e}")

    def _generate_final_report(self) -> str:
        """最大轮数结束或未显式完成时，生成当前最佳模型总结。"""
        prompt = self._build_final_report_prompt()

        try:
            response = self.llm.call(
                prompt=prompt,
                system_prompt="请基于机器学习模型迭代过程生成最终报告。必须诚实总结已执行结果，不要虚构指标。",
                max_tokens=16384,
            )
            try:
                yaml_data = self.llm.parse_yaml_response(response)
                if yaml_data.get("action") == "model_complete":
                    return yaml_data.get("final_report", response)
            except Exception:
                pass
            return response
        except Exception as e:
            return f"模型报告生成失败: {str(e)}"

    def _build_initial_prompt(
        self,
        user_input: str,
        files: List[str] = None,
        target: str = None,
        metric: str = None,
        expected_performance: str = None,
    ) -> str:
        """构建初始建模需求提示。"""
        prompt_parts = [f"用户建模需求: {user_input}"]
        if files:
            prompt_parts.append(f"数据文件: {', '.join(files)}")
        if target:
            prompt_parts.append(f"目标列: {target}")
        if metric:
            prompt_parts.append(f"主要评价指标: {metric}")
        if expected_performance:
            prompt_parts.append(f"预期性能阈值: {expected_performance}")
        prompt_parts.append(
            "请先探索数据，再建立基线模型。如果未达到预期，请继续迭代改进并保存最佳模型。"
        )
        return "\n".join(prompt_parts)

    def _build_conversation_prompt(self) -> str:
        """构建完整对话上下文。"""
        prompt_parts = []
        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"用户: {content}")
            else:
                prompt_parts.append(f"助手: {content}")
        return "\n\n".join(prompt_parts)

    def _build_final_report_prompt(self) -> str:
        """构建最终建模报告提示。"""
        successful_outputs = []
        for result in self.iteration_results:
            exec_result = result.get("result", {})
            if exec_result.get("success"):
                output = exec_result.get("output", "")
                if output:
                    successful_outputs.append(
                        f"第 {result.get('round')} 轮输出:\n{output}"
                    )

        outputs_summary = "\n\n".join(successful_outputs[-8:])
        return f"""请基于以下机器学习模型迭代过程生成最终报告。
输出目录: {self.session_output_dir}
总轮数: {self.current_round}

关键执行输出:
{outputs_summary or "暂无成功执行输出。"}

请严格使用 YAML 格式：
```yaml
action: "model_complete"
final_report: |
  # 机器学习模型迭代报告
  ...
```
报告中必须说明最佳模型、核心指标、模型文件路径、是否达到预期，以及后续改进建议。"""

    def _save_final_report(self, final_report: str) -> str:
        """保存最终报告。"""
        report_file_path = os.path.join(self.session_output_dir, "机器学习模型迭代报告.md")
        try:
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"最终报告已保存至: {report_file_path}")
        except Exception as e:
            print(f"保存报告文件失败: {str(e)}")
        return report_file_path

    def _reset_run_state(self):
        """重置单次运行状态。"""
        self.conversation_history = []
        self.iteration_results = []
        self.current_round = 0
        self.session_output_dir = None
        self.executor = None

    def reset(self):
        """重置智能体和执行环境。"""
        if self.executor:
            self.executor.reset_environment()
        self._reset_run_state()


def quick_ml_modeling(
    query: str,
    files: List[str] = None,
    target: str = None,
    metric: str = None,
    expected_performance: str = None,
    output_dir: str = "outputs",
    max_rounds: int = 12,
) -> Dict[str, Any]:
    """快速创建并运行机器学习模型迭代任务。"""
    agent = MLModelAgent(output_dir=output_dir, max_rounds=max_rounds)
    return agent.build_model(
        user_input=query,
        files=files,
        target=target,
        metric=metric,
        expected_performance=expected_performance,
    )
