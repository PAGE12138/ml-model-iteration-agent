# -*- coding: utf-8 -*-
"""基于 IPython 的受控 Python 代码执行器。"""

import ast
import importlib.util
import os
import traceback
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


class CodeExecutor:
    """在 Notebook 风格环境中执行 LLM 生成的建模代码。"""

    ALLOWED_IMPORTS = {
        "pandas", "pd",
        "numpy", "np",
        "matplotlib", "matplotlib.pyplot", "plt",
        "duckdb", "scipy", "sklearn", "joblib",
        "plotly", "dash", "requests", "urllib",
        "os", "sys", "json", "csv", "datetime", "time",
        "math", "statistics", "re", "pathlib", "io",
        "collections", "itertools", "functools", "operator",
        "warnings", "logging", "copy", "pickle", "gzip", "zipfile",
        "typing", "dataclasses", "enum", "sqlite3",
    }
    ALLOWED_IMPORT_PREFIXES = {
        "matplotlib",
        "scipy",
        "sklearn",
        "plotly",
        "urllib",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "typing",
    }

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.shell = InteractiveShell.instance()
        self._setup_chinese_font()
        self._setup_common_imports()

    def _setup_chinese_font(self):
        """设置 matplotlib 中文显示和非交互后端。"""
        try:
            matplotlib.use("Agg")
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
            plt.rcParams["axes.unicode_minus"] = False
            self.shell.run_cell(
                """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
"""
            )
        except Exception as e:
            print(f"设置中文字体失败: {e}")

    def _setup_common_imports(self):
        """预导入常用数据科学库。"""
        common_imports = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from IPython.display import display
"""
        try:
            self.shell.run_cell(common_imports)
            optional_imports = {"duckdb": "import duckdb"}
            for package_name, import_code in optional_imports.items():
                if importlib.util.find_spec(package_name) is None:
                    print(f"可选库未安装，跳过预导入: {package_name}")
                    continue
                result = self.shell.run_cell(import_code, silent=True)
                if result.error_in_exec or result.error_before_exec:
                    print(f"可选库预导入失败，跳过: {package_name}")

            from IPython.display import display

            self.shell.user_ns["display"] = display
        except Exception as e:
            print(f"预导入库失败: {e}")

    def _is_allowed_import(self, module_name: str) -> bool:
        """判断导入模块是否在允许列表或允许前缀下。"""
        if not module_name:
            return False
        if module_name in self.ALLOWED_IMPORTS:
            return True
        return any(module_name.startswith(f"{prefix}.") for prefix in self.ALLOWED_IMPORT_PREFIXES)

    def _is_read_only_open_call(self, node: ast.Call) -> bool:
        """只允许只读方式打开文件。"""
        mode = "r"
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        for keyword in node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                mode = str(keyword.value.value)
        return not any(flag in mode for flag in ("w", "a", "x", "+"))

    def _is_open_call(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
            return True
        return False

    def _check_code_safety(self, code: str) -> Tuple[bool, str]:
        """检查代码安全性。"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_allowed_import(alias.name):
                        return False, f"不允许的导入: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if not self._is_allowed_import(node.module):
                    return False, f"不允许的导入: {node.module}"
            elif isinstance(node, ast.Call):
                if self._is_open_call(node):
                    if not self._is_read_only_open_call(node):
                        return False, "不允许以写入或追加模式打开文件"
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in ["exec", "eval", "__import__"]:
                    return False, f"不允许的函数调用: {node.func.id}"

        return True, ""

    def get_current_figures_info(self) -> List[Dict[str, Any]]:
        """获取当前 matplotlib 图形信息。"""
        figures_info = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            if fig.get_axes():
                figures_info.append({
                    "figure_number": fig_num,
                    "axes_count": len(fig.get_axes()),
                    "figure_size": fig.get_size_inches().tolist(),
                    "has_content": True,
                })
        return figures_info

    def _format_table_output(self, obj: Any) -> str:
        """格式化表格输出，避免超长结果刷屏。"""
        if hasattr(obj, "shape") and hasattr(obj, "head"):
            rows, cols = obj.shape
            print(f"\n数据表形状: {rows} 行 x {cols} 列")
            print(f"列名: {list(obj.columns)}")

            if rows <= 15:
                return str(obj)
            head_part = obj.head(5)
            tail_part = obj.tail(5)
            return f"{head_part}\n...\n(省略 {rows - 10} 行)\n...\n{tail_part}"

        return str(obj)

    def execute_code(self, code: str) -> Dict[str, Any]:
        """执行代码并返回结构化结果。"""
        is_safe, safety_error = self._check_code_safety(code)
        if not is_safe:
            return {
                "success": False,
                "output": "",
                "error": f"代码安全检查失败: {safety_error}",
                "variables": {},
            }

        vars_before = set(self.shell.user_ns.keys())

        try:
            with capture_output() as captured:
                result = self.shell.run_cell(code)

            if result.error_before_exec:
                return {
                    "success": False,
                    "output": captured.stdout,
                    "error": f"执行前错误: {result.error_before_exec}",
                    "variables": {},
                }

            if result.error_in_exec:
                return {
                    "success": False,
                    "output": captured.stdout,
                    "error": f"执行错误: {result.error_in_exec}",
                    "variables": {},
                }

            output = captured.stdout
            if result.result is not None:
                output += f"\n{self._format_table_output(result.result)}"

            vars_after = set(self.shell.user_ns.keys())
            important_new_vars = {}
            for var_name in vars_after - vars_before:
                if var_name.startswith("_"):
                    continue
                try:
                    var_value = self.shell.user_ns[var_name]
                    if hasattr(var_value, "shape"):
                        important_new_vars[var_name] = f"{type(var_value).__name__} with shape {var_value.shape}"
                    elif var_name in ["session_output_dir"]:
                        important_new_vars[var_name] = str(var_value)
                except Exception:
                    continue

            return {
                "success": True,
                "output": output,
                "error": "",
                "variables": important_new_vars,
            }
        except Exception as e:
            return {
                "success": False,
                "output": captured.stdout if "captured" in locals() else "",
                "error": f"执行异常: {str(e)}\n{traceback.format_exc()}",
                "variables": {},
            }

    def reset_environment(self):
        """重置执行环境。"""
        self.shell.reset()
        self._setup_common_imports()
        self._setup_chinese_font()
        plt.close("all")

    def set_variable(self, name: str, value: Any):
        """设置执行环境变量。"""
        self.shell.user_ns[name] = value

    def get_environment_info(self) -> str:
        """获取当前执行环境变量信息，用于提示词上下文。"""
        info_parts = []
        important_vars = {}
        ignored = {"In", "Out", "get_ipython", "exit", "quit"}

        for var_name, var_value in self.shell.user_ns.items():
            if var_name.startswith("_") or var_name in ignored:
                continue
            try:
                if hasattr(var_value, "shape"):
                    important_vars[var_name] = f"{type(var_value).__name__} with shape {var_value.shape}"
                elif var_name == "session_output_dir":
                    important_vars[var_name] = str(var_value)
                elif isinstance(var_value, (int, float, str, bool)) and len(str(var_value)) < 100:
                    important_vars[var_name] = f"{type(var_value).__name__}: {var_value}"
            except Exception:
                continue

        if important_vars:
            info_parts.append("当前环境变量:")
            for var_name, var_info in important_vars.items():
                info_parts.append(f"- {var_name}: {var_info}")
        else:
            info_parts.append("当前环境已预导入 pandas、numpy、matplotlib 等常用库。")

        if "session_output_dir" in self.shell.user_ns:
            info_parts.append(f"输出目录: session_output_dir = '{self.shell.user_ns['session_output_dir']}'")

        return "\n".join(info_parts)
