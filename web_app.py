# -*- coding: utf-8 -*-
"""机器学习模型迭代智能体 Web 服务。"""

import cgi
import json
import os
import threading
import time
import uuid
from copy import deepcopy
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from config.llm_config import LLMConfig
from ml_model_agent import MLModelAgent


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "uploads"
TASKS = {}
TASKS_LOCK = threading.Lock()


def _json_safe(value):
    """将任务状态转换为 JSON 可序列化对象。"""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)


def _public_task(task):
    """返回给前端的任务副本，避免暴露 API Key。"""
    public = deepcopy(task)
    request = public.get("request", {})
    if request.get("api_key"):
        request["api_key"] = "********"
    return _json_safe(public)


def _now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _create_task(payload):
    task_id = uuid.uuid4().hex
    task = {
        "id": task_id,
        "status": "queued",
        "created_at": _now_ts(),
        "updated_at": _now_ts(),
        "request": payload,
        "events": [],
        "versions": [],
        "messages": [
            {
                "role": "user",
                "content": payload.get("user_input", ""),
                "time": _now_ts(),
            }
        ],
        "final_report": "",
        "report_file_path": "",
        "session_output_dir": "",
        "error": "",
        "cancel_requested": False,
    }
    with TASKS_LOCK:
        TASKS[task_id] = task
    return task


def _append_event(task_id, event):
    with TASKS_LOCK:
        task = TASKS[task_id]
        event = deepcopy(event)
        event["time"] = _now_ts()
        task["events"].append(event)
        task["updated_at"] = _now_ts()

        if event.get("type") == "started":
            task["status"] = "running"
            task["session_output_dir"] = event.get("session_output_dir", "")
            task["messages"].append({
                "role": "assistant",
                "content": "任务已启动，我会先探索数据，再训练基线模型。",
                "time": event["time"],
            })
        elif event.get("type") == "round_started":
            task["status"] = "running"
        elif event.get("type") == "round_finished":
            _append_version_from_event(task, event)
        elif event.get("type") == "round_error":
            task["status"] = "running"
            task["error"] = event.get("message", "")
            task["messages"].append({
                "role": "assistant",
                "content": event.get("message", "本轮执行出现错误，正在尝试修复。"),
                "time": event["time"],
            })
        elif event.get("type") == "finished":
            task["status"] = "stopped" if task.get("cancel_requested") else "completed"
            task["final_report"] = event.get("final_report", "")
            task["report_file_path"] = event.get("report_file_path", "")
            task["messages"].append({
                "role": "assistant",
                "content": "模型迭代完成，最终报告已生成。",
                "time": event["time"],
            })
        elif event.get("type") == "stopped":
            task["status"] = "stopped"
            task["final_report"] = event.get("final_report", "任务已由用户停止。")
            task["messages"].append({
                "role": "assistant",
                "content": "任务已停止。",
                "time": event["time"],
            })


def _append_version_from_event(task, event):
    version = {
        "version": len(task["versions"]) + 1,
        "round": event.get("round"),
        "action": event.get("action"),
        "success": event.get("success"),
        "code": event.get("code", ""),
        "output": event.get("output", ""),
        "error": event.get("error", ""),
        "assistant_response": event.get("assistant_response", ""),
        "variables": event.get("variables", {}),
        "feedback": event.get("feedback", ""),
        "time": event["time"],
    }
    task["versions"].append(version)
    if version["action"] == "generate_code":
        status = "成功" if version["success"] else "失败"
        content = f"第 {version['version']} 个训练版本执行{status}，中间栏可以查看输出和代码。"
    elif version["action"] == "invalid_response":
        content = f"第 {version['version']} 轮没有生成可执行代码，请查看该版本的错误或原始回复。"
    else:
        content = f"第 {version['version']} 轮已记录。"
    task["messages"].append({
        "role": "assistant",
        "content": content,
        "time": event["time"],
    })
    if event.get("final_report"):
        task["final_report"] = event.get("final_report", "")


def _run_task(task_id):
    with TASKS_LOCK:
        payload = deepcopy(TASKS[task_id]["request"])

    try:
        max_tokens = min(int(payload.get("max_tokens", 16384)), 32768)
        llm_config = LLMConfig(
            api_key=payload.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
            base_url=payload.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=payload.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview"),
            temperature=float(payload.get("temperature", 0.1)),
            max_tokens=max_tokens,
        )
        llm_config.validate()

        agent = MLModelAgent(
            llm_config=llm_config,
            output_dir=payload.get("output_dir") or "outputs",
            max_rounds=int(payload.get("max_rounds", 12)),
        )
        result = agent.build_model(
            user_input=payload.get("user_input", ""),
            files=payload.get("files") or [],
            target=payload.get("target") or None,
            metric=payload.get("metric") or None,
            expected_performance=payload.get("expected_performance") or None,
            progress_callback=lambda event: _append_event(task_id, event),
            should_stop=lambda: _should_stop(task_id),
        )

        with TASKS_LOCK:
            task = TASKS[task_id]
            if task.get("cancel_requested"):
                task["status"] = "stopped"
            elif task["status"] != "stopped":
                task["status"] = "completed"
            task["final_report"] = result.get("final_report", "")
            task["report_file_path"] = result.get("report_file_path", "")
            task["session_output_dir"] = result.get("session_output_dir", "")
            task["request"]["api_key"] = ""
            task["updated_at"] = _now_ts()

    except Exception as e:
        with TASKS_LOCK:
            task = TASKS[task_id]
            task["status"] = "failed"
            task["error"] = str(e)
            task["request"]["api_key"] = ""
            task["updated_at"] = _now_ts()
            task["messages"].append({
                "role": "assistant",
                "content": f"任务失败: {e}",
                "time": _now_ts(),
            })


def _should_stop(task_id):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        return bool(task and task.get("cancel_requested"))


def _request_stop(task_id):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return None
        if task["status"] in {"completed", "failed", "stopped"}:
            return task
        task["cancel_requested"] = True
        task["status"] = "stopping"
        task["updated_at"] = _now_ts()
        task["messages"].append({
            "role": "assistant",
            "content": "已收到停止请求，会在当前调用结束后停止。",
            "time": _now_ts(),
        })
        return task


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/tasks":
            self._send_json(self._list_tasks())
            return

        if parsed.path.startswith("/api/tasks/"):
            task_id = parsed.path.rsplit("/", 1)[-1]
            task = self._get_task(task_id)
            if not task:
                self._send_json({"error": "任务不存在"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(task)
            return

        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/tasks":
            payload = self._read_json_body()
            task = _create_task(payload)
            worker = threading.Thread(target=_run_task, args=(task["id"],), daemon=True)
            worker.start()
            self._send_json(_public_task(task), HTTPStatus.CREATED)
            return

        if parsed.path == "/api/upload":
            uploaded = self._handle_upload()
            self._send_json(uploaded, HTTPStatus.CREATED)
            return

        if parsed.path.startswith("/api/tasks/") and parsed.path.endswith("/stop"):
            task_id = parsed.path.split("/")[-2]
            task = _request_stop(task_id)
            if not task:
                self._send_json({"error": "任务不存在"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(_public_task(task))
            return

        self._send_json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _handle_upload(self):
        ctype, pdict = cgi.parse_header(self.headers.get("Content-Type", ""))
        if ctype != "multipart/form-data":
            raise ValueError("请使用 multipart/form-data 上传文件")

        pdict["boundary"] = bytes(pdict["boundary"], "utf-8")
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )
        file_item = form["file"]
        original_name = Path(file_item.filename).name
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = f"{uuid.uuid4().hex}_{original_name}"
        file_path = UPLOAD_DIR / safe_name
        with file_path.open("wb") as f:
            f.write(file_item.file.read())
        return {
            "filename": original_name,
            "path": str(file_path),
        }

    def _list_tasks(self):
        with TASKS_LOCK:
            return [
                {
                    "id": task["id"],
                    "status": task["status"],
                    "created_at": task["created_at"],
                    "updated_at": task["updated_at"],
                    "user_input": task["request"].get("user_input", ""),
                }
                for task in TASKS.values()
            ]

    def _get_task(self, task_id):
        with TASKS_LOCK:
            task = TASKS.get(task_id)
            if not task:
                return None
            return _public_task(task)

    def _send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(_json_safe(data), ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host="127.0.0.1", port=7860):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Web 页面运行在 http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    run(host=host, port=port)
