const state = {
  taskId: null,
  uploadedPath: "",
  pollTimer: null,
  selectedTabs: new Map(),
  versionsSignature: "",
  messagesSignature: "",
};

const CONFIG_STORAGE_KEY = "mlWorkbench.modelConfig.v1";
const DEFAULT_CONFIG = {
  apiKey: "",
  baseUrl: "https://ark.cn-beijing.volces.com/api/v3",
  modelName: "deepseek-v3-250324",
  maxRounds: "12",
  temperature: "0.1",
  maxTokens: "16384",
};

const els = {
  form: document.querySelector("#taskForm"),
  chatLog: document.querySelector("#chatLog"),
  versionList: document.querySelector("#versionList"),
  versionCount: document.querySelector("#versionCount"),
  taskIdLabel: document.querySelector("#taskIdLabel"),
  runState: document.querySelector("#runState"),
  startButton: document.querySelector("#startButton"),
  fileUpload: document.querySelector("#fileUpload"),
  uploadLabel: document.querySelector("#uploadLabel"),
  finalReport: document.querySelector("#finalReport"),
  reportPath: document.querySelector("#reportPath"),
  stopButton: document.querySelector("#stopButton"),
  saveConfigButton: document.querySelector("#saveConfigButton"),
  clearConfigButton: document.querySelector("#clearConfigButton"),
  configStatus: document.querySelector("#configStatus"),
};

function value(id) {
  const el = document.querySelector(`#${id}`);
  return el ? el.value.trim() : "";
}

function setRunState(status, label) {
  els.runState.classList.remove("running", "completed", "failed");
  if (status) els.runState.classList.add(status);
  els.runState.querySelector("span:last-child").textContent = label;
}

function escapeHtml(text = "") {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function splitFiles(raw) {
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildPayload() {
  const files = splitFiles(value("filePaths"));
  if (state.uploadedPath && !files.includes(state.uploadedPath)) {
    files.push(state.uploadedPath);
  }

  return {
    user_input: value("userInput"),
    target: "",
    metric: "",
    expected_performance: "",
    files,
    api_key: value("apiKey"),
    base_url: value("baseUrl"),
    model: value("modelName"),
    max_rounds: Number(value("maxRounds") || 12),
    temperature: Number(value("temperature") || 0.1),
    max_tokens: Number(value("maxTokens") || 16384),
  };
}

function setInputValue(id, nextValue) {
  const el = document.querySelector(`#${id}`);
  if (el) el.value = nextValue;
}

function getConfigFromForm() {
  return {
    apiKey: value("apiKey"),
    baseUrl: value("baseUrl"),
    modelName: value("modelName"),
    maxRounds: value("maxRounds") || DEFAULT_CONFIG.maxRounds,
    temperature: value("temperature") || DEFAULT_CONFIG.temperature,
    maxTokens: value("maxTokens") || DEFAULT_CONFIG.maxTokens,
  };
}

function applyConfig(config) {
  setInputValue("apiKey", config.apiKey || "");
  setInputValue("baseUrl", config.baseUrl || DEFAULT_CONFIG.baseUrl);
  setInputValue("modelName", config.modelName || DEFAULT_CONFIG.modelName);
  setInputValue("maxRounds", config.maxRounds || DEFAULT_CONFIG.maxRounds);
  setInputValue("temperature", config.temperature || DEFAULT_CONFIG.temperature);
  setInputValue("maxTokens", config.maxTokens || DEFAULT_CONFIG.maxTokens);
}

function setConfigStatus(text) {
  if (!els.configStatus) return;
  els.configStatus.textContent = text;
}

function loadSavedConfig() {
  const raw = localStorage.getItem(CONFIG_STORAGE_KEY);
  if (!raw) {
    applyConfig(DEFAULT_CONFIG);
    setConfigStatus("未保存");
    return;
  }

  try {
    const saved = JSON.parse(raw);
    applyConfig({ ...DEFAULT_CONFIG, ...saved });
    setConfigStatus(`已载入保存配置 · ${saved.savedAt || "本机"}`);
  } catch {
    applyConfig(DEFAULT_CONFIG);
    setConfigStatus("配置读取失败，已恢复默认");
  }
}

function saveConfig() {
  const config = {
    ...getConfigFromForm(),
    savedAt: new Date().toLocaleString(),
  };
  localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(config));
  setConfigStatus(`已保存 · ${config.savedAt}`);
}

function clearConfig() {
  localStorage.removeItem(CONFIG_STORAGE_KEY);
  applyConfig(DEFAULT_CONFIG);
  setConfigStatus("已清除，恢复默认配置");
}

function renderMessages(messages = []) {
  const base = messages.length
    ? messages
    : [
        {
          role: "assistant",
          content: "告诉我训练目标并上传数据文件就行。目标列、评价指标、验证方式和迭代策略如果没有写明，我会先读数据再自行判断。",
        },
      ];
  const nextSignature = base
    .map((message) => `${message.role}:${message.time || ""}:${message.content}`)
    .join("|");
  if (nextSignature === state.messagesSignature) {
    return;
  }
  state.messagesSignature = nextSignature;

  els.chatLog.innerHTML = base
    .map(
      (message) => `
        <article class="message ${message.role === "user" ? "user" : "assistant"}">
          <div class="message-meta">${message.role === "user" ? "You" : "Agent"}${message.time ? ` · ${escapeHtml(message.time)}` : ""}</div>
          <p>${escapeHtml(message.content)}</p>
        </article>
      `,
    )
    .join("");
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function chooseTab(version) {
  return state.selectedTabs.get(version.version) || "output";
}

function buildVersionsSignature(versions = []) {
  return versions
    .map((version) => {
      const tab = state.selectedTabs.get(version.version) || "";
      return [
        version.version,
        version.round,
        version.action,
        version.success,
        version.time,
        tab,
        (version.output || "").length,
        (version.code || "").length,
        (version.feedback || "").length,
        (version.error || "").length,
        (version.assistant_response || "").length,
      ].join(":");
    })
    .join("|");
}

function tabContent(version, tab) {
  if (tab === "code") return version.code || "暂无代码";
  if (tab === "error") {
    if (version.error) return version.error;
    if (version.action === "invalid_response") return "LLM 没有返回符合 YAML 格式的可执行代码。请切到“原始回复”查看模型实际返回内容；如果原始回复为空，请检查模型名称、Base URL、API Key 或服务商 max_tokens 限制。";
    return "暂无错误";
  }
  if (tab === "feedback") return version.feedback || "暂无反馈";
  if (tab === "raw") return version.assistant_response || "暂无原始回复";
  return version.output || "暂无输出";
}

function renderVersions(versions = []) {
  els.versionCount.textContent = `${versions.length} 个版本`;
  const nextSignature = buildVersionsSignature(versions);
  if (nextSignature === state.versionsSignature) {
    return;
  }
  const previousScrollTop = els.versionList.scrollTop;
  state.versionsSignature = nextSignature;

  if (!versions.length) {
    els.versionList.innerHTML = `
      <div class="empty-state">
        <p>训练开始后，每一轮代码执行结果都会出现在这里。</p>
      </div>
    `;
    els.versionList.scrollTop = previousScrollTop;
    return;
  }

  els.versionList.innerHTML = versions
    .map((version) => {
      const tab = version.action === "invalid_response" && !state.selectedTabs.has(version.version)
        ? "error"
        : chooseTab(version);
      const ok = version.success === true;
      const failed = version.success === false;
      const badgeClass = ok ? "success" : failed ? "failed" : "";
      const badgeText = ok ? "成功" : failed ? "失败" : "记录";
      return `
        <article class="version-card" data-version="${version.version}">
          <div class="version-summary">
            <div class="version-number">V${version.version}</div>
            <div>
              <p class="version-title">第 ${version.round || version.version} 轮 · ${escapeHtml(version.action || "unknown")}</p>
              <p class="version-subtitle">${escapeHtml(version.time || "")}</p>
            </div>
            <span class="badge ${badgeClass}">${badgeText}</span>
          </div>
          <div class="version-body">
            <div class="tabs">
              ${["output", "code", "feedback", "error", "raw"]
                .map(
                  (item) => `
                    <button class="tab-button ${tab === item ? "active" : ""}" type="button" data-tab="${item}" data-version="${version.version}">
                      ${item === "output" ? "输出" : item === "code" ? "代码" : item === "feedback" ? "反馈" : item === "error" ? "错误" : "原始回复"}
                    </button>
                  `,
                )
                .join("")}
            </div>
            <pre>${escapeHtml(tabContent(version, tab))}</pre>
          </div>
        </article>
      `;
    })
    .join("");
  els.versionList.scrollTop = previousScrollTop;
}

function renderTask(task) {
  state.taskId = task.id;
  els.taskIdLabel.textContent = task.id ? `任务 ${task.id.slice(0, 8)}` : "未创建任务";

  const statusMap = {
    queued: ["running", "排队中"],
    running: ["running", "训练中"],
    stopping: ["running", "停止中"],
    stopped: ["failed", "已停止"],
    completed: ["completed", "已完成"],
    failed: ["failed", "失败"],
  };
  const [statusClass, statusLabel] = statusMap[task.status] || ["", "待启动"];
  setRunState(statusClass, statusLabel);

  renderMessages(task.messages);
  renderVersions(task.versions);

  if (task.final_report) {
    els.finalReport.textContent = task.final_report;
  }
  if (task.report_file_path) {
    els.reportPath.textContent = task.report_file_path;
  }

  const isActive = task.status === "queued" || task.status === "running" || task.status === "stopping";
  els.startButton.disabled = isActive;
  els.startButton.querySelector("span").textContent = els.startButton.disabled ? "训练进行中" : "启动训练";
  els.stopButton.disabled = !state.taskId || !isActive || task.status === "stopping";
  els.stopButton.querySelector("span").textContent = task.status === "stopping" ? "停止中" : "停止任务";

  if (task.status === "completed" || task.status === "failed" || task.status === "stopped") {
    stopPolling();
  }
}

async function fetchTask() {
  if (!state.taskId) return;
  const res = await fetch(`/api/tasks/${state.taskId}`);
  const task = await res.json();
  renderTask(task);
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(fetchTask, 1800);
}

function stopPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  els.uploadLabel.textContent = "上传中...";
  const res = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    els.uploadLabel.textContent = "上传失败";
    throw new Error("上传失败");
  }
  const data = await res.json();
  state.uploadedPath = data.path;
  els.uploadLabel.textContent = `${data.filename}`;
}

els.fileUpload.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    await uploadFile(file);
  } catch (error) {
    console.error(error);
  }
});

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = buildPayload();
  if (!payload.user_input) return;
  if (!payload.files.length) {
    renderMessages([
      {
        role: "assistant",
        content: "请先上传一个 CSV/Excel 文件，或者在“使用本地文件路径”里填写数据文件路径。",
      },
    ]);
    return;
  }

  setRunState("running", "提交中");
  els.startButton.disabled = true;
  els.startButton.querySelector("span").textContent = "提交中";
  els.finalReport.textContent = "暂无报告";
  els.reportPath.textContent = "等待生成";
  state.selectedTabs.clear();
  state.versionsSignature = "";
  state.messagesSignature = "";

  const res = await fetch("/api/tasks", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const task = await res.json();
  renderTask(task);
  startPolling();
});

els.stopButton.addEventListener("click", async () => {
  if (!state.taskId) return;
  els.stopButton.disabled = true;
  els.stopButton.querySelector("span").textContent = "停止中";
  const res = await fetch(`/api/tasks/${state.taskId}/stop`, {
    method: "POST",
  });
  const task = await res.json();
  renderTask(task);
  startPolling();
});

els.versionList.addEventListener("click", (event) => {
  const button = event.target.closest(".tab-button");
  if (!button) return;
  state.selectedTabs.set(Number(button.dataset.version), button.dataset.tab);
  state.versionsSignature = "";
  fetchTask();
});

els.saveConfigButton.addEventListener("click", saveConfig);
els.clearConfigButton.addEventListener("click", clearConfig);

loadSavedConfig();
