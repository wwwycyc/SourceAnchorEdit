import { useRef, useState } from "react";
import { postJson, resolveApiUrl } from "./lib/api";
import { prepareImageAsset } from "./lib/image";
import { buildMockPromptResponse, buildMockVisionResponse } from "./lib/mock";

const INITIAL_STEP_STATE = {
  source: "idle",
  target: "idle",
  backend: "idle",
};

const INITIAL_PROMPTS = {
  sourcePrompt: "等待识图...",
  translatedInstruction: "等待指令翻译...",
  targetPrompt: "等待 target prompt 生成...",
};

const INITIAL_SUMMARY = {
  mode: "待机",
  durationSeconds: "-",
  jobId: "-",
  methodName: "source_anchor",
};

export default function App() {
  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [preparedImage, setPreparedImage] = useState(null);
  const [instruction, setInstruction] = useState("");
  const [sourceAnchorStart, setSourceAnchorStart] = useState(0);
  const [sharedSteps, setSharedSteps] = useState(5);
  const [useMockFallback, setUseMockFallback] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [runtimeBadge, setRuntimeBadge] = useState("等待运行");
  const [formError, setFormError] = useState("");
  const [stepState, setStepState] = useState(INITIAL_STEP_STATE);
  const [resultImageUrl, setResultImageUrl] = useState("");
  const [prompts, setPrompts] = useState(INITIAL_PROMPTS);
  const [runSummary, setRunSummary] = useState(INITIAL_SUMMARY);
  const [payloadSummary, setPayloadSummary] = useState(null);
  const [logs, setLogs] = useState(() => [createLogEntry("页面已就绪，请上传图片并输入指令。")]);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setFormError("");
    setSelectedFile(file);

    try {
      const nextPreparedImage = await prepareImageAsset(file, 512);
      setPreparedImage(nextPreparedImage);
      appendLog("图片已加载，并统一处理成 512 x 512。");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSelectedFile(null);
      setPreparedImage(null);
      setFormError(`图片读取失败: ${message}`);
      appendLog(`图片读取失败: ${message}`);
    }
  }

  async function handleRun(event) {
    event.preventDefault();
    if (isRunning) {
      return;
    }

    const validationError = validateSubmission({ selectedFile, preparedImage, instruction });
    if (validationError) {
      setFormError(validationError);
      appendLog(validationError);
      return;
    }

    let activePreparedImage = preparedImage;
    if (!activePreparedImage && selectedFile) {
      activePreparedImage = await prepareImageAsset(selectedFile, 512);
      setPreparedImage(activePreparedImage);
    }

    if (!selectedFile || !activePreparedImage) {
      return;
    }

    resetRunView();
    setFormError("");
    setIsRunning(true);
    setPrompts({
      sourcePrompt: "正在生成 source prompt...",
      translatedInstruction: "正在翻译指令...",
      targetPrompt: "正在生成 target prompt...",
    });
    appendLog("开始执行整条链路。");

    try {
      setStepState({ source: "active", target: "idle", backend: "idle" });

      const visionPayload = {
        filename: selectedFile.name,
        imageBase64: activePreparedImage.base64,
        imageMimeType: activePreparedImage.mimeType,
        imageSize: 512,
      };

      const visionResult = await requestWithFallback(
        "/api/vision/source-prompt",
        visionPayload,
        buildMockVisionResponse,
      );

      setRuntimeBadge(visionResult.mode === "mimo" ? "Mimo 识图" : "Mock 回退");
      if (visionResult.warning) {
        appendLog(`识图警告: ${visionResult.warning}`);
      }

      setPrompts((current) => ({
        ...current,
        sourcePrompt: visionResult.sourcePrompt,
      }));
      setStepState({ source: "done", target: "active", backend: "idle" });
      setRunSummary((current) => ({
        ...current,
        mode: visionResult.mode === "mimo" ? "识图完成" : "识图回退",
      }));
      appendLog("source prompt 已生成。");

      const promptPayload = {
        sourcePrompt: visionResult.sourcePrompt,
        userInstruction: instruction.trim(),
      };

      const promptResult = await requestWithFallback(
        "/api/prompt/compose-target",
        promptPayload,
        buildMockPromptResponse,
      );

      setRuntimeBadge(promptResult.mode === "mimo" ? "Mimo Prompt" : "规则回退");
      if (promptResult.warning) {
        appendLog(`Prompt 警告: ${promptResult.warning}`);
      }

      setPrompts({
        sourcePrompt: visionResult.sourcePrompt,
        translatedInstruction: promptResult.translatedInstruction,
        targetPrompt: promptResult.targetPrompt,
      });
      setStepState({ source: "done", target: "done", backend: "active" });
      setRunSummary((current) => ({
        ...current,
        mode: promptResult.mode === "mimo" ? "Prompt 完成" : "Prompt 回退",
      }));
      appendLog("target prompt 已生成。");

      const supportPayload = buildSupportPayload({
        preparedImage: activePreparedImage,
        filename: selectedFile.name,
        instruction,
        sourcePrompt: visionResult.sourcePrompt,
        translatedInstruction: promptResult.translatedInstruction,
        targetPrompt: promptResult.targetPrompt,
        sourceAnchorStart,
        sharedSteps,
      });

      const backendResult = await postJson("/api/source-anchor/run", supportPayload);
      const nextResultUrl = resolveApiUrl(backendResult.editedImageUrl);
      setResultImageUrl(withCacheBreaker(nextResultUrl));
      setRuntimeBadge("已完成");
      setStepState({ source: "done", target: "done", backend: "done" });
      setRunSummary({
        mode: "执行完成",
        durationSeconds: backendResult.durationSeconds ?? "-",
        jobId: backendResult.jobId ?? "-",
        methodName: backendResult.methodName ?? "source_anchor",
      });
      setPayloadSummary(backendResult.payloadSummary ?? null);
      appendLog("编辑后的图片已生成。");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setRuntimeBadge("运行失败");
      setFormError(`运行失败: ${message}`);
      setStepState((current) => ({
        source: current.source === "active" ? "idle" : current.source,
        target: current.target === "active" ? "idle" : current.target,
        backend: current.backend === "active" ? "idle" : current.backend,
      }));
      setRunSummary((current) => ({
        ...current,
        mode: "失败",
      }));
      appendLog(`运行失败: ${message}`);
    } finally {
      setIsRunning(false);
    }
  }

  function handleClearLog() {
    setLogs([]);
  }

  function handleToggleFallback() {
    const next = !useMockFallback;
    setUseMockFallback(next);
    appendLog(`规则回退已${next ? "开启" : "关闭"}。`);
    if (!isRunning) {
      setRuntimeBadge("等待运行");
    }
  }

  async function requestWithFallback(endpoint, payload, fallbackFactory) {
    try {
      return await postJson(endpoint, payload);
    } catch (error) {
      if (!useMockFallback) {
        throw error;
      }
      appendLog(`接口 ${endpoint} 调用失败，已回退到浏览器模拟。`);
      return fallbackFactory(payload);
    }
  }

  function resetRunView() {
    setRuntimeBadge("运行中");
    setResultImageUrl("");
    setPayloadSummary(null);
    setStepState({ ...INITIAL_STEP_STATE });
    setRunSummary({
      mode: "准备执行",
      durationSeconds: "-",
      jobId: "-",
      methodName: "source_anchor",
    });
  }

  function appendLog(message) {
    setLogs((current) => [createLogEntry(message), ...current].slice(0, 80));
  }

  return (
    <div className="app">
      <div className="aurora aurora-a" />
      <div className="aurora aurora-b" />
      <div className="shell">
        <header className="hero">
          <div className="hero-copy">
            <h1>Source Anchor Studio</h1>
          </div>
          <div className="hero-tags">
            <Tag label="Method" value="source_anchor" />
            <Tag label="Canvas" value="512 x 512" />
            <Tag label="Fallback" value={useMockFallback ? "on" : "off"} />
          </div>
        </header>

        <main className="studio-grid">
          <section className="left-stage">
            <Panel className="preview-panel" kicker="Input" title="预览图" badge={runtimeBadge}>
              <ImageStage
                imageUrl={preparedImage?.previewUrl}
                alt="上传预览图"
                mark="INPUT"
                headline="预览图"
                description="上传后的原图会显示在这里"
                onClick={() => fileInputRef.current?.click()}
              />
            </Panel>

            <Panel className="result-panel" kicker="Output" title="结果图">
              <ImageStage
                imageUrl={resultImageUrl}
                alt="编辑结果图"
                mark="OUTPUT"
                headline="结果图"
                description="生成完成后，这里会显示 source_anchor 输出"
              />
            </Panel>

            <Panel
              className="status-panel"
              kicker="Workflow"
              title="运行状态"
              action={
                <button className="ghost-button" type="button" onClick={handleClearLog}>
                  清空
                </button>
              }
            >
              <div className="step-grid">
                <StepCard index="01" title="识图" text="生成英文 source prompt" state={stepState.source} />
                <StepCard
                  index="02"
                  title="Prompt"
                  text="翻译指令并拼出 target prompt"
                  state={stepState.target}
                />
                <StepCard
                  index="03"
                  title="编辑"
                  text="调用 source_anchor 返回最终结果"
                  state={stepState.backend}
                />
              </div>

              <div className="status-strip">
                <StatusToken label="模式" value={runSummary.mode} />
                <StatusToken label="耗时" value={String(runSummary.durationSeconds)} />
                <StatusToken label="任务" value={runSummary.jobId} />
                <StatusToken label="方法" value={runSummary.methodName} />
              </div>

              {payloadSummary ? (
                <div className="payload-card">
                  <h3>本次执行参数</h3>
                  <div className="payload-grid">
                    <MetricCard label="Steps" value={`${payloadSummary.numEditSteps}`} compact />
                    <MetricCard label="Anchor Start" value={`${payloadSummary.sourceAnchorStart}`} compact />
                    <MetricCard label="Batch" value={`${payloadSummary.batchSize}`} compact />
                    <MetricCard label="DType" value={`${payloadSummary.dtype}`} compact />
                  </div>
                </div>
              ) : null}

              <div className="log-panel" aria-live="polite">
                {logs.length === 0 ? (
                  <p className="log-empty">日志已清空。</p>
                ) : (
                  logs.map((entry) => (
                    <div className="log-entry" key={entry.id}>
                      <time>{entry.time}</time>
                      <p>{entry.message}</p>
                    </div>
                  ))
                )}
              </div>
            </Panel>
          </section>

          <aside className="right-rail">
            <Panel className="settings-panel" kicker="Controls" title="编辑配置">
              <div className="settings-stack">
                <label className="control-card">
                  <div className="control-head">
                    <span>source_anchor_start</span>
                    <strong>{sourceAnchorStart.toFixed(2)}</strong>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={sourceAnchorStart}
                    disabled={isRunning}
                    onChange={(event) => setSourceAnchorStart(Number(event.target.value))}
                  />
                </label>

                <label className="control-card">
                  <div className="control-head">
                    <span>统一步数</span>
                    <strong>{sharedSteps}</strong>
                  </div>
                  <input
                    type="range"
                    min="4"
                    max="30"
                    step="1"
                    value={sharedSteps}
                    disabled={isRunning}
                    onChange={(event) => setSharedSteps(Number(event.target.value))}
                  />
                </label>

                <div className="toggle-card">
                  <div>
                    <span className="card-label">回退策略</span>
                    <strong>接口失败时启用浏览器规则回退</strong>
                  </div>
                  <button
                    className={`switch-button${useMockFallback ? " is-on" : ""}`}
                    type="button"
                    onClick={handleToggleFallback}
                    disabled={isRunning}
                  >
                    {useMockFallback ? "开启" : "关闭"}
                  </button>
                </div>
              </div>
            </Panel>

            <Panel className="prompt-panel" kicker="Prompt" title="文本结果">
              <PromptBlock title="source prompt" content={prompts.sourcePrompt} />
              <PromptBlock title="translated instruction" content={prompts.translatedInstruction} />
              <PromptBlock title="target prompt" content={prompts.targetPrompt} />
            </Panel>
          </aside>
        </main>

        <form className="composer" onSubmit={handleRun}>
          <div className="composer-shell">
            <label className="upload-chip">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                disabled={isRunning}
                onChange={handleFileChange}
              />
              <span className="upload-icon">+</span>
            </label>

            <div className="composer-input-wrap">
              <textarea
                rows="1"
                value={instruction}
                disabled={isRunning}
                onChange={(event) => setInstruction(event.target.value)}
                placeholder="输入编辑指令，例如：把猫改成狗，其余不变。"
              />
            </div>

            <button
              className="run-button"
              type="submit"
              disabled={isRunning}
              aria-label={isRunning ? "正在生成" : "开始生成"}
              title={isRunning ? "正在生成" : "开始生成"}
            >
              <span className="run-arrow" aria-hidden="true">
                ↑
              </span>
            </button>
          </div>

          <p className="form-error">{formError}</p>
        </form>
      </div>
    </div>
  );
}

function Panel({ className = "", kicker, title, badge, action, children }) {
  return (
    <section className={`panel ${className}`.trim()}>
      <div className="panel-header">
        <div>
          <p className="panel-kicker">{kicker}</p>
          <h2>{title}</h2>
        </div>
        {action ? action : badge ? <span className="panel-badge">{badge}</span> : null}
      </div>
      {children}
    </section>
  );
}

function ImageStage({ imageUrl, alt, mark, headline, description, onClick }) {
  const Component = onClick ? "button" : "div";
  return (
    <Component
      className={`image-stage${imageUrl ? " has-image" : ""}`}
      {...(onClick ? { type: "button", onClick } : {})}
    >
      {imageUrl ? <img src={imageUrl} alt={alt} /> : null}
      {!imageUrl ? (
        <div className="stage-placeholder">
          <span className="stage-mark">{mark}</span>
          <strong>{headline}</strong>
          <p>{description}</p>
        </div>
      ) : null}
    </Component>
  );
}

function MetricCard({ label, value, compact = false }) {
  return (
    <div className={`metric-card${compact ? " compact" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StepCard({ index, title, text, state }) {
  return (
    <article className={`step-card ${state}`}>
      <span className="step-index">{index}</span>
      <strong>{title}</strong>
      <p>{text}</p>
    </article>
  );
}

function StatusToken({ label, value }) {
  return (
    <div className="status-token">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PromptBlock({ title, content }) {
  return (
    <div className="prompt-block">
      <span>{title}</span>
      <pre>{content}</pre>
    </div>
  );
}

function Tag({ label, value }) {
  return (
    <div className="hero-tag">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function createLogEntry(message) {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    time: new Date().toLocaleTimeString("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }),
    message,
  };
}

function validateSubmission({ selectedFile, preparedImage, instruction }) {
  if (!selectedFile) {
    return "请先上传图片。";
  }
  if (!preparedImage) {
    return "图片仍在处理中，请稍后再试。";
  }
  if (!instruction.trim()) {
    return "请输入编辑指令。";
  }
  return "";
}

function buildSupportPayload({
  preparedImage,
  filename,
  instruction,
  sourcePrompt,
  translatedInstruction,
  targetPrompt,
  sourceAnchorStart,
  sharedSteps,
}) {
  return {
    workflow: "source_anchor_no_target_hints",
    variant: "source_anchor",
    noTargetHints: true,
    imageBase64: preparedImage.base64,
    imageMimeType: preparedImage.mimeType,
    imageSize: 512,
    filename,
    userInstruction: instruction.trim(),
    translatedInstruction,
    sourcePrompt,
    targetPrompt,
    targetTokenHints: [],
    sourceAnchorStart,
    numSteps: sharedSteps,
    supportLineArgs: {
      variant: "source_anchor",
      source_anchor_start: sourceAnchorStart,
      num_steps: sharedSteps,
      image_size: 512,
      target_token_hints: [],
    },
  };
}

function withCacheBreaker(url) {
  if (!url) {
    return "";
  }
  return `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
}
