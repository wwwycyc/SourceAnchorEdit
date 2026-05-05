(function () {
  const state = {
    selectedFile: null,
    resizedImage: null,
    useMockFallback: true,
    isRunning: false,
  };

  const refs = {
    composerForm: document.getElementById("composerForm"),
    imageInput: document.getElementById("imageInput"),
    instructionInput: document.getElementById("instructionInput"),
    runButton: document.getElementById("runButton"),
    formError: document.getElementById("formError"),
    previewImage: document.getElementById("previewImage"),
    previewPlaceholder: document.getElementById("previewPlaceholder"),
    previewStage: document.getElementById("previewStage"),
    fileNameValue: document.getElementById("fileNameValue"),
    originalSizeValue: document.getElementById("originalSizeValue"),
    processedSizeValue: document.getElementById("processedSizeValue"),
    sourcePromptOutput: document.getElementById("sourcePromptOutput"),
    translatedInstructionOutput: document.getElementById("translatedInstructionOutput"),
    targetPromptOutput: document.getElementById("targetPromptOutput"),
    resultStage: document.getElementById("resultStage"),
    resultImage: document.getElementById("resultImage"),
    resultPlaceholder: document.getElementById("resultPlaceholder"),
    openEditedLink: document.getElementById("openEditedLink"),
    sourceAnchorStartInput: document.getElementById("sourceAnchorStartInput"),
    sharedStepsInput: document.getElementById("sharedStepsInput"),
    anchorValueDisplay: document.getElementById("anchorValueDisplay"),
    anchorValueMirror: document.getElementById("anchorValueMirror"),
    stepsValueDisplay: document.getElementById("stepsValueDisplay"),
    activityLog: document.getElementById("activityLog"),
    clearLogButton: document.getElementById("clearLogButton"),
    mockToggleButton: document.getElementById("mockToggleButton"),
    apiModeBadge: document.getElementById("apiModeBadge"),
    stepSource: document.getElementById("step-source"),
    stepTarget: document.getElementById("step-target"),
    stepBackend: document.getElementById("step-backend"),
  };

  const STEP_ORDER = ["source", "target", "backend"];

  function formatNow() {
    return new Date().toLocaleTimeString("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }

  function log(message) {
    const row = document.createElement("div");
    row.className = "log-entry";
    const time = document.createElement("time");
    time.textContent = formatNow();
    const text = document.createElement("p");
    text.textContent = message;
    row.appendChild(time);
    row.appendChild(text);
    refs.activityLog.prepend(row);
  }

  function setError(message) {
    refs.formError.textContent = message || "";
  }

  function setBusy(isBusy) {
    state.isRunning = isBusy;
    refs.runButton.disabled = isBusy;
    refs.instructionInput.disabled = isBusy;
    refs.imageInput.disabled = isBusy;
    refs.sourceAnchorStartInput.disabled = isBusy;
    refs.sharedStepsInput.disabled = isBusy;
    refs.mockToggleButton.disabled = isBusy;
  }

  function updateAnchorLabel() {
    const value = Number(refs.sourceAnchorStartInput.value).toFixed(2);
    refs.anchorValueDisplay.textContent = value;
    refs.anchorValueMirror.textContent = value;
  }

  function updateStepsLabel() {
    refs.stepsValueDisplay.textContent = String(Number(refs.sharedStepsInput.value));
  }

  function updateApiModeBadge() {
    refs.mockToggleButton.textContent = state.useMockFallback ? "开启" : "关闭";
    refs.mockToggleButton.classList.toggle("active", state.useMockFallback);
    refs.apiModeBadge.textContent = "等待运行";
  }

  function setRuntimeApiBadge(label) {
    refs.apiModeBadge.textContent = label;
  }

  function setStepState(step, status) {
    const card =
      step === "source" ? refs.stepSource : step === "target" ? refs.stepTarget : refs.stepBackend;
    card.classList.remove("is-active", "is-done");
    if (status === "active") {
      card.classList.add("is-active");
    }
    if (status === "done") {
      card.classList.add("is-done");
    }
  }

  function resetStepState() {
    STEP_ORDER.forEach((step) => setStepState(step, "idle"));
  }

  function resetResultView() {
    refs.resultImage.hidden = true;
    refs.resultImage.removeAttribute("src");
    refs.resultPlaceholder.hidden = false;
    refs.resultStage.classList.add("empty");
    refs.openEditedLink.href = "#";
    refs.openEditedLink.classList.add("disabled");
  }

  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  async function readImageDimensions(file) {
    const dataUrl = await fileToDataUrl(file);
    const image = new Image();
    image.src = dataUrl;
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
    });
    file.__imageWidth = image.naturalWidth;
    file.__imageHeight = image.naturalHeight;
    return dataUrl;
  }

  async function resizeImageToSquare(file, size) {
    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#f3f7f1";
    ctx.fillRect(0, 0, size, size);

    const scale = Math.min(size / bitmap.width, size / bitmap.height);
    const drawWidth = Math.round(bitmap.width * scale);
    const drawHeight = Math.round(bitmap.height * scale);
    const offsetX = Math.round((size - drawWidth) / 2);
    const offsetY = Math.round((size - drawHeight) / 2);
    ctx.drawImage(bitmap, offsetX, offsetY, drawWidth, drawHeight);
    bitmap.close();

    const dataUrl = canvas.toDataURL("image/png");
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    return {
      blob,
      dataUrl,
      base64: dataUrl.split(",")[1],
      originalWidth: file.__imageWidth || null,
      originalHeight: file.__imageHeight || null,
      resizedWidth: size,
      resizedHeight: size,
    };
  }

  async function prepareImage(file) {
    const previewUrl = await readImageDimensions(file);
    const resized = await resizeImageToSquare(file, 512);
    state.resizedImage = resized;
    refs.previewImage.src = resized.dataUrl || previewUrl;
    refs.previewImage.hidden = false;
    refs.previewPlaceholder.hidden = true;
    refs.previewStage.classList.remove("empty");
    refs.fileNameValue.textContent = file.name;
    refs.originalSizeValue.textContent = file.__imageWidth + " x " + file.__imageHeight;
    refs.processedSizeValue.textContent = resized.resizedWidth + " x " + resized.resizedHeight;
    log("图片已加载，并统一处理成 512 x 512。");
  }

  function validateSubmission() {
    if (!state.selectedFile) {
      return "请先上传图片。";
    }
    if (!state.resizedImage) {
      return "图片仍在处理中，请稍后再试。";
    }
    if (!refs.instructionInput.value.trim()) {
      return "请输入编辑指令。";
    }
    return "";
  }

  function normalizeFilenameSubject(filename) {
    const stem = filename.replace(/\.[^.]+$/, "");
    const cleaned = stem.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
    return cleaned || "uploaded subject";
  }

  function mockVisionResponse(payload) {
    const subject = normalizeFilenameSubject(payload.filename || "uploaded subject");
    return {
      ok: true,
      mode: "mock",
      sourcePrompt:
        "a realistic photo of " +
        subject +
        ", clean composition, natural lighting, detailed texture, high fidelity",
    };
  }

  function mockPromptResponse(payload) {
    const translatedInstruction = /[\u3400-\u9fff]/.test(payload.userInstruction)
      ? "apply the requested edit and keep the rest unchanged"
      : payload.userInstruction.trim();
    return {
      ok: true,
      mode: "mock",
      translatedInstruction,
      targetPrompt:
        payload.sourcePrompt +
        ". " +
        translatedInstruction.charAt(0).toUpperCase() +
        translatedInstruction.slice(1) +
        ". Preserve subject identity, framing, and realistic lighting.",
    };
  }

  async function postJson(endpoint, payload, fallbackFactory) {
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error("HTTP " + response.status);
      }
      return await response.json();
    } catch (error) {
      if (!state.useMockFallback) {
        throw error;
      }
      log("接口 " + endpoint + " 调用失败，已回退到 mock。");
      return fallbackFactory(payload);
    }
  }

  async function postJsonStrict(endpoint, payload) {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok || body.ok === false) {
      throw new Error(body.error || "HTTP " + response.status);
    }
    return body;
  }

  function buildSupportPayload(sourcePrompt, translatedInstruction, targetPrompt) {
    return {
      workflow: "source_anchor_no_target_hints",
      variant: "source_anchor",
      noTargetHints: true,
      imageBase64: state.resizedImage.base64,
      imageMimeType: "image/png",
      imageSize: 512,
      filename: state.selectedFile.name,
      userInstruction: refs.instructionInput.value.trim(),
      translatedInstruction,
      sourcePrompt,
      targetPrompt,
      targetTokenHints: [],
      sourceAnchorStart: Number(refs.sourceAnchorStartInput.value),
      numSteps: Number(refs.sharedStepsInput.value),
      supportLineArgs: {
        variant: "source_anchor",
        source_anchor_start: Number(refs.sourceAnchorStartInput.value),
        num_steps: Number(refs.sharedStepsInput.value),
        image_size: 512,
        target_token_hints: [],
      },
    };
  }

  function renderResult(response) {
    if (!response.editedImageUrl) {
      return;
    }
    const imageUrl = response.editedImageUrl + (response.editedImageUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
    refs.resultImage.src = imageUrl;
    refs.resultImage.hidden = false;
    refs.resultPlaceholder.hidden = true;
    refs.resultStage.classList.remove("empty");
    refs.openEditedLink.href = response.editedImageUrl;
    refs.openEditedLink.classList.remove("disabled");
  }

  async function runPipeline() {
    setError("");
    const validationError = validateSubmission();
    if (validationError) {
      setError(validationError);
      log(validationError);
      return;
    }

    if (!state.resizedImage) {
      await prepareImage(state.selectedFile);
    }

    resetStepState();
    resetResultView();
    setBusy(true);
    refs.sourcePromptOutput.textContent = "正在生成 source prompt...";
    refs.translatedInstructionOutput.textContent = "正在翻译指令...";
    refs.targetPromptOutput.textContent = "正在生成 target prompt...";
    log("开始执行整条链路。");

    try {
      setStepState("source", "active");
      const visionPayload = {
        filename: state.selectedFile.name,
        imageBase64: state.resizedImage.base64,
        imageMimeType: "image/png",
        imageSize: 512,
      };
      const visionResult = await postJson("/api/vision/source-prompt", visionPayload, mockVisionResponse);
      setRuntimeApiBadge(visionResult.mode === "mimo" ? "Mimo 识图" : "Mock 回退");
      if (visionResult.warning) {
        log("识图警告：" + visionResult.warning);
      }
      refs.sourcePromptOutput.textContent = visionResult.sourcePrompt;
      setStepState("source", "done");
      log("source prompt 已生成。");

      setStepState("target", "active");
      const promptPayload = {
        sourcePrompt: visionResult.sourcePrompt,
        userInstruction: refs.instructionInput.value.trim(),
      };
      const promptResult = await postJson("/api/prompt/compose-target", promptPayload, mockPromptResponse);
      setRuntimeApiBadge(promptResult.mode === "mimo" ? "Mimo Prompt" : "规则回退");
      if (promptResult.warning) {
        log("Prompt 警告：" + promptResult.warning);
      }
      refs.translatedInstructionOutput.textContent = promptResult.translatedInstruction;
      refs.targetPromptOutput.textContent = promptResult.targetPrompt;
      setStepState("target", "done");
      log("target prompt 已生成。");

      setStepState("backend", "active");
      const supportPayload = buildSupportPayload(
        visionResult.sourcePrompt,
        promptResult.translatedInstruction,
        promptResult.targetPrompt,
      );
      const backendResult = await postJsonStrict("/api/source-anchor/run", supportPayload);
      renderResult(backendResult);
      setRuntimeApiBadge("已完成");
      setStepState("backend", "done");
      log("编辑后的图片已生成。");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setError("运行失败：" + message);
      log("运行失败：" + message);
    } finally {
      setBusy(false);
    }
  }

  refs.imageInput.addEventListener("change", async function (event) {
    const input = event.currentTarget;
    const file = input.files && input.files[0];
    if (!file) {
      return;
    }
    state.selectedFile = file;
    state.resizedImage = null;
    setError("");
    try {
      await prepareImage(file);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      state.selectedFile = null;
      state.resizedImage = null;
      refs.previewImage.hidden = true;
      refs.previewPlaceholder.hidden = false;
      setError("图片读取失败：" + message);
      log("图片读取失败：" + message);
    }
  });

  refs.composerForm.addEventListener("submit", function (event) {
    event.preventDefault();
    if (state.isRunning) {
      return;
    }
    runPipeline();
  });

  refs.sourceAnchorStartInput.addEventListener("input", updateAnchorLabel);
  refs.sharedStepsInput.addEventListener("input", updateStepsLabel);
  refs.mockToggleButton.addEventListener("click", function () {
    state.useMockFallback = !state.useMockFallback;
    updateApiModeBadge();
    log("规则回退已" + (state.useMockFallback ? "开启。" : "关闭。"));
  });
  refs.clearLogButton.addEventListener("click", function () {
    refs.activityLog.innerHTML = "";
  });

  updateAnchorLabel();
  updateStepsLabel();
  updateApiModeBadge();
  resetStepState();
  resetResultView();
  log("页面已就绪，请上传图片并输入指令。");
})();
