function normalizeFilenameSubject(filename) {
  const stem = filename.replace(/\.[^.]+$/, "");
  const cleaned = stem.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
  return cleaned || "uploaded subject";
}

export function buildMockVisionResponse(payload) {
  const subject = normalizeFilenameSubject(payload.filename || "uploaded subject");
  return {
    ok: true,
    mode: "mock",
    sourcePrompt: `a realistic photo of ${subject}, clean composition, natural lighting, detailed texture, high fidelity`,
  };
}

export function buildMockPromptResponse(payload) {
  const translatedInstruction = /[\u3400-\u9fff]/.test(payload.userInstruction)
    ? "apply the requested edit and keep the rest unchanged"
    : payload.userInstruction.trim();

  return {
    ok: true,
    mode: "mock",
    translatedInstruction,
    targetPrompt: `${payload.sourcePrompt}. ${capitalize(translatedInstruction)}. Preserve subject identity, framing, and realistic lighting.`,
  };
}

function capitalize(value) {
  if (!value) {
    return value;
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}
