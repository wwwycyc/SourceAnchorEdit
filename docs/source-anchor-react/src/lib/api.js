const rawApiBase = String(import.meta.env.VITE_API_BASE_URL || "").trim();
const apiBase = rawApiBase.replace(/\/+$/, "");

export function resolveApiUrl(path) {
  if (!path) {
    return "";
  }
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  if (!apiBase) {
    return path;
  }
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${apiBase}${normalizedPath}`;
}

export async function postJson(endpoint, payload) {
  const response = await fetch(resolveApiUrl(endpoint), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof body?.error === "string" ? body.error : `HTTP ${response.status}`;
    throw new Error(message);
  }
  return body;
}
