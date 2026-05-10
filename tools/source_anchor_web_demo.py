from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib import parse as urllib_parse
from urllib import error as urllib_error
from urllib import request as urllib_request

from PIL import Image, ImageOps
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
STATIC_ROOT = REPO_ROOT / "docs" / "source-anchor-web"
DEFAULT_JOB_ROOT = REPO_ROOT / "runs" / "source_anchor_web_demo" / "jobs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "source_anchor_web_demo" / "outputs"
WEB_CONFIG_EXAMPLE_PATH = REPO_ROOT / "configs" / "web" / "source_anchor_web.example.yaml"
WEB_CONFIG_LOCAL_PATH = REPO_ROOT / "configs" / "web" / "source_anchor_web.local.yaml"
KNOWN_TERM_TRANSLATIONS = {
    "\u732b": "cat",
    "\u732b\u54aa": "cat",
    "\u5c0f\u732b": "cat",
    "\u72d7": "dog",
    "\u5c0f\u72d7": "dog",
    "\u80cc\u666f": "background",
    "\u5916\u5957": "jacket",
    "\u8863\u670d": "clothes",
    "\u88d9\u5b50": "dress",
    "\u5e3d\u5b50": "hat",
    "\u773c\u955c": "glasses",
    "\u5934\u53d1": "hair",
    "\u6905\u5b50": "chair",
    "\u684c\u5b50": "table",
    "\u6c99\u53d1": "sofa",
    "\u7ea2\u8272": "red",
    "\u84dd\u8272": "blue",
    "\u7eff\u8272": "green",
    "\u6df1\u7eff\u8272": "dark green",
    "\u9ec4\u8272": "yellow",
    "\u9ed1\u8272": "black",
    "\u767d\u8272": "white",
    "\u7070\u8272": "gray",
    "\u68d5\u8272": "brown",
    "\u7d2b\u8272": "purple",
    "\u6a59\u8272": "orange",
}

RUN_LOCK = threading.Lock()
_WEB_CONFIG_CACHE: dict | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the source-anchor web demo with local mock APIs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--job-root", default=str(DEFAULT_JOB_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_web_demo_config() -> dict:
    global _WEB_CONFIG_CACHE
    if _WEB_CONFIG_CACHE is not None:
        return _WEB_CONFIG_CACHE

    if not WEB_CONFIG_EXAMPLE_PATH.exists():
        raise FileNotFoundError(f"Web demo example config not found: {WEB_CONFIG_EXAMPLE_PATH}")
    example_payload = yaml.safe_load(WEB_CONFIG_EXAMPLE_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(example_payload, dict):
        raise ValueError("Web demo example config must be a mapping.")

    local_payload: dict = {}
    if WEB_CONFIG_LOCAL_PATH.exists():
        loaded = yaml.safe_load(WEB_CONFIG_LOCAL_PATH.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Web demo local config must be a mapping.")
        local_payload = loaded

    _WEB_CONFIG_CACHE = _merge_dicts(example_payload, local_payload)
    return _WEB_CONFIG_CACHE


def decode_base64_image(raw_value: str) -> bytes:
    value = raw_value.split(",", 1)[-1].strip()
    return base64.b64decode(value)


def resize_to_square(image_bytes: bytes, size: int = 512) -> tuple[bytes, tuple[int, int]]:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        original_size = rgb.size
        contained = ImageOps.contain(rgb, (size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), color=(243, 247, 241))
        offset = ((size - contained.width) // 2, (size - contained.height) // 2)
        canvas.paste(contained, offset)
        buffer = BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue(), original_size


def normalize_filename_subject(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return stem or "uploaded subject"


def build_mock_source_prompt(filename: str) -> str:
    subject = normalize_filename_subject(filename)
    return (
        f"a realistic photo of {subject}, clean composition, natural lighting, "
        "detailed surface texture, high fidelity"
    )


def build_rule_based_translated_instruction(user_instruction: str) -> str:
    normalized = re.sub(r"\s+", "", user_instruction.strip())
    pattern = re.search(
        r"\u628a(.+?)\u6539\u6210(.+?)(?:[\uff0c,](?:\u5176\u4f59|\u5176\u4ed6)\u4e0d\u53d8)?$",
        normalized,
    )
    if pattern:
        source_term = KNOWN_TERM_TRANSLATIONS.get(pattern.group(1).strip(), pattern.group(1).strip())
        target_term = KNOWN_TERM_TRANSLATIONS.get(pattern.group(2).strip(), pattern.group(2).strip())
        return f"change the {source_term} to {target_term}, keep everything else unchanged"
    if any("\u3400" <= char <= "\u9fff" for char in user_instruction):
        return "apply the requested edit and keep the rest unchanged"
    return user_instruction.strip() or "apply the requested edit"


def build_rule_based_target_prompt(source_prompt: str, translated_instruction: str) -> str:
    base_prompt = source_prompt.strip().rstrip(".")
    sentence = translated_instruction.strip().rstrip(".")
    replace_match = re.search(
        r"change the (.+?) to (.+?)(?:,| and)? keep (?:everything else|the rest) unchanged",
        sentence,
        flags=re.IGNORECASE,
    )
    if replace_match:
        source_term = replace_match.group(1).strip()
        target_term = replace_match.group(2).strip()
        candidates = [source_term]
        lowered = source_term.lower()
        if lowered.startswith("a "):
            candidates.append(source_term[2:].strip())
        if lowered.startswith("an "):
            candidates.append(source_term[3:].strip())
        if lowered.startswith("the "):
            candidates.append(source_term[4:].strip())
        for candidate in candidates:
            if not candidate:
                continue
            pattern = re.compile(rf"\b{re.escape(candidate)}\b", flags=re.IGNORECASE)
            replaced_prompt, count = pattern.subn(target_term, base_prompt, count=1)
            if count > 0:
                return replaced_prompt + "."
    if sentence and not sentence.endswith("."):
        sentence = sentence + "."
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return f"{base_prompt}. {sentence} Preserve subject identity, framing, and realistic lighting.".strip()


def build_mock_translated_instruction(user_instruction: str) -> str:
    pattern = re.search(r"把(.+?)改成(.+?)(?:，|,)?其余不变", user_instruction.strip())
    if pattern:
        source_term = pattern.group(1).strip()
        target_term = pattern.group(2).strip()
        return f"change the {source_term} to {target_term}, keep everything else unchanged"
    if any("\u3400" <= char <= "\u9fff" for char in user_instruction):
        return "apply the requested edit and keep the rest unchanged"
    return user_instruction.strip() or "apply the requested edit"


def build_mock_target_prompt(source_prompt: str, translated_instruction: str) -> str:
    base_prompt = source_prompt.strip().rstrip(".")
    sentence = translated_instruction.strip().rstrip(".")
    if sentence and not sentence.endswith("."):
        sentence = sentence + "."
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return (
        f"{base_prompt}. {sentence} Preserve subject identity, framing, and realistic lighting."
    ).strip()


def get_mimo_api_key() -> str:
    return str(load_web_demo_config().get("web_demo", {}).get("mimo", {}).get("api_key") or "").strip()


def get_mimo_base_url() -> str:
    value = str(load_web_demo_config().get("web_demo", {}).get("mimo", {}).get("base_url") or "").strip()
    return value.rstrip("/")


def get_mimo_model() -> str:
    return str(load_web_demo_config().get("web_demo", {}).get("mimo", {}).get("model") or "").strip()


def get_source_anchor_model_id() -> str:
    return str(load_web_demo_config().get("web_demo", {}).get("source_anchor", {}).get("sd_model") or "").strip()


def get_source_anchor_clip_model_id() -> str:
    return str(load_web_demo_config().get("web_demo", {}).get("source_anchor", {}).get("clip_model") or "").strip()


def get_source_anchor_ntip2p_root() -> str:
    value = str(load_web_demo_config().get("web_demo", {}).get("source_anchor", {}).get("ntip2p_root") or "").strip()
    return value


def normalize_shared_steps(value: object) -> int:
    try:
        steps = int(value)
    except (TypeError, ValueError):
        configured = load_web_demo_config().get("web_demo", {}).get("source_anchor", {}).get("num_steps")
        try:
            return max(1, min(int(configured), 50))
        except (TypeError, ValueError):
            return 5
    return max(1, min(steps, 50))


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                clean = part.strip()
                if clean:
                    text_parts.append(clean)
                continue
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                text_parts.append(text_value.strip())
        return "\n".join(text_parts).strip()
    raise ValueError(f"Unsupported completion content format: {type(content)!r}")


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in model response: {text!r}")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model JSON response is not an object.")
    return payload


def call_mimo_chat_completion(messages: list[dict], *, max_completion_tokens: int = 1024) -> str:
    payload = {
        "model": get_mimo_model(),
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib_request.Request(
        url=f"{get_mimo_base_url()}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_mimo_api_key()}",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Mimo API HTTP {exc.code}: {error_body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Mimo API network error: {exc}") from exc

    payload = json.loads(raw)
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError(f"Mimo API returned no choices: {payload}")
    message = choices[0].get("message") or {}
    return _content_to_text(message.get("content"))


def call_mimo_chat_completion_with_retry(messages: list[dict], *, max_completion_tokens: int = 1024) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            content = call_mimo_chat_completion(messages, max_completion_tokens=max_completion_tokens)
            if not content:
                raise ValueError("Mimo API returned empty content.")
            return content
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))
    assert last_error is not None
    raise last_error


def generate_source_prompt_with_mimo(*, image_base64: str, image_mime_type: str) -> str:
    image_url = f"data:{image_mime_type};base64,{image_base64}"
    response_text = call_mimo_chat_completion_with_retry(
        [
            {
                "role": "system",
                "content": (
                    "You generate English source prompts for an image editing pipeline. "
                    "Your description will be used as the baseline for editing, so describe the image in a way that "
                    "makes it easy to precisely specify edits later. "
                    "Capture distinctive visual details, spatial relationships between objects, materials and textures, "
                    "lighting and shadows, colors, and the overall atmosphere. "
                    "Use natural, concrete language. Avoid vague terms like 'nice' or 'beautiful'. "
                    "Output only the prompt text itself — no markdown, no headings, no labels, no quotes, no prefixes."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image as an English source prompt for image editing. "
                            "Focus on details that would help an editor know exactly what to change and what to keep. "
                            "Return only the prompt text."
                        ),
                    },
                ],
            },
        ],
        max_completion_tokens=300,
    )
    source_prompt = response_text.strip()
    if not source_prompt:
        raise ValueError("Mimo returned an empty source prompt.")
    return source_prompt


def translate_instruction_with_mimo(*, user_instruction: str) -> str:
    response_text = call_mimo_chat_completion_with_retry(
        [
            {
                "role": "system",
                "content": (
                    "Translate the user's image-edit instruction into natural English. "
                    "Preserve all details and constraints from the original instruction. "
                    "Use clear, actionable language suitable for an image editing pipeline. "
                    "Output only the translated instruction text — no markdown, no headings, no labels, no quotes, no prefixes."
                ),
            },
            {
                "role": "user",
                "content": f"Translate the following user instruction into English:\n{user_instruction}\n\nReturn only the translated text.",
            },
        ],
        max_completion_tokens=160,
    )
    translated_instruction = response_text.strip().strip('"').strip()
    if not translated_instruction:
        raise ValueError("Mimo returned an empty translated instruction.")
    return translated_instruction


def compose_target_prompt_with_mimo(*, source_prompt: str, translated_instruction: str) -> str:
    response_text = call_mimo_chat_completion_with_retry(
        [
            {
                "role": "system",
                "content": (
                    "You are helping an image editing pipeline. "
                    "Rewrite the source prompt so it reflects the requested edit while preserving all unmentioned details. "
                    "The output should match the level of detail and style of the source prompt. "
                    "Output only the target prompt text — no markdown, no headings, no labels, no quotes, no prefixes."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Source prompt:\n"
                    f"{source_prompt}\n\n"
                    "Translated edit instruction:\n"
                    f"{translated_instruction}\n\n"
                    "Return the final English target prompt text only."
                ),
            },
        ],
        max_completion_tokens=300,
    )
    target_prompt = response_text.strip().strip('"').strip()
    if not target_prompt:
        raise ValueError("Mimo returned an empty target prompt.")
    return target_prompt


class SourceAnchorWebDemoHandler(SimpleHTTPRequestHandler):
    job_root = DEFAULT_JOB_ROOT
    output_root = DEFAULT_OUTPUT_ROOT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def do_GET(self) -> None:
        if self.path in {"/healthz", "/api/healthz"}:
            self._write_json({"ok": True, "service": "source-anchor-web-demo"})
            return
        if self.path.startswith("/api/artifact"):
            try:
                self._handle_artifact()
            except Exception as exc:
                self._write_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        try:
            if self.path == "/api/vision/source-prompt":
                self._handle_source_prompt()
                return
            if self.path == "/api/prompt/compose-target":
                self._handle_compose_target()
                return
            if self.path == "/api/source-anchor/run":
                self._handle_source_anchor_run()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except Exception as exc:
            self._write_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print("[%s] %s" % (self.log_date_time_string(), format % args))

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _is_within_root(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    def _artifact_url(self, path: Path) -> str:
        encoded = urllib_parse.quote(str(path.resolve()))
        return f"/api/artifact?path={encoded}"

    def _handle_artifact(self) -> None:
        parsed = urllib_parse.urlparse(self.path)
        query = urllib_parse.parse_qs(parsed.query)
        raw_path = (query.get("path") or [""])[0]
        if not raw_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing artifact path")
            return
        artifact_path = Path(urllib_parse.unquote(raw_path)).resolve()
        allowed = self._is_within_root(artifact_path, self.job_root) or self._is_within_root(artifact_path, self.output_root)
        if not allowed:
            self.send_error(HTTPStatus.FORBIDDEN, "Artifact path is outside allowed roots")
            return
        if not artifact_path.exists() or not artifact_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
            return
        mime_type, _ = mimetypes.guess_type(str(artifact_path))
        data = artifact_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_source_prompt(self) -> None:
        payload = self._read_json()
        filename = str(payload.get("filename") or "uploaded-image.png")
        image_base64 = str(payload.get("imageBase64") or "").strip()
        image_mime_type = str(payload.get("imageMimeType") or "image/png").strip() or "image/png"
        try:
            source_prompt = generate_source_prompt_with_mimo(
                image_base64=image_base64,
                image_mime_type=image_mime_type,
            )
            mode = "mimo"
            warning = None
        except Exception as exc:
            print(f"[warn] source prompt fallback: {exc}")
            source_prompt = build_mock_source_prompt(filename)
            mode = "fallback_mock"
            warning = str(exc)
        self._write_json(
            {
                "ok": True,
                "mode": mode,
                "sourcePrompt": source_prompt,
                "warning": warning,
            }
        )

    def _handle_compose_target(self) -> None:
        payload = self._read_json()
        source_prompt = str(payload.get("sourcePrompt") or "").strip()
        user_instruction = str(payload.get("userInstruction") or "").strip()
        try:
            translated_instruction = translate_instruction_with_mimo(
                user_instruction=user_instruction,
            )
            target_prompt = compose_target_prompt_with_mimo(
                source_prompt=source_prompt,
                translated_instruction=translated_instruction,
            )
            mode = "mimo"
            warning = None
        except Exception as exc:
            print(f"[warn] target prompt fallback: {exc}")
            translated_instruction = build_rule_based_translated_instruction(user_instruction)
            target_prompt = build_rule_based_target_prompt(source_prompt, translated_instruction)
            mode = "fallback_rule"
            warning = str(exc)
        self._write_json(
            {
                "ok": True,
                "mode": mode,
                "translatedInstruction": translated_instruction,
                "targetPrompt": target_prompt,
                "warning": warning,
            }
        )

    def _handle_source_anchor_run(self) -> None:
        payload = self._read_json()
        filename = str(payload.get("filename") or "uploaded-image.png")
        source_prompt = str(payload.get("sourcePrompt") or "").strip()
        target_prompt = str(payload.get("targetPrompt") or "").strip()
        user_instruction = str(payload.get("userInstruction") or "").strip()
        translated_instruction = str(payload.get("translatedInstruction") or "").strip()
        source_anchor_start = float(payload.get("sourceAnchorStart") or 0.0)
        num_steps = normalize_shared_steps(payload.get("numSteps"))
        no_target_hints = bool(payload.get("noTargetHints", True))

        image_bytes = decode_base64_image(str(payload.get("imageBase64") or ""))
        image_512, original_size = resize_to_square(image_bytes, size=512)

        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job_dir = self.job_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        source_image_path = job_dir / "source_512.png"
        source_image_path.write_bytes(image_512)

        sample_json_path = job_dir / "sample.json"
        request_json_path = job_dir / "request.json"

        sample_payload = {
            "sample_id": job_id,
            "row_index": 0,
            "record_id": job_id,
            "source_image_path": str(source_image_path.resolve()),
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "target_token_hints": [],
            "core_input": {
                "source_image_path": str(source_image_path.resolve()),
                "target_prompt": target_prompt,
                "target_token_hints": [],
            },
            "metadata": {
                "source_prompt": source_prompt,
                "edit_prompt": user_instruction,
                "blended_word": None,
                "extras": {
                    "filename": filename,
                    "translated_instruction": translated_instruction,
                    "no_target_hints": no_target_hints,
                    "source_anchor_start": source_anchor_start,
                    "original_size": {
                        "width": original_size[0],
                        "height": original_size[1],
                    },
                    "processed_size": {
                        "width": 512,
                        "height": 512,
                    },
                },
                "has_gt_mask": False,
            },
            "target_reference_path": None,
        }
        request_payload = {
            "workflow": "source_anchor_no_target_hints",
            "variant": "source_anchor",
            "source_anchor_start": source_anchor_start,
            "no_target_hints": no_target_hints,
            "filename": filename,
            "source_prompt": source_prompt,
            "translated_instruction": translated_instruction,
            "target_prompt": target_prompt,
            "sample_json_path": str(sample_json_path.resolve()),
            "source_anchor_args": {
                "variant": "source_anchor",
                "sample_json": str(sample_json_path.resolve()),
                "source_anchor_start": source_anchor_start,
                "skip_metrics": True,
                "output_root": str(self.output_root.resolve()),
                "model_id": get_source_anchor_model_id(),
                "clip_model_id": get_source_anchor_clip_model_id(),
                "ntip2p_root": get_source_anchor_ntip2p_root(),
                "num_inversion_steps": num_steps,
                "num_edit_steps": num_steps,
                "method": "source_anchor",
                "dtype": "float16",
                "attention_slicing": True,
                "vae_slicing": True,
                "sample_batch_size": 1,
                "min_sample_batch_size": 1,
                "channels_last": True,
            },
        }
        save_json(sample_json_path, sample_payload)
        save_json(request_json_path, request_payload)

        prepared_command = (
            "python scripts\\run_single.py "
            f'--config "{(job_dir / "experiment.json").resolve()}"'
        )

        start_time = time.perf_counter()
        with RUN_LOCK:
            run_dir = self._execute_source_anchor(
                job_dir=job_dir,
                sample_json_path=sample_json_path,
                source_anchor_start=source_anchor_start,
                num_steps=num_steps,
            )
        duration_seconds = time.perf_counter() - start_time
        sample_output_dir = run_dir / "samples" / job_id
        method_name = "source_anchor"
        edited_image_path = sample_output_dir / "edited.png"
        overview_image_path = sample_output_dir / "overview.png"
        source_image_materialized_path = sample_output_dir / "source.png"
        if not edited_image_path.exists():
            raise FileNotFoundError(f"Edited image was not produced: {edited_image_path}")

        self._write_json(
            {
                "ok": True,
                "mode": "executed",
                "jobId": job_id,
                "jobDir": str(job_dir.resolve()),
                "sampleJsonPath": str(sample_json_path.resolve()),
                "requestJsonPath": str(request_json_path.resolve()),
                "runDir": str(run_dir.resolve()),
                "preparedCommand": prepared_command,
                "sampleId": job_id,
                "methodName": method_name,
                "editedImagePath": str(edited_image_path.resolve()),
                "editedImageUrl": self._artifact_url(edited_image_path),
                "overviewImagePath": str(overview_image_path.resolve()) if overview_image_path.exists() else None,
                "overviewImageUrl": self._artifact_url(overview_image_path) if overview_image_path.exists() else None,
                "sourceImageUrl": self._artifact_url(source_image_materialized_path) if source_image_materialized_path.exists() else None,
                "durationSeconds": round(duration_seconds, 2),
                "payloadSummary": {
                    "variant": "source_anchor",
                    "noTargetHints": no_target_hints,
                    "sourceAnchorStart": round(source_anchor_start, 4),
                    "imageSize": 512,
                    "numInversionSteps": num_steps,
                    "numEditSteps": num_steps,
                    "dtype": "float16",
                    "attentionSlicing": True,
                    "vaeSlicing": True,
                    "batchSize": 1,
                },
            }
        )

    def _execute_source_anchor(self, *, job_dir: Path, sample_json_path: Path, source_anchor_start: float, num_steps: int) -> Path:
        from sourceanchor.config import ExperimentConfig, MethodConfig, ModelConfig, RoiConfig, RuntimeConfig, SaveConfig
        from sourceanchor.input.loader import load_samples_from_path
        from sourceanchor.method.editor import run_experiment

        experiment_config_payload = {
            "experiment": {
                "input_manifest": str(sample_json_path.resolve()),
                "output_root": str(self.output_root.resolve()),
            },
            "method": {
                "name": "source_anchor",
                "no_target_hints": True,
                "source_anchor_start": float(source_anchor_start),
                "num_inversion_steps": int(num_steps),
                "num_edit_steps": int(num_steps),
                "guidance_scale": 7.5,
                "soft_roi_start_weight": 0.75,
                "soft_roi_end_weight": 0.10,
                "anchor_hardness_start": 0.35,
                "anchor_hardness_end": 1.0,
                "discrepancy_weight": 0.55,
                "attention_weight": 0.30,
                "latent_weight": 0.15,
                "temperature": 8.0,
                "threshold": 0.35,
                "min_value": 0.0,
                "max_value": 1.0,
                "smoothing_kernel": 5,
                "attention_locations": ["down", "mid", "up"],
            },
            "models": {
                "sd_model": get_source_anchor_model_id(),
                "clip_model": get_source_anchor_clip_model_id(),
                "ntip2p_root": get_source_anchor_ntip2p_root(),
                "dino_weights": None,
            },
            "roi": {
                "source": "live",
                "cache_root": None,
                "save_cache": False,
                "threshold": 0.5,
                "num_maps_per_mask": 1,
                "mask_encode_strength": 0.5,
                "mask_thresholding_ratio": 3.0,
            },
            "runtime": {
                "device": "cuda",
                "dtype": "float16",
                "batch_size": 1,
                "local_files_only": True,
                "attention_slicing": True,
                "vae_slicing": True,
                "channels_last": True,
                "enable_tf32": True,
                "enable_cpu_offload": False,
                "enable_xformers": False,
            },
            "save": {
                "roi_cache": False,
                "inversion_tensors": False,
                "debug_json": True,
                "step_visualizations": False,
                "overview": True,
            },
        }
        experiment_config_path = job_dir / "experiment.json"
        save_json(experiment_config_path, experiment_config_payload)
        config = ExperimentConfig(
            input_manifest=sample_json_path.resolve(),
            output_root=self.output_root.resolve(),
            method=MethodConfig(**experiment_config_payload["method"]),
            roi=RoiConfig(**experiment_config_payload["roi"]),
            models=ModelConfig(
                sd_model=experiment_config_payload["models"]["sd_model"],
                clip_model=experiment_config_payload["models"]["clip_model"],
                ntip2p_root=Path(experiment_config_payload["models"]["ntip2p_root"]).resolve(),
                dino_weights=experiment_config_payload["models"]["dino_weights"],
            ),
            runtime=RuntimeConfig(**experiment_config_payload["runtime"]),
            save=SaveConfig(**experiment_config_payload["save"]),
        )
        samples = load_samples_from_path(sample_json_path.resolve())
        return run_experiment(config, samples)


def main() -> None:
    args = build_arg_parser().parse_args()
    SourceAnchorWebDemoHandler.job_root = Path(args.job_root)
    SourceAnchorWebDemoHandler.output_root = Path(args.output_root)
    SourceAnchorWebDemoHandler.job_root.mkdir(parents=True, exist_ok=True)
    SourceAnchorWebDemoHandler.output_root.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), SourceAnchorWebDemoHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving source-anchor web demo at {url}")
    print(f"Static root: {STATIC_ROOT}")
    print(f"Job root: {SourceAnchorWebDemoHandler.job_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
