#!/usr/bin/env python3
"""
worldknower – Automated WorldGuessr solver.

Workflow:
  1. Open https://www.worldguessr.com/ and start a singleplayer round.
  2. Take a screenshot of the location panorama.
    3. Send the screenshot to Gemini Vision and request a strict
         `region, country` reply.
    4. Fall back to Hugging Face / local captioning when needed.
    5. Parse the AI answer to identify the geographic region.
    6. Convert that region to approximate lat/lon coordinates.
    7. Find the minimap in the corner, convert lat/lon → pixel (Mercator),
     and click the correct spot.
    8. Submit the guess.
"""

import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from playwright.async_api import Page, async_playwright

load_dotenv()

_REQ_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Hugging Face Inference API (router endpoint; token-based auth)
# ---------------------------------------------------------------------------
_HF_VQA_URL = (
    "https://router.huggingface.co/hf-inference/models/Salesforce/blip-vqa-base"
)
_HF_CAPTION_URLS = [
    "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-large",
    "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base",
    "https://router.huggingface.co/hf-inference/models/nlpconnect/vit-gpt2-image-captioning",
]
_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
_LOCAL_CAPTION_MODEL = os.environ.get(
    "LOCAL_CAPTION_MODEL", "Salesforce/blip-image-captioning-base"
)
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
_GEMINI_MODEL_FALLBACKS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_MODEL_FALLBACKS", "gemini-2.0-flash-lite,gemini-flash-latest"
    ).split(",")
    if m.strip()
]
_GEMINI_MAX_RETRIES = max(1, int(os.environ.get("GEMINI_MAX_RETRIES", "2")))

VERBOSE = False
_LOCAL_CAPTION_PIPELINE = None


def vprint(message: str) -> None:
    """Print verbose diagnostics only when verbose mode is enabled."""
    if VERBOSE:
        print(message)

# ---------------------------------------------------------------------------
# Geographic region → (latitude, longitude) lookup table
# Coordinates are approximate centres; good enough for a map click.
# ---------------------------------------------------------------------------
REGION_COORDS: dict[str, tuple[float, float]] = {
    # The Americas
    "united states": (39.50, -98.35),
    "usa": (39.50, -98.35),
    "america": (39.50, -98.35),
    "american": (39.50, -98.35),
    "canada": (56.13, -106.35),
    "canadian": (56.13, -106.35),
    "mexico": (23.63, -102.55),
    "mexican": (23.63, -102.55),
    "brazil": (-14.24, -51.93),
    "brazilian": (-14.24, -51.93),
    "argentina": (-38.42, -63.62),
    "chile": (-35.68, -71.54),
    "colombia": (4.57, -74.30),
    "peru": (-9.19, -75.02),
    "south america": (-14.24, -51.93),
    "north america": (39.50, -98.35),
    # Europe
    "united kingdom": (55.38, -3.44),
    "uk": (55.38, -3.44),
    "britain": (55.38, -3.44),
    "british": (55.38, -3.44),
    "england": (52.36, -1.17),
    "english": (52.36, -1.17),
    "scotland": (56.49, -4.20),
    "ireland": (53.14, -8.24),
    "french": (46.23, 2.21),
    "france": (46.23, 2.21),
    "germany": (51.17, 10.45),
    "german": (51.17, 10.45),
    "spain": (40.46, -3.75),
    "spanish": (40.46, -3.75),
    "italy": (41.87, 12.57),
    "italian": (41.87, 12.57),
    "portugal": (39.40, -8.22),
    "portuguese": (39.40, -8.22),
    "netherlands": (52.13, 5.29),
    "dutch": (52.13, 5.29),
    "belgium": (50.50, 4.47),
    "switzerland": (46.82, 8.23),
    "austria": (47.52, 14.55),
    "poland": (51.92, 19.15),
    "polish": (51.92, 19.15),
    "czech": (49.82, 15.47),
    "hungary": (47.16, 19.50),
    "romania": (45.94, 24.97),
    "ukraine": (48.38, 31.17),
    "russian": (61.52, 105.32),
    "russia": (61.52, 105.32),
    "turkey": (38.96, 35.24),
    "turkish": (38.96, 35.24),
    "greece": (39.07, 21.82),
    "greek": (39.07, 21.82),
    "sweden": (60.13, 18.64),
    "swedish": (60.13, 18.64),
    "norway": (60.47, 8.47),
    "norwegian": (60.47, 8.47),
    "finland": (61.92, 25.75),
    "finnish": (61.92, 25.75),
    "denmark": (56.26, 9.50),
    "danish": (56.26, 9.50),
    "europe": (54.53, 15.26),
    "european": (54.53, 15.26),
    # Asia
    "japan": (36.20, 138.25),
    "japanese": (36.20, 138.25),
    "china": (35.86, 104.20),
    "chinese": (35.86, 104.20),
    "south korea": (35.91, 127.77),
    "korean": (35.91, 127.77),
    "india": (20.59, 78.96),
    "indian": (20.59, 78.96),
    "thailand": (15.87, 100.99),
    "thai": (15.87, 100.99),
    "indonesia": (-0.79, 113.92),
    "indonesian": (-0.79, 113.92),
    "philippines": (12.88, 121.77),
    "filipino": (12.88, 121.77),
    "vietnam": (14.06, 108.28),
    "vietnamese": (14.06, 108.28),
    "malaysia": (4.21, 101.98),
    "singapore": (1.35, 103.82),
    "taiwan": (23.70, 121.00),
    "asia": (34.05, 100.62),
    "asian": (34.05, 100.62),
    "middle east": (29.31, 42.46),
    "israel": (31.05, 34.85),
    "saudi": (23.89, 45.08),
    "saudi arabia": (23.89, 45.08),
    "iran": (32.43, 53.69),
    "iraq": (33.22, 43.68),
    "jordan": (30.59, 36.24),
    # Africa
    "south africa": (-30.56, 22.94),
    "nigeria": (9.08, 8.68),
    "kenya": (-0.02, 37.91),
    "egypt": (26.82, 30.80),
    "egyptian": (26.82, 30.80),
    "morocco": (31.79, -7.09),
    "ghana": (7.96, -1.02),
    "africa": (8.78, 34.51),
    "african": (8.78, 34.51),
    # Oceania
    "australia": (-25.27, 133.78),
    "australian": (-25.27, 133.78),
    "new zealand": (-40.90, 174.89),
}


# ---------------------------------------------------------------------------
# AI image analysis (remote HF + local fallback)
# ---------------------------------------------------------------------------

def _wait_for_model(response: requests.Response) -> float:
    """Parse the 'estimated_time' from a 503 model-loading response."""
    try:
        data = response.json()
        return float(data.get("estimated_time", 20))
    except Exception:
        return 20.0


def _sleep_with_progress(seconds: float, label: str) -> None:
    """Sleep while printing progress updates so work is visible live."""
    total = max(1, int(round(seconds)))
    for i in range(total):
        vprint(f"  [{label}] waiting... {i + 1}/{total}s")
        time.sleep(1)


def preprocess_image_for_hf(image_bytes: bytes, max_side: int = 768) -> bytes:
    """Resize/compress screenshot so free inference endpoints accept payload size."""
    original_size = len(image_bytes)
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=78, optimize=True)
        processed = out.getvalue()

    vprint(
        "  [preprocess] image bytes "
        f"{original_size:,} -> {len(processed):,} "
        f"({(len(processed) / max(original_size, 1)) * 100:.1f}% of original)"
    )
    return processed


def _hf_headers(content_type: str) -> dict[str, str]:
    """Build Hugging Face request headers, adding auth token when available."""
    headers = {"Content-Type": content_type}
    if _HF_TOKEN:
        headers["Authorization"] = f"Bearer {_HF_TOKEN}"
    return headers


def _normalize_region_country(text: str) -> str:
    """Return a compact 'region, country' style location string."""
    cleaned = text.strip().replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip("` ")

    if not cleaned:
        return ""

    json_match = re.search(r"\{.*\}", cleaned)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            region = str(data.get("region", "")).strip()
            country = str(data.get("country", "")).strip()
            if country:
                return f"{region}, {country}".strip(", ")
        except Exception:
            pass

    cleaned = re.sub(r"(?i)^region\s*:\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\bcountry\s*:\s*", "", cleaned)
    cleaned = cleaned.strip(" ,")

    for sep in [".", ";", "|"]:
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()

    return cleaned


def _is_usable_region_country(text: str) -> bool:
    """Accept only likely 'region, country' values for map matching."""
    if not text:
        return False
    if "," not in text:
        return False
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return False
    if len(parts[-1]) < 3:
        return False
    return True


def _gemini_endpoint_for(model: str) -> str:
    model_id = model.removeprefix("models/")
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"


def _extract_gemini_retry_after(resp: requests.Response, default_wait: int) -> int:
    """Extract recommended wait time from headers/body for 429 responses."""
    retry_after = resp.headers.get("Retry-After", "").strip()
    if retry_after.isdigit():
        return max(1, int(retry_after))

    try:
        data = resp.json()
        details = data.get("error", {}).get("details", [])
        for d in details:
            delay = str(d.get("retryDelay", "")).strip()
            if delay.endswith("s"):
                return max(1, int(float(delay[:-1])))
    except Exception:
        pass

    return default_wait


def _is_quota_exhausted(resp: requests.Response) -> bool:
    """Detect non-recoverable quota exhaustion from Gemini error payload."""
    text = resp.text.lower()
    return "quota" in text or "resource_exhausted" in text


def query_gemini_region_country(image_bytes: bytes) -> str:
    """Ask Gemini Vision for one-line region,country output."""
    if not _GEMINI_API_KEY:
        return ""

    prompt = (
        "Identify the most likely location from this street-view image. "
        "Reply with exactly one short line in this format: <region>, <country>. "
        "No explanation, no markdown, no extra text."
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 64},
    }

    models_to_try = [_GEMINI_MODEL] + [m for m in _GEMINI_MODEL_FALLBACKS if m != _GEMINI_MODEL]

    # 1) Prefer official SDK path (as recommended in Gemini docs).
    try:
        from google import genai
        from google.genai import errors as genai_errors

        client = genai.Client(api_key=_GEMINI_API_KEY)
        for model in models_to_try:
            print(f"  [gemini-sdk] Using model: {model}")
            for attempt in range(_GEMINI_MAX_RETRIES):
                try:
                    vprint(
                        f"  [gemini-sdk] attempt {attempt + 1}/{_GEMINI_MAX_RETRIES} -> {model}"
                    )
                    response = client.models.generate_content(
                        model=model,
                        contents=[
                            prompt,
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                                }
                            },
                        ],
                    )
                    raw_text = (response.text or "").strip()
                    guess = _normalize_region_country(raw_text)
                    if _is_usable_region_country(guess):
                        return guess
                    if guess:
                        print(f"  [gemini-sdk] Ignoring non region,country reply: '{guess}'")
                        break
                except genai_errors.ClientError as exc:
                    msg = str(exc)
                    msg_lower = msg.lower()
                    if "resource_exhausted" in msg_lower or "429" in msg_lower:
                        if "limit: 0" in msg_lower or "quota exceeded" in msg_lower:
                            print(
                                "  [gemini-sdk] Quota exhausted (limit 0) for this model; switching model."
                            )
                            break
                        wait = 2 + attempt * 2
                        if attempt >= _GEMINI_MAX_RETRIES - 1:
                            print("  [gemini-sdk] HTTP 429 after retries; switching model.")
                            break
                        print(f"  [gemini-sdk] HTTP 429; retrying in {wait}s...")
                        _sleep_with_progress(wait, "gemini-sdk")
                        continue
                    if "404" in msg_lower or "not found" in msg_lower:
                        print(f"  [gemini-sdk] Model unavailable: {model}; switching model.")
                        break
                    if "401" in msg_lower or "permission_denied" in msg_lower:
                        print("  [gemini-sdk] Invalid GEMINI_API_KEY or permissions.")
                        return ""
                    print(f"  [gemini-sdk] Error: {msg[:180]}")
                    break
                except Exception as exc:
                    print(f"  [gemini-sdk] Error: {exc}")
                    break
    except Exception as exc:
        print(f"  [gemini-sdk] Unavailable ({exc}); falling back to REST.")

    # 2) REST fallback path.
    for model in models_to_try:
        endpoint = _gemini_endpoint_for(model)
        print(f"  [gemini] Using model: {model}")
        for attempt in range(_GEMINI_MAX_RETRIES):
            try:
                vprint(
                    f"  [gemini] POST attempt {attempt + 1}/{_GEMINI_MAX_RETRIES} -> {model}"
                )
                resp = requests.post(
                    endpoint,
                    params={"key": _GEMINI_API_KEY},
                    json=payload,
                    timeout=_REQ_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        raw_text = " ".join(str(p.get("text", "")) for p in parts).strip()
                        guess = _normalize_region_country(raw_text)
                        if _is_usable_region_country(guess):
                            return guess
                        if guess:
                            print(f"  [gemini] Ignoring non region,country reply: '{guess}'")
                            break
                elif resp.status_code == 429:
                    if _is_quota_exhausted(resp):
                        print("  [gemini] HTTP 429: quota exhausted for this model; switching model.")
                        break

                    wait = _extract_gemini_retry_after(resp, default_wait=2 + attempt * 2)
                    if attempt >= _GEMINI_MAX_RETRIES - 1:
                        print("  [gemini] HTTP 429 after retries; switching model.")
                        break
                    print(f"  [gemini] HTTP 429; retrying in {wait}s...")
                    _sleep_with_progress(wait, "gemini")
                elif resp.status_code == 503:
                    wait = _extract_gemini_retry_after(resp, default_wait=3 + attempt * 2)
                    if attempt >= _GEMINI_MAX_RETRIES - 1:
                        print("  [gemini] HTTP 503 after retries; switching model.")
                        break
                    print(f"  [gemini] HTTP 503; retrying in {wait}s...")
                    _sleep_with_progress(wait, "gemini")
                elif resp.status_code == 401:
                    print("  [gemini] HTTP 401: invalid GEMINI_API_KEY.")
                    return ""
                else:
                    print(f"  [gemini] HTTP {resp.status_code}: {resp.text[:160]}")
                    break
            except requests.RequestException as exc:
                print(f"  [gemini] Request error: {exc}")
                break

    return ""


def query_vqa(image_bytes: bytes, question: str) -> str:
    """
    Ask a visual question about the image using BLIP-VQA.
    Uses the Hugging Face Inference API.
    Returns the model's text answer, or '' on failure.
    """
    payload = {
        "inputs": {
            "question": question,
            "image": base64.b64encode(image_bytes).decode("utf-8"),
        }
    }
    vprint(f"  [VQA] payload size: {len(payload['inputs']['image']):,} b64 chars")
    for attempt in range(4):
        try:
            vprint(f"  [VQA] POST attempt {attempt + 1}/4 -> {_HF_VQA_URL}")
            resp = requests.post(
                _HF_VQA_URL,
                json=payload,
                headers=_hf_headers("application/json"),
                timeout=_REQ_TIMEOUT,
            )
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and result:
                    return str(result[0].get("answer", "")).strip()
                if isinstance(result, dict):
                    return str(result.get("answer", "")).strip()
            elif resp.status_code == 503:
                wait = _wait_for_model(resp)
                print(f"  [VQA] Model loading, retrying in {wait:.0f}s… "
                      f"(attempt {attempt + 1}/4)")
                _sleep_with_progress(min(wait, 30), "VQA")
            elif resp.status_code == 401:
                print("  [VQA] HTTP 401: Hugging Face token required.")
                print("  [VQA] Set HF_TOKEN in your environment and re-run.")
                break
            elif resp.status_code == 413:
                print("  [VQA] HTTP 413: payload too large even after preprocessing.")
                break
            else:
                print(f"  [VQA] HTTP {resp.status_code}: {resp.text[:120]}")
                break
        except requests.RequestException as exc:
            print(f"  [VQA] Request error: {exc}")
            break
    return ""


def query_caption(image_bytes: bytes) -> str:
    """
    Generate a caption for the image using BLIP-image-captioning-large.
    Used as a fallback when VQA returns no useful answer.
    Uses the Hugging Face Inference API.
    Returns the generated caption, or '' on failure.
    """
    for url in _HF_CAPTION_URLS:
        print(f"  [caption] Trying model: {url.rsplit('/', 1)[-1]}")
        for attempt in range(3):
            try:
                vprint(f"  [caption] POST attempt {attempt + 1}/3 -> {url}")
                resp = requests.post(
                    url,
                    data=image_bytes,
                    headers=_hf_headers("image/jpeg"),
                    timeout=_REQ_TIMEOUT,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if isinstance(result, list) and result:
                        return str(result[0].get("generated_text", "")).strip()
                    if isinstance(result, dict):
                        return str(result.get("generated_text", "")).strip()
                elif resp.status_code == 503:
                    wait = _wait_for_model(resp)
                    print(
                        f"  [caption] Model loading, retrying in {wait:.0f}s… "
                        f"(attempt {attempt + 1}/3)"
                    )
                    _sleep_with_progress(min(wait, 30), "caption")
                elif resp.status_code == 401:
                    print("  [caption] HTTP 401: Hugging Face token required.")
                    print("  [caption] Set HF_TOKEN in your environment and re-run.")
                    return ""
                else:
                    print(f"  [caption] HTTP {resp.status_code}: {resp.text[:120]}")
                    break
            except requests.RequestException as exc:
                print(f"  [caption] Request error: {exc}")
                break
    return ""


def query_local_caption(image_bytes: bytes) -> str:
    """Run local image captioning as a robust fallback when remote inference fails."""
    global _LOCAL_CAPTION_PIPELINE
    try:
        if _LOCAL_CAPTION_PIPELINE is None:
            print(f"  [local-caption] Loading local model: {_LOCAL_CAPTION_MODEL}")
            print("  [local-caption] First run may take a while (downloading weights).")
            from transformers import pipeline

            _LOCAL_CAPTION_PIPELINE = pipeline(
                "image-to-text",
                model=_LOCAL_CAPTION_MODEL,
            )

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = _LOCAL_CAPTION_PIPELINE(pil_image, max_new_tokens=40)
        if isinstance(result, list) and result:
            return str(result[0].get("generated_text", "")).strip()
        if isinstance(result, dict):
            return str(result.get("generated_text", "")).strip()
    except Exception as exc:
        print(f"  [local-caption] Error: {exc}")

    return ""


def analyze_location(image_bytes: bytes) -> str:
    """
    Identify the geographic location shown in a screenshot.
    Tries VQA first; falls back to image captioning.
    Returns a text description that contains geographic keywords.
    """
    prepared_image = preprocess_image_for_hf(image_bytes)

    if _GEMINI_API_KEY:
        print(f"  Trying Gemini geolocation model ({_GEMINI_MODEL})...")
        gemini_guess = query_gemini_region_country(prepared_image)
        if gemini_guess:
            print(f"  Gemini guess: '{gemini_guess}'")
            return gemini_guess
        print("  Gemini returned no usable location; falling back...")
    else:
        print("  GEMINI_API_KEY not set; skipping Gemini geolocation.")

    if not _HF_TOKEN:
        print("  HF_TOKEN is not set; skipping remote HF and using local caption fallback.")
        local_caption = query_local_caption(prepared_image)
        if local_caption:
            normalized = _normalize_region_country(local_caption)
            print(f"  Local caption: '{normalized or local_caption}'")
            return normalized or local_caption
        return ""

    questions = [
        "What country is this?",
        "What country or continent is shown in this image?",
        "Where in the world was this photo taken?",
    ]
    for question in questions:
        print(f"  Asking: '{question}'")
        answer = query_vqa(prepared_image, question)
        if answer:
            normalized = _normalize_region_country(answer)
            print(f"  Answer: '{normalized or answer}'")
            return normalized or answer

    print("  VQA returned no answer; trying image captioning…")
    caption = query_caption(prepared_image)
    if caption:
        normalized = _normalize_region_country(caption)
        print(f"  Caption: '{normalized or caption}'")
        return normalized or caption

    print("  Remote captioning returned nothing; trying local AI fallback…")
    local_caption = query_local_caption(prepared_image)
    normalized = _normalize_region_country(local_caption)
    print(f"  Local caption: '{normalized or local_caption}'")
    return normalized or local_caption


# ---------------------------------------------------------------------------
# Geographic coordinate helpers
# ---------------------------------------------------------------------------

def description_to_latlon(description: str) -> tuple[float, float]:
    """
    Scan a text description for known geographic keywords and return the
    best-matching (latitude, longitude) pair.  Falls back to (20, 0) —
    roughly the centre of the populated world — when nothing matches.
    """
    desc = description.lower()
    # Longer / more-specific phrases first to avoid partial matches
    for region in sorted(REGION_COORDS, key=len, reverse=True):
        if region in desc:
            lat, lon = REGION_COORDS[region]
            print(f"  Matched region '{region}' → lat={lat}, lon={lon}")
            return lat, lon
    print("  No region matched; defaulting to (20, 0)")
    return 20.0, 0.0


def latlon_to_minimap_pixel(
    lat: float,
    lon: float,
    map_bounds: dict[str, float],
    map_width_px: int,
    map_height_px: int,
) -> tuple[int, int]:
    """
    Convert geographic coordinates to a pixel position within the minimap
    using the Web-Mercator projection (same as Leaflet's default).

    map_bounds keys: "north", "south", "east", "west"  (all in degrees)
    Returns (x, y) pixel offset from the top-left corner of the minimap.
    """

    def _mercator_y(lat_deg: float) -> float:
        lat_rad = math.radians(lat_deg)
        return math.log(math.tan(math.pi / 4 + lat_rad / 2))

    x_frac = (lon - map_bounds["west"]) / (
        map_bounds["east"] - map_bounds["west"]
    )
    y_north = _mercator_y(min(map_bounds["north"], 85.0))
    y_south = _mercator_y(max(map_bounds["south"], -85.0))
    y_point = _mercator_y(max(-85.0, min(85.0, lat)))
    y_frac = (y_north - y_point) / (y_north - y_south)

    x_frac = max(0.0, min(1.0, x_frac))
    y_frac = max(0.0, min(1.0, y_frac))

    return int(x_frac * map_width_px), int(y_frac * map_height_px)


# ---------------------------------------------------------------------------
# WorldGuessr page interaction helpers
# ---------------------------------------------------------------------------

# Ordered list of CSS selectors tried when looking for a "start game" button
_START_SELECTORS = [
    "button:has-text('Singleplayer')",
    "button:has-text('Single Player')",
    "a:has-text('Singleplayer')",
    "button:has-text('Play')",
    "a:has-text('Play')",
    "[data-cy='play-btn']",
    ".play-btn",
    "#play",
]

# Buttons that often block the panorama on first load and should be dismissed.
_PREPARE_VIEW_SELECTORS = [
    "button:has-text('Skip tutorial')",
    "button:has-text('Skip')",
    "button:has-text('Show Street View')",
    "button:has-text('Start')",
    "button:has-text('Continue')",
    "button:has-text('OK')",
]

# Selectors that indicate the panorama/game view is active
_GAME_INDICATORS = [
    "canvas",
    ".panorama",
    "#pano",
    ".game-container",
    "[class*='panorama']",
    "[class*='streetview']",
    "[class*='game']",
]

# Selectors for the minimap element
_MINIMAP_SELECTORS = [
    ".minimap",
    "#minimap",
    ".guess-map",
    ".map-container",
    "[class*='minimap']",
    "[class*='mini-map']",
    "[id*='minimap']",
    ".leaflet-container",
    "#map",
]

# Selectors for the "submit guess" button
_SUBMIT_SELECTORS = [
    "button:has-text('Guess')",
    "button:has-text('Submit')",
    "button:has-text('Confirm')",
    ".guess-btn",
    "[class*='submit']",
    "[class*='guess']",
]


async def _try_click(page: Page, selector: str) -> bool:
    """Click the first visible element matching *selector*. Returns True on success."""
    try:
        el = await page.query_selector(selector)
        if el and await el.is_visible():
            await el.click()
            return True
    except Exception:
        pass
    return False


async def start_game(page: Page) -> None:
    """Navigate the WorldGuessr home screen to a singleplayer round."""
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(2000)

    for selector in _START_SELECTORS:
        if await _try_click(page, selector):
            print(f"  Clicked start-button: {selector}")
            await page.wait_for_timeout(2000)
            break

    # Dismiss tutorial overlays / reveal street view when present.
    for selector in _PREPARE_VIEW_SELECTORS:
        if await _try_click(page, selector):
            print(f"  Clicked view-prep button: {selector}")
            await page.wait_for_timeout(1200)

    # Some sites show a second confirmation modal
    secondary = [
        "button:has-text('Start')",
        "button:has-text('Play')",
        "button:has-text('OK')",
    ]
    for sel in secondary:
        if await _try_click(page, sel):
            print(f"  Clicked secondary button: {sel}")
            await page.wait_for_timeout(1500)
            break


async def wait_for_game_view(page: Page, timeout_ms: int = 20_000) -> bool:
    """Wait until panorama/game UI indicators are visible before screenshotting."""
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        for selector in _GAME_INDICATORS:
            try:
                el = await page.query_selector(selector)
                if el and await el.is_visible():
                    vprint(f"  [game] active indicator: {selector}")
                    return True
            except Exception:
                continue
        await page.wait_for_timeout(500)

    return False


async def find_minimap(page: Page) -> tuple | None:
    """
    Find the minimap element.
    Returns (element, bounding_box_dict) or None if not found.
    """
    for selector in _MINIMAP_SELECTORS:
        try:
            el = await page.query_selector(selector)
            if el and await el.is_visible():
                box = await el.bounding_box()
                if box and box["width"] > 10 and box["height"] > 10:
                    print(f"  Found minimap via '{selector}': {box}")
                    return el, box
        except Exception:
            pass
    return None


async def click_minimap(
    page: Page,
    minimap_box: dict,
    lat: float,
    lon: float,
) -> None:
    """
    Click the minimap at the pixel position corresponding to (lat, lon).
    Assumes the minimap shows the full world in Web-Mercator projection.
    """
    world_bounds = {"north": 85.0, "south": -85.0, "west": -180.0, "east": 180.0}
    w, h = int(minimap_box["width"]), int(minimap_box["height"])
    px, py = latlon_to_minimap_pixel(lat, lon, world_bounds, w, h)

    click_x = minimap_box["x"] + px
    click_y = minimap_box["y"] + py
    print(f"  Clicking minimap at page ({click_x:.0f}, {click_y:.0f}) "
          f"[map-offset ({px}, {py}) of {w}×{h}]")
    await page.mouse.click(click_x, click_y)
    await page.wait_for_timeout(800)


async def submit_guess(page: Page) -> None:
    """Click the first visible submit/guess button found on the page."""
    for selector in _SUBMIT_SELECTORS:
        if await _try_click(page, selector):
            print(f"  Guess submitted via '{selector}'")
            return
    print("  No submit button found; the click on the map may auto-submit.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run() -> None:
    print("=== worldknower starting ===")

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    headless_mode = not has_display
    if headless_mode:
        print("No GUI display detected; launching browser in headless mode.")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless_mode,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await ctx.new_page()

        # ── Step 1: navigate and start a game ────────────────────────────
        print("\n[1] Navigating to WorldGuessr…")
        await page.goto(
            "https://www.worldguessr.com/",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        print("[1] Starting singleplayer game…")
        await start_game(page)

        # Wait for the panorama/game UI to be ready before screenshotting
        ready = await wait_for_game_view(page)
        if not ready:
            print("  Warning: game view indicators not detected; continuing anyway.")
            await page.wait_for_timeout(2000)

        # ── Step 2: screenshot ────────────────────────────────────────────
        print("\n[2] Taking screenshot of the location…")
        screenshot_bytes = await page.screenshot(full_page=False, type="png")
        out_path = Path("/tmp/worldguessr_location.png")
        out_path.write_bytes(screenshot_bytes)
        print(f"    Saved to {out_path}")

        # ── Step 3: AI analysis ──────────────────────────────────────────
        print("\n[3] Analysing location with AI…")
        description = analyze_location(screenshot_bytes)

        if not description:
            print("    Warning: AI returned no description. "
                  "Defaulting to world centre.")
            description = "unknown"

        # ── Step 4: coordinates ──────────────────────────────────────────
        print("\n[4] Converting description to coordinates…")
        lat, lon = description_to_latlon(description)
        print(f"    Estimated location: lat={lat:.2f}, lon={lon:.2f}")

        # ── Step 5: find minimap and click ───────────────────────────────
        print("\n[5] Looking for minimap…")
        result = await find_minimap(page)

        if result:
            _el, box = result
            await click_minimap(page, box, lat, lon)
        else:
            print("    Minimap not found – skipping click.")

        # ── Step 6: submit guess ─────────────────────────────────────────
        print("\n[6] Submitting guess…")
        await submit_guess(page)

        # ── Save final screenshot ────────────────────────────────────────
        await page.wait_for_timeout(2000)
        final_path = Path("/tmp/worldguessr_result.png")
        final_path.write_bytes(await page.screenshot(type="png"))
        print(f"\n    Final screenshot saved to {final_path}")

        await browser.close()

    print("\n=== worldknower done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated WorldGuessr solver")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose live logs (API attempts, waits, payload sizes)",
    )
    args = parser.parse_args()

    VERBOSE = args.verbose
    asyncio.run(run())
