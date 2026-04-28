#!/usr/bin/env python3
"""
worldknower – Automated WorldGuessr solver.

Workflow:
  1. Open https://www.worldguessr.com/ and start a singleplayer round.
  2. Take a screenshot of the location panorama.
  3. Send the screenshot to the Hugging Face Inference API (BLIP VQA) –
     completely free, no API key required for anonymous use.
  4. Parse the AI answer to identify the geographic region.
  5. Convert that region to approximate lat/lon coordinates.
  6. Find the minimap in the corner, convert lat/lon → pixel (Mercator),
     and click the correct spot.
  7. Submit the guess.
"""

import asyncio
import base64
import math
import time
from pathlib import Path

import requests
from playwright.async_api import Page, async_playwright

# ---------------------------------------------------------------------------
# Hugging Face Inference API  (free, no API key needed for anonymous access)
# ---------------------------------------------------------------------------
_HF_VQA_URL = (
    "https://api-inference.huggingface.co/models/Salesforce/blip-vqa-base"
)
_HF_CAPTION_URL = (
    "https://api-inference.huggingface.co"
    "/models/Salesforce/blip-image-captioning-large"
)

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
# AI image analysis (free, no API key)
# ---------------------------------------------------------------------------

def _wait_for_model(response: requests.Response) -> float:
    """Parse the 'estimated_time' from a 503 model-loading response."""
    try:
        data = response.json()
        return float(data.get("estimated_time", 20))
    except Exception:
        return 20.0


def query_vqa(image_bytes: bytes, question: str) -> str:
    """
    Ask a visual question about the image using BLIP-VQA.
    Uses the Hugging Face Inference API anonymously (free, no API key).
    Returns the model's text answer, or '' on failure.
    """
    payload = {
        "inputs": {
            "question": question,
            "image": base64.b64encode(image_bytes).decode("utf-8"),
        }
    }
    for attempt in range(4):
        try:
            resp = requests.post(_HF_VQA_URL, json=payload, timeout=60)
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
                time.sleep(min(wait, 30))
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
    Uses the Hugging Face Inference API anonymously (free, no API key).
    Returns the generated caption, or '' on failure.
    """
    for attempt in range(4):
        try:
            resp = requests.post(
                _HF_CAPTION_URL,
                data=image_bytes,
                headers={"Content-Type": "image/png"},
                timeout=60,
            )
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and result:
                    return str(result[0].get("generated_text", "")).strip()
                if isinstance(result, dict):
                    return str(result.get("generated_text", "")).strip()
            elif resp.status_code == 503:
                wait = _wait_for_model(resp)
                print(f"  [caption] Model loading, retrying in {wait:.0f}s… "
                      f"(attempt {attempt + 1}/4)")
                time.sleep(min(wait, 30))
            else:
                print(f"  [caption] HTTP {resp.status_code}: {resp.text[:120]}")
                break
        except requests.RequestException as exc:
            print(f"  [caption] Request error: {exc}")
            break
    return ""


def analyze_location(image_bytes: bytes) -> str:
    """
    Identify the geographic location shown in a screenshot.
    Tries VQA first; falls back to image captioning.
    Returns a text description that contains geographic keywords.
    """
    questions = [
        "What country is this?",
        "What country or continent is shown in this image?",
        "Where in the world was this photo taken?",
    ]
    for question in questions:
        print(f"  Asking: '{question}'")
        answer = query_vqa(image_bytes, question)
        if answer:
            print(f"  Answer: '{answer}'")
            return answer

    print("  VQA returned no answer; trying image captioning…")
    caption = query_caption(image_bytes)
    print(f"  Caption: '{caption}'")
    return caption


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

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,
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

        # Extra wait for the panorama to render
        await page.wait_for_timeout(4000)

        # ── Step 2: screenshot ────────────────────────────────────────────
        print("\n[2] Taking screenshot of the location…")
        screenshot_bytes = await page.screenshot(full_page=False, type="png")
        out_path = Path("/tmp/worldguessr_location.png")
        out_path.write_bytes(screenshot_bytes)
        print(f"    Saved to {out_path}")

        # ── Step 3: AI analysis ──────────────────────────────────────────
        print("\n[3] Analysing with free AI (Hugging Face, no API key)…")
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
    asyncio.run(run())
