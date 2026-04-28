# worldKnower

Automated solver for [WorldGuessr](https://www.worldguessr.com/).

## What it does

1. Opens the WorldGuessr website with Playwright and starts a singleplayer round.
2. Takes a screenshot of the location panorama shown in the game.
3. Sends the screenshot to **Gemini Vision** (primary) and requests a strict
   one-line location format: `region, country`.
4. Falls back to **Hugging Face** vision models (if `HF_TOKEN` is set).
5. Falls back to local BLIP captioning if remote services fail or keys are missing.
6. Parses the AI answer to identify the country / continent.
7. Converts that geographic description to approximate (lat, lon) coordinates.
8. Finds the minimap widget in the corner of the page.
9. Converts lat/lon → pixel position using **Web-Mercator projection** (matching Leaflet.js).
10. Clicks the minimap at the correct location and submits the guess.

## Requirements

- Python 3.11+
- A Chromium browser (installed via Playwright)

```
pip install -r requirements.txt
playwright install chromium
```

Create a `.env` file in the project root and add keys:

```
GEMINI_API_KEY=your_gemini_key_here
# Optional overrides
# GEMINI_MODEL=gemini-2.0-flash
# GEMINI_MODEL_FALLBACKS=gemini-2.0-flash-lite,gemini-flash-latest
# GEMINI_MAX_RETRIES=2
# HF_TOKEN=your_huggingface_key_here
# LOCAL_CAPTION_MODEL=Salesforce/blip-image-captioning-base
```

Optional local model override:

```
export LOCAL_CAPTION_MODEL=Salesforce/blip-image-captioning-base
```

## Usage

```
python worldknower.py
```

Verbose live logs:

```
python worldknower.py --verbose
```

The script opens a visible browser window when a display is available,
and automatically uses headless mode in container/headless environments.
Two screenshots are saved to `/tmp/`:

| File | Content |
|---|---|
| `/tmp/worldguessr_location.png` | The panorama screenshot sent to the AI |
| `/tmp/worldguessr_result.png`   | The final state after submitting the guess |

## How the free AI works

`worldknower.py` uses the [Hugging Face Inference API](https://huggingface.co/inference-api)
with `Salesforce/blip-vqa-base` / captioning models as fallback.
Primary geolocation is Gemini Vision, prompted to return only `region, country`
for more stable parsing and matching.
If Gemini returns rate limits (`HTTP 429`), the script now switches to fallback
Gemini models and uses retry hints when available.
Gemini calls use the official `google-genai` client first, then REST fallback.
If your key has `limit: 0` free-tier quota, Gemini will be skipped quickly and
the script will continue with fallback models.

## Running the tests

```
python -m pytest test_worldknower.py -v
```

The test suite covers the pure-Python helper functions
(`description_to_latlon` and `latlon_to_minimap_pixel`) with 14 test cases.
