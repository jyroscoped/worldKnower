# worldKnower

Automated solver for [WorldGuessr](https://www.worldguessr.com/).

## What it does

1. Opens the WorldGuessr website with Playwright and starts a singleplayer round.
2. Takes a screenshot of the location panorama shown in the game.
3. Sends the screenshot to the **Hugging Face Inference API** (BLIP VQA model) —
   completely **free**, **no API key required** for anonymous use.
4. Parses the AI answer to identify the country / continent.
5. Converts that geographic description to approximate (lat, lon) coordinates.
6. Finds the minimap widget in the corner of the page.
7. Converts lat/lon → pixel position using **Web-Mercator projection** (matching Leaflet.js).
8. Clicks the minimap at the correct location and submits the guess.

## Requirements

- Python 3.11+
- A Chromium browser (installed via Playwright)

```
pip install -r requirements.txt
playwright install chromium
```

## Usage

```
python worldknower.py
```

The script opens a visible browser window so you can watch it work.
Two screenshots are saved to `/tmp/`:

| File | Content |
|---|---|
| `/tmp/worldguessr_location.png` | The panorama screenshot sent to the AI |
| `/tmp/worldguessr_result.png`   | The final state after submitting the guess |

## How the free AI works

`worldknower.py` uses the [Hugging Face Inference API](https://huggingface.co/inference-api)
with the `Salesforce/blip-vqa-base` model (Visual Question Answering).
It asks the model questions such as *"What country is this?"* about the screenshot.
No account and no API key are needed — anonymous requests are accepted within rate limits.
If VQA returns no answer, the script falls back to `Salesforce/blip-image-captioning-large`
for plain image captioning.

## Running the tests

```
python -m pytest test_worldknower.py -v
```

The test suite covers the pure-Python helper functions
(`description_to_latlon` and `latlon_to_minimap_pixel`) with 14 test cases.
