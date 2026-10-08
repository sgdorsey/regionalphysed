#!/usr/bin/env python3
"""
Demo 1: Alumnae Directory to CSV
================================

Source: Office of Civilian Defense, Division of Physical Fitness).
Each page lists about 50 graduates, so each image becomes MANY rows.

How it works:
  1. FIELDS is our codebook: one line per column, with instructions.
  2. The codebook is turned into a schema. Gemini must answer with JSON
     that has exactly these fields -- no extra text, no missing columns.
  3. Python's csv module writes the file, so commas inside the data
     ("12 Vine St., Sharon, Pa.") can't break the columns.
  4. Code adds what code can do reliably: the image name, the page
     number on every row, and the death year pulled out of "Died 1898".

Output: output/directory.csv (all pages in images/directory/ combined)
"""

import csv
import json
import mimetypes
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Codebook ---------------------------------------------------------------
# One entry per column: (column name, instructions for the model).
# Change the columns here and the schema and CSV header change with them.

FIELDS = [
    ("regional_areas",
     "If the state is Connecticut, Maine, New Hampshire, Vermont, Massachusetts, Rhode Island then the regional area is 1. If the state is New York, New York City, Delaware, or New Jersey the regional area is 2."),
    ("surname",
     "Family name(s) exactly as printed, including hyphenated names (e.g. 'Jones', 'Saunders')"),
    ("given_names",
     "Given names and initials exactly as printed (e.g. 'Audley H. F.')."),
    ("title",
     "The title exactly as printed (e.g. 'Mr.', 'Dr.', 'Mrs.' 'Colonel')."),
    ("category_of_role",
     "The heading before the list of names. The document prints this in all caps. (e.g. 'STATE PHYSICAL FITNESS DIRECTORS')."),
    ("role",
     "The role listed after the name.  If it wraps onto a second line, join the lines with a space. Empty if there is no role. (e.g. 'Director of Athletics')"),
    ("institution",
        "The place listed after the role. (e.g. 'State Board of Health')"),
    ("institution_city",
     "The city or town from the listing, exactly as printed. Empty if there is no state. (e.g. 'Trenton')"),
    ("institution_state",
     "The state from the listing, exactly as printed "
     "(e.g. 'Delaware', 'New York'). Empty if there is no state."),
    ("page_number",
     "At the top of the page, with dashes on either side."),
    ("handwritten_mark",
     "'yes' if someone has added a handwritten mark (a check mark or a "
     "correction) on or next to this entry, otherwise 'no'."),
]

PROMPT = """This image is a page from the Office of Civilian Defense directory for the Office of Physical Fitness. Extract every person listed on the page,
one entry per person, in the order they appear.

Rules:
- Extract EVERY entry. Never skip an entry because a field is unclear or
  missing: leave that field empty instead. An entry with blank fields is
  better than a missing entry.
- Transcribe exactly as printed. Do not correct spelling, expand
  abbreviations, or modernize anything.
- The regional representatives and regional directors are at the top of the page and are separated from the general list. Make sure to include those in the listings.
- If an answer was crossed out and replaced, use the final answer and record
  the change in 'corrections'.
- An address that continues on the next line belongs to the same entry."""

# --- Settings ---------------------------------------------------------------

MODEL = "gemini-3.5-flash-lite"

# Paid-tier prices in US dollars per 1 million tokens: (input, output).
# Check https://ai.google.dev/gemini-api/docs/pricing -- prices change.
PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

SCRIPT_DIR = Path(__file__).parent
IMAGE_FOLDER = SCRIPT_DIR / "images"
OUTPUT_FILE = SCRIPT_DIR / "output" / "directory.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ----------------------------------------------------------------------------


def build_schema(fields):
    """Turn the codebook into a JSON schema: one page, with a list of entries."""
    entry = {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": description}
            for name, description in fields
        },
        "required": [name for name, _ in fields],
    }
    return {
        "type": "object",
        "properties": {
            "page_number": {
                "type": "string",
                "description": "The printed page number, or empty if none is visible.",
            },
            "entries": {"type": "array", "items": entry},
        },
        "required": ["page_number", "entries"],
    }


def extract_page(client, image_path):
    """Send one image to Gemini and return (parsed data, response)."""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)

    response = client.models.generate_content(
        model=MODEL,
        contents=[image_part, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",   # answer in JSON only...
            response_json_schema=build_schema(FIELDS),  # ...shaped like this
        ),
    )
    return json.loads(response.text), response



def calculate_cost(input_tokens, output_tokens):
    input_price, output_price = PRICES[MODEL]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def main():
    load_dotenv(SCRIPT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("No API key found. Copy .env.example to .env and add your GEMINI_API_KEY.")
        return

    images = sorted(p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images found in {IMAGE_FOLDER}")
        return

    client = genai.Client(api_key=api_key)
    all_rows = []
    total_cost = 0.0

    for image_path in images:
        print(f"Extracting {image_path.name} ...")
        data, response = extract_page(client, image_path)

        # Add the columns that code fills in, not the model
        for entry in data["entries"]:
            entry["source_image"] = image_path.name
            all_rows.append(entry)

        usage = response.usage_metadata
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        cost = calculate_cost(usage.prompt_token_count or 0, output_tokens)
        total_cost += cost

        print(f"  Page {data['page_number'] or '?'}: {len(data['entries'])} entries "
              f"(count them on the page -- do they match?)")
        print(f"  Tokens: {usage.prompt_token_count:,} in, {output_tokens:,} out   "
              f"Cost: ${cost:.6f}\n")

    # The codebook sets the column order; code-added columns go at the end
    columns = [name for name, _ in FIELDS] + ["source_image"]
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print("=" * 40)
    print(f"Saved {len(all_rows)} rows to {OUTPUT_FILE.relative_to(SCRIPT_DIR)}")
    print(f"Total cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
