"""
src/ocr_helper.py
─────────────────
OCR helper for extracting text from sportsbook prop board screenshots.

Uses Tesseract via pytesseract.  The extracted text is printed to stdout
and saved to data/ocr_raw_output.txt for you to paste into an LLM.

Usage
-----
    python src/ocr_helper.py --image screenshots/props_today.png
    python src/ocr_helper.py --image screenshots/props_today.png --output data/ocr_raw_output.txt

After running, paste the content of data/ocr_raw_output.txt into your LLM
with the prompt from README.md to get structured JSON.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "ocr_raw_output.txt"

# Tesseract binary path from .env (Windows users need this)
_TESSERACT_CMD = os.getenv("TESSERACT_CMD")


# ---------------------------------------------------------------------------
# Tesseract setup
# ---------------------------------------------------------------------------

def _configure_tesseract() -> None:
    """Set pytesseract.pytesseract.tesseract_cmd from env if provided."""
    if _TESSERACT_CMD:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
            logger.debug("Tesseract cmd set to: %s", _TESSERACT_CMD)
        except ImportError:
            logger.error("pytesseract is not installed. Run: pip install pytesseract")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess_image(image: "Image.Image") -> "Image.Image":
    """Apply basic image preprocessing to improve OCR accuracy.

    Current steps:
    - Convert to greyscale.
    - Increase contrast (optional — can add PIL.ImageEnhance).
    - Resize if too small (Tesseract works best at 300+ DPI / large images).

    Parameters
    ----------
    image : PIL.Image.Image

    Returns
    -------
    PIL.Image.Image
        Pre-processed image.
    """
    # Greyscale
    img = image.convert("L")

    # TODO (optional): Enhance contrast
    # from PIL import ImageEnhance
    # img = ImageEnhance.Contrast(img).enhance(2.0)

    # TODO (optional): Upscale if image width < 1000px for better OCR
    # if img.width < 1000:
    #     scale = 1000 / img.width
    #     img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    return img


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def extract_text(image_path: str | Path, psm: int = 6) -> str:
    """Run Tesseract OCR on a screenshot and return the extracted text.

    Parameters
    ----------
    image_path : str or Path
        Path to the screenshot file.
    psm : int
        Tesseract page segmentation mode (PSM).
        - 3 = Fully automatic page segmentation (default for Tesseract)
        - 6 = Assume a single uniform block of text (good for prop boards)
        - 11 = Sparse text — good for irregular layouts

    Returns
    -------
    str
        Raw OCR text.
    """
    import pytesseract

    _configure_tesseract()

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info("Loading image: %s", image_path)
    image = Image.open(image_path)
    image = preprocess_image(image)

    config = f"--psm {psm} --oem 3"
    logger.info("Running Tesseract (psm=%d)...", psm)
    text = pytesseract.image_to_string(image, config=config)

    logger.info("Extracted %d characters.", len(text))
    return text


def extract_text_from_multiple(
    image_paths: list[str | Path],
    psm: int = 6,
) -> str:
    """Run OCR on multiple screenshots and concatenate results.

    Useful when you have a long prop board split across multiple screenshots.

    Parameters
    ----------
    image_paths : list
        List of image file paths.
    psm : int
        Tesseract PSM mode.

    Returns
    -------
    str
        Concatenated OCR text with a separator between each image's output.
    """
    parts = []
    for i, path in enumerate(image_paths, 1):
        logger.info("[%d/%d] Processing %s", i, len(image_paths), path)
        try:
            text = extract_text(path, psm=psm)
            parts.append(f"--- Screenshot {i}: {Path(path).name} ---\n{text}")
        except Exception as exc:
            logger.warning("Failed to OCR %s: %s", path, exc)
            parts.append(f"--- Screenshot {i}: ERROR ({exc}) ---\n")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM prompt helper
# ---------------------------------------------------------------------------

LLM_PROMPT_TEMPLATE = """
You are a data extractor for sports betting research.

Parse the following sportsbook prop board text into a JSON array.
Each element in the array must have exactly these fields:
  - player_name   : string, full player name
  - team          : string, team abbreviation (e.g. "LAL")
  - opponent      : string, opponent abbreviation (e.g. "GSW")
  - market        : string, stat type (e.g. "points", "rebounds", "assists",
                    "threepm", "points_rebounds", "points_assists",
                    "rebounds_assists", "points_rebounds_assists")
  - line          : number, the prop line (e.g. 24.5)
  - over_odds     : integer, American odds for the Over (e.g. -115)
  - under_odds    : integer, American odds for the Under (e.g. -105)
  - book          : string, sportsbook name
  - game_date     : string, date in YYYY-MM-DD format

Rules:
- Return ONLY the JSON array, with no explanation, markdown, or extra text.
- If a field cannot be determined, use null.
- If Over and Under odds are not separated, use the same value for both.
- Normalise market names to the canonical strings listed above.

---
{ocr_text}
---
""".strip()


def build_llm_prompt(ocr_text: str) -> str:
    """Format the LLM prompt with the OCR text inserted.

    Parameters
    ----------
    ocr_text : str

    Returns
    -------
    str
        Complete prompt ready to paste into an LLM.
    """
    return LLM_PROMPT_TEMPLATE.format(ocr_text=ocr_text)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="OCR a sportsbook screenshot to extract prop board text."
    )
    parser.add_argument(
        "--image", "-i",
        nargs="+",
        required=True,
        help="Path(s) to screenshot image file(s).",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output text file path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6)",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the formatted LLM prompt instead of raw OCR text.",
    )
    args = parser.parse_args()

    if len(args.image) == 1:
        text = extract_text(args.image[0], psm=args.psm)
    else:
        text = extract_text_from_multiple(args.image, psm=args.psm)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.show_prompt:
        output_text = build_llm_prompt(text)
    else:
        output_text = text

    output_path.write_text(output_text, encoding="utf-8")
    print(output_text)
    logger.info("\nSaved to %s", output_path)


if __name__ == "__main__":
    main()
