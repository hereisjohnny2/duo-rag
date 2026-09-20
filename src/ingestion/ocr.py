"""Wrapper de OCR (Tesseract) para páginas de PDF sem texto extraível."""
from __future__ import annotations

import io
import logging

import pytesseract
from PIL import Image

from src import config

logger = logging.getLogger(__name__)

_configured = False


def _ensure_tesseract_configured() -> None:
    """Aplica o caminho customizado do binário do Tesseract, se definido."""
    global _configured
    if _configured:
        return
    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    _configured = True


def ocr_image_bytes(image_bytes: bytes, lang: str = config.OCR_LANGUAGE) -> str:
    """Roda OCR sobre uma imagem (bytes PNG/JPEG) e retorna o texto extraído."""
    _ensure_tesseract_configured()
    image = Image.open(io.BytesIO(image_bytes))
    try:
        text = pytesseract.image_to_string(image, lang=lang)
    except pytesseract.TesseractError as exc:
        logger.error("Falha ao rodar OCR: %s", exc)
        raise
    return text.strip()
