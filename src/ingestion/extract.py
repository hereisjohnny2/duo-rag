"""Extração de texto de PDFs: texto nativo quando disponível, OCR quando o
PDF é um scan (página sem camada de texto).

Cada documento processado é salvo em ``data/processed/<nome>.json`` com o
texto por página e metadados (se usou OCR, hash do arquivo original, etc.).
O hash do arquivo é usado para pular reprocessamento de documentos que não
mudaram (ingestão incremental).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (nome novo do pacote; fitz é o alias legado)
import pytesseract

from src import config
from src.ingestion.ocr import ocr_image_bytes

logger = logging.getLogger(__name__)

# Resolução usada para renderizar páginas escaneadas antes do OCR.
_OCR_RENDER_ZOOM = 2.0  # ~144 DPI (72 * zoom)


@dataclass
class PageResult:
    page_number: int
    text: str
    used_ocr: bool


@dataclass
class DocumentResult:
    source_file: str
    file_hash: str
    processed_at: str
    num_pages: int
    used_ocr: bool
    pages: list[PageResult]

    def to_dict(self) -> dict:
        return asdict(self)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _processed_path_for(pdf_path: Path) -> Path:
    return config.PROCESSED_DIR / f"{pdf_path.stem}.json"


def is_already_processed(pdf_path: Path) -> bool:
    """Verifica se o PDF já foi processado e não mudou desde então."""
    out_path = _processed_path_for(pdf_path)
    if not out_path.exists():
        return False
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return existing.get("file_hash") == _file_hash(pdf_path)


def extract_document(pdf_path: Path) -> DocumentResult:
    """Extrai o texto de todas as páginas de um PDF.

    Para cada página: tenta extrair texto nativo; se a página não tiver
    texto suficiente (provável scan), renderiza a página como imagem e
    roda OCR sobre ela.
    """
    doc = fitz.open(pdf_path)
    pages: list[PageResult] = []
    doc_used_ocr = False

    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            native_text = page.get_text().strip()

            if len(native_text) >= config.MIN_NATIVE_TEXT_CHARS:
                pages.append(
                    PageResult(
                        page_number=page_index + 1,
                        text=native_text,
                        used_ocr=False,
                    )
                )
                continue

            logger.info(
                "Página %s de %s parece ser um scan; rodando OCR",
                page_index + 1,
                pdf_path.name,
            )
            pix = page.get_pixmap(matrix=fitz.Matrix(_OCR_RENDER_ZOOM, _OCR_RENDER_ZOOM))
            image_bytes = pix.tobytes("png")
            try:
                ocr_text = ocr_image_bytes(image_bytes)
            except pytesseract.TesseractNotFoundError as exc:
                raise RuntimeError(
                    "O Tesseract OCR não foi encontrado. Esta página do PDF "
                    f"'{pdf_path.name}' parece ser um scan e precisa de OCR. "
                    "Instale o Tesseract OCR (veja o README) ou configure a "
                    "variável de ambiente DUO_RAG_TESSERACT_CMD apontando "
                    "para o executável do Tesseract."
                ) from exc
            except PermissionError as exc:
                raise RuntimeError(
                    "O Tesseract foi encontrado, mas o Windows negou "
                    "permissão para executá-lo (WinError 5 / Access is "
                    "denied). Isso costuma acontecer quando o Tesseract não "
                    "está corretamente resolvível pelo PATH. Defina a "
                    "variável de ambiente DUO_RAG_TESSERACT_CMD com o "
                    "caminho completo do tesseract.exe (ex.: "
                    r"C:\Users\<seu_usuario>\AppData\Local\Tesseract-OCR\tesseract.exe"
                    ") e tente novamente."
                ) from exc
            doc_used_ocr = True
            pages.append(
                PageResult(
                    page_number=page_index + 1,
                    text=ocr_text,
                    used_ocr=True,
                )
            )
    finally:
        doc.close()

    return DocumentResult(
        source_file=pdf_path.name,
        file_hash=_file_hash(pdf_path),
        processed_at=datetime.now(timezone.utc).isoformat(),
        num_pages=len(pages),
        used_ocr=doc_used_ocr,
        pages=pages,
    )


def extract_and_save(pdf_path: Path, force: bool = False) -> Path:
    """Extrai o texto do PDF e salva o resultado em ``data/processed``.

    Retorna o caminho do JSON gerado. Pula o processamento se o arquivo já
    foi processado e não mudou (a menos que ``force=True``).
    """
    out_path = _processed_path_for(pdf_path)

    if not force and is_already_processed(pdf_path):
        logger.info("Pulando %s (já processado e sem alterações)", pdf_path.name)
        return out_path

    logger.info("Extraindo texto de %s", pdf_path.name)
    result = extract_document(pdf_path)
    out_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Salvo %s (%s páginas, OCR usado: %s)",
        out_path.name,
        result.num_pages,
        result.used_ocr,
    )
    return out_path
