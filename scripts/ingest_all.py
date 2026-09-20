"""Script de ingestão completa.

Percorre ``data/raw``, garante que TODOS os PDFs sejam extraídos (texto
nativo ou OCR) para ``data/processed`` antes de qualquer indexação, e só
então indexa tudo no ChromaDB. Isso corresponde à etapa de "coletar todos
os textos antes de alimentar o RAG".

Uso:
    python scripts/ingest_all.py [--force]

    --force  reprocessa e reindexa mesmo documentos já processados.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.ingestion.chunk_and_index import index_all_processed
from src.ingestion.extract import extract_and_save
from src.ingestion.metadata_extractor import extract_all_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def collect_all_texts(force: bool = False) -> list[Path]:
    """Etapa 1: extrai o texto de TODOS os PDFs pendentes em data/raw.

    Retorna a lista de PDFs encontrados. Só depois que essa etapa termina
    (todo o texto coletado) é que a indexação no RAG é acionada.
    """
    pdf_paths = sorted(config.RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("Nenhum PDF encontrado em %s", config.RAW_DIR)
        return []

    logger.info("Encontrados %s PDF(s) em %s", len(pdf_paths), config.RAW_DIR)
    for pdf_path in pdf_paths:
        extract_and_save(pdf_path, force=force)

    logger.info("Coleta de texto concluída para todos os documentos.")
    return pdf_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocessa e reindexa mesmo documentos já processados",
    )
    args = parser.parse_args()

    pdf_paths = collect_all_texts(force=args.force)
    if not pdf_paths:
        return

    logger.info("Iniciando indexação no RAG (ChromaDB)...")
    total_chunks = index_all_processed()

    logger.info("Extraindo metadados estruturados (cidade, proprietários, valor, etc.)...")
    extract_all_metadata(force=args.force)

    logger.info(
        "Ingestão concluída: %s documento(s), %s chunk(s) indexado(s).",
        len(pdf_paths),
        total_chunks,
    )


if __name__ == "__main__":
    main()
