"""Chunking + geração de embeddings + indexação no ChromaDB.

Lê os JSONs de ``data/processed`` (texto já extraído/OCR'd), quebra o texto
em chunks e indexa no Chroma usando embeddings gerados localmente via
Ollama.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection():
    return get_chroma_client().get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Gera embeddings para uma lista de textos usando o Ollama local."""
    embeddings: list[list[float]] = []
    for text in texts:
        response = ollama.embeddings(model=config.OLLAMA_EMBEDDING_MODEL, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings


def _load_processed_document(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def _document_title(source_file: str) -> str:
    """Deriva um título legível a partir do nome do arquivo (sem extensão).

    Esse título é incluído no texto usado para gerar o embedding (mas não no
    texto exibido/citado), pois é comum o usuário se referir ao documento
    pelo nome do arquivo (ex.: "Escritura Marileira") mesmo quando essa
    palavra não aparece no corpo do texto extraído/OCR'd.
    """
    return Path(source_file).stem


def _build_chunks_for_document(
    doc: dict,
) -> tuple[list[str], list[str], list[dict], list[str]]:
    """Gera (textos_p/embedding, textos_para_exibir, metadados, ids) de um documento.

    Retorna duas listas de texto: uma enriquecida com o título do documento
    (usada apenas para gerar o embedding, melhorando a recuperação quando a
    pergunta cita o nome do arquivo) e outra com o texto original limpo
    (armazenada/exibida como fonte da resposta).
    """
    source_file = doc["source_file"]
    title = _document_title(source_file)
    embed_texts_input: list[str] = []
    display_texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for page in doc["pages"]:
        page_number = page["page_number"]
        page_text = page["text"]
        if not page_text.strip():
            continue
        page_chunks = _splitter.split_text(page_text)
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = f"{source_file}::p{page_number}::c{chunk_index}"
            display_texts.append(chunk_text)
            embed_texts_input.append(f"Documento: {title}\n\n{chunk_text}")
            metadatas.append(
                {
                    "source_file": source_file,
                    "page_number": page_number,
                    "used_ocr": page["used_ocr"],
                }
            )
            ids.append(chunk_id)

    return embed_texts_input, display_texts, metadatas, ids


def index_processed_document(json_path: Path) -> int:
    """Indexa (ou reindexa) um único documento já processado no Chroma.

    Remove chunks antigos desse arquivo antes de inserir os novos, para que
    reprocessar um documento não deixe chunks duplicados/obsoletos.
    """
    doc = _load_processed_document(json_path)
    source_file = doc["source_file"]
    collection = get_collection()

    # Remove chunks antigos deste arquivo (idempotência em reindexações).
    collection.delete(where={"source_file": source_file})

    embed_inputs, display_texts, metadatas, ids = _build_chunks_for_document(doc)
    if not display_texts:
        logger.warning("Nenhum texto para indexar em %s", source_file)
        return 0

    embeddings = embed_texts(embed_inputs)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=display_texts,
        metadatas=metadatas,
    )
    logger.info("Indexados %s chunks de %s", len(display_texts), source_file)
    return len(display_texts)


def index_all_processed() -> int:
    """Indexa todos os documentos processados encontrados em data/processed."""
    total = 0
    for json_path in sorted(config.PROCESSED_DIR.glob("*.json")):
        total += index_processed_document(json_path)
    return total
