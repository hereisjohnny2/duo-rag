"""Retrieval: busca os chunks mais relevantes no ChromaDB para uma pergunta."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.ingestion.chunk_and_index import embed_texts, get_collection


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    used_ocr: bool
    distance: float


def _normalize(text: str) -> str:
    """Remove acentos e baixa a caixa, para comparação tolerante a acentos."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_accents.lower()


def _known_source_files() -> list[str]:
    return [p.name.replace(".json", ".pdf") for p in config.PROCESSED_DIR.glob("*.json")]


def _find_referenced_documents(query: str) -> list[str]:
    """Detecta se a pergunta menciona o nome de algum documento conhecido.

    Compara os "tokens" do nome de cada arquivo (stem, sem extensão) com a
    pergunta normalizada (sem acentos, minúscula). Isso é importante porque
    o usuário costuma se referir ao documento pelo nome do arquivo (ex.:
    "Escritura Marileira"), que muitas vezes não aparece no corpo do texto
    extraído/OCR'd do próprio documento.
    """
    normalized_query = _normalize(query)
    matches = []
    for source_file in _known_source_files():
        stem = Path(source_file).stem
        tokens = [t for t in re.split(r"[\s_\-]+", stem) if len(t) > 2]
        if not tokens:
            continue
        normalized_tokens = [_normalize(t) for t in tokens]
        if all(token in normalized_query for token in normalized_tokens):
            matches.append(source_file)
    return matches


def retrieve(query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
    """Retorna os ``top_k`` chunks mais similares à pergunta do usuário.

    Se a pergunta citar claramente o nome de um documento conhecido (ex.:
    "Escritura Marileira"), a busca é restrita a esse(s) documento(s), pois
    a busca puramente semântica costuma falhar nesse caso (o nome do
    arquivo raramente aparece no corpo do texto extraído).
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]
    referenced_files = _find_referenced_documents(query)

    # Quando a pergunta cita um documento específico, a busca já fica restrita
    # a ele; nesse caso vale a pena trazer mais trechos (o espaço de busca é
    # muito menor), aumentando a chance da informação certa entrar no contexto.
    effective_top_k = top_k * 3 if referenced_files else top_k
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(effective_top_k, collection.count()),
    }
    if referenced_files:
        query_kwargs["where"] = {"source_file": {"$in": referenced_files}}

    results = collection.query(**query_kwargs)

    chunks: list[RetrievedChunk] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                text=text,
                source_file=metadata.get("source_file", "desconhecido"),
                page_number=metadata.get("page_number", 0),
                used_ocr=bool(metadata.get("used_ocr", False)),
                distance=distance,
            )
        )
    return chunks

