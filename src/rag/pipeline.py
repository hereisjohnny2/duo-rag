"""Orquestra a pipeline de RAG: retrieval + generation."""
from __future__ import annotations

from dataclasses import dataclass, field

from src import config
from src.rag import catalog
from src.rag.generator import generate_answer, generate_catalog_answer
from src.rag.retriever import RetrievedChunk, retrieve


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    used_catalog: bool = False


def ask(question: str, top_k: int = config.RETRIEVAL_TOP_K) -> RagAnswer:
    """Responde a uma pergunta usando o RAG (retrieval + geração local).

    Perguntas de agregação/contagem (ex.: "quantos imóveis em Rio das
    Ostras?") são detectadas e respondidas a partir do catálogo de
    metadados estruturados (todos os documentos de uma vez), pois a busca
    semântica por chunk não enxerga o conjunto completo de documentos.
    """
    if catalog.is_aggregate_question(question):
        entries = catalog.load_catalog()
        deterministic_count = catalog.count_properties_in_city(question, entries)
        if deterministic_count:
            return RagAnswer(
                answer=deterministic_count,
                sources=[],
                used_catalog=True,
            )
        catalog_text = catalog.format_catalog_for_prompt(entries)
        answer = generate_catalog_answer(question, catalog_text)
        return RagAnswer(answer=answer, sources=[], used_catalog=True)

    chunks = retrieve(question, top_k=top_k)
    answer = generate_answer(question, chunks)
    return RagAnswer(answer=answer, sources=chunks, used_catalog=False)
