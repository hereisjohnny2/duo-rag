"""Geração de resposta com o LLM local (Ollama), com base nos trechos
recuperados do RAG. O prompt instrui o modelo a responder apenas com base
no contexto fornecido e a citar a fonte (arquivo + página)."""
from __future__ import annotations

import ollama

from src import config
from src.rag.retriever import RetrievedChunk

_SYSTEM_PROMPT = (
    "Você é um assistente da firma de arquitetura Duo. Responda perguntas "
    "sobre os documentos internos da firma (ex.: escrituras de imóveis) "
    "usando APENAS as informações do CONTEXTO abaixo. Se a resposta não "
    "estiver no contexto, diga claramente que não encontrou essa "
    "informação nos documentos indexados — não invente dados. Sempre que "
    "possível, cite a fonte (nome do arquivo e número da página) das "
    "informações usadas na resposta. Responda em português."
)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Trecho {i} - fonte: {chunk.source_file}, página {chunk.page_number}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Gera a resposta final do RAG a partir da pergunta e dos chunks recuperados."""
    if not chunks:
        return (
            "Ainda não há documentos indexados (ou nenhum trecho relevante foi "
            "encontrado) para responder a essa pergunta. Adicione documentos e "
            "tente novamente."
        )

    context = _format_context(chunks)
    user_prompt = (
        f"CONTEXTO:\n{context}\n\n"
        f"PERGUNTA: {question}\n\n"
        "Responda com base apenas no contexto acima, citando a fonte "
        "(arquivo e página)."
    )

    response = ollama.chat(
        model=config.OLLAMA_LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


_CATALOG_SYSTEM_PROMPT = (
    "Você é um assistente da firma de arquitetura Duo. Você recebeu um "
    "CATÁLOGO com os metadados estruturados (cidade, endereço, "
    "proprietários, valor, data, matrícula) de TODOS os documentos "
    "indexados. Use o catálogo para responder perguntas de contagem, "
    "listagem ou agregação (ex.: quantos imóveis existem em determinada "
    "cidade). Conte/liste com base apenas nos dados do catálogo, cite os "
    "nomes dos arquivos usados como evidência, e não invente dados que não "
    "estejam nele. Se um campo estiver 'não informado', não o inclua nas "
    "contagens. Responda em português."
)


def generate_catalog_answer(question: str, catalog_text: str) -> str:
    """Gera a resposta a partir do catálogo completo de metadados (usado
    para perguntas de agregação/contagem, onde a busca por chunk falha)."""
    if not catalog_text.strip():
        return (
            "Ainda não há metadados extraídos dos documentos para responder "
            "perguntas de contagem/listagem. Rode a extração de metadados "
            "(ingestão) e tente novamente."
        )

    user_prompt = (
        f"CATÁLOGO DE DOCUMENTOS:\n{catalog_text}\n\n"
        f"PERGUNTA: {question}\n\n"
        "Responda com base apenas no catálogo acima, citando os arquivos "
        "usados como evidência."
    )

    response = ollama.chat(
        model=config.OLLAMA_LLM_MODEL,
        messages=[
            {"role": "system", "content": _CATALOG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]
