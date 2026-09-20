"""Camada de abstração para o provedor de LLM e de embeddings.

Permite alternar entre execução 100% local (Ollama) e uma API de LLM
hospedada e compatível com a API da OpenAI (ex.: DeepSeek) sem alterar o
restante do código de ingestão/RAG. A escolha é feita via variáveis de
ambiente (``DUO_RAG_LLM_PROVIDER`` / ``DUO_RAG_EMBEDDING_PROVIDER``).

Isso existe porque instâncias de nuvem pequenas/baratas (ex.: o menor plano
do Lightsail) não têm RAM/CPU suficiente para rodar o Ollama com um modelo
como o llama3.1:8b. Nesse cenário, o LLM passa a ser uma API externa
(DeepSeek), mas os embeddings continuam sendo gerados localmente (modelo
pequeno via sentence-transformers), evitando depender de um provedor de
embeddings externo e mantendo o texto dos documentos fora de uma segunda
API sempre que possível.
"""
from __future__ import annotations

import logging

from src import config

logger = logging.getLogger(__name__)

_deepseek_client = None
_local_embedder = None


def _get_deepseek_client():
    global _deepseek_client
    if _deepseek_client is None:
        from openai import OpenAI

        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError(
                "DUO_RAG_LLM_PROVIDER=deepseek, mas DUO_RAG_DEEPSEEK_API_KEY "
                "não foi definida."
            )
        _deepseek_client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
    return _deepseek_client


def _get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Carregando modelo de embeddings local: %s", config.LOCAL_EMBEDDING_MODEL
        )
        _local_embedder = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    return _local_embedder


def chat_completion(messages: list[dict], json_mode: bool = False) -> str:
    """Envia uma conversa ao LLM configurado (Ollama ou DeepSeek) e retorna
    o texto da resposta."""
    if config.LLM_PROVIDER == "deepseek":
        client = _get_deepseek_client()
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    import ollama

    kwargs = {"format": "json"} if json_mode else {}
    response = ollama.chat(model=config.OLLAMA_LLM_MODEL, messages=messages, **kwargs)
    return response["message"]["content"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Gera embeddings para uma lista de textos usando o provedor configurado
    (modelo local via sentence-transformers, ou Ollama)."""
    if config.EMBEDDING_PROVIDER == "local":
        embedder = _get_local_embedder()
        vectors = embedder.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    import ollama

    embeddings: list[list[float]] = []
    for text in texts:
        response = ollama.embeddings(model=config.OLLAMA_EMBEDDING_MODEL, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings
