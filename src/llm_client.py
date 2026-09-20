"""Camada de abstração para o provedor de LLM e de embeddings.

Permite alternar entre execução 100% local (Ollama) e uma API de LLM
hospedada e compatível com a API da OpenAI (DeepSeek ou Gemini) sem
alterar o restante do código de ingestão/RAG. A escolha é feita via
variáveis de ambiente (``DUO_RAG_LLM_PROVIDER`` / ``DUO_RAG_EMBEDDING_PROVIDER``).

Isso existe porque instâncias de nuvem pequenas/baratas (ex.: o menor plano
do Lightsail) não têm RAM/CPU suficiente para rodar o Ollama com um modelo
como o llama3.1:8b. Nesse cenário, o LLM passa a ser uma API externa
(DeepSeek ou o free tier do Gemini), mas os embeddings continuam sendo
gerados localmente por padrão (modelo pequeno via sentence-transformers),
evitando depender de cota extra de um provedor externo e mantendo o texto
dos documentos fora de uma segunda API sempre que possível.
"""
from __future__ import annotations

import logging

from src import config

logger = logging.getLogger(__name__)

# Configuração de cada provedor compatível com a API da OpenAI: chave, URL
# base e nome do modelo de chat.
_OPENAI_COMPATIBLE_PROVIDERS = {
    "deepseek": lambda: (
        config.DEEPSEEK_API_KEY,
        config.DEEPSEEK_BASE_URL,
        config.DEEPSEEK_MODEL,
    ),
    "gemini": lambda: (
        config.GEMINI_API_KEY,
        config.GEMINI_BASE_URL,
        config.GEMINI_MODEL,
    ),
}

_openai_clients: dict[str, object] = {}
_local_embedder = None


def _get_openai_compatible_client(provider: str):
    if provider not in _openai_clients:
        from openai import OpenAI

        api_key, base_url, _ = _OPENAI_COMPATIBLE_PROVIDERS[provider]()
        if not api_key:
            env_var = f"DUO_RAG_{provider.upper()}_API_KEY"
            raise RuntimeError(
                f"DUO_RAG_LLM_PROVIDER={provider}, mas {env_var} não foi definida."
            )
        _openai_clients[provider] = OpenAI(api_key=api_key, base_url=base_url)
    return _openai_clients[provider]


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
    """Envia uma conversa ao LLM configurado (Ollama, DeepSeek ou Gemini) e
    retorna o texto da resposta."""
    provider = config.LLM_PROVIDER
    if provider in _OPENAI_COMPATIBLE_PROVIDERS:
        client = _get_openai_compatible_client(provider)
        _, _, model = _OPENAI_COMPATIBLE_PROVIDERS[provider]()
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=model,
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
    (modelo local via sentence-transformers, Gemini ou Ollama)."""
    provider = config.EMBEDDING_PROVIDER

    if provider == "local":
        embedder = _get_local_embedder()
        vectors = embedder.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    if provider == "gemini":
        client = _get_openai_compatible_client("gemini")
        response = client.embeddings.create(
            model=config.GEMINI_EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    import ollama

    embeddings: list[list[float]] = []
    for text in texts:
        response = ollama.embeddings(model=config.OLLAMA_EMBEDDING_MODEL, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings
