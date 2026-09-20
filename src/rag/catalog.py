"""Catálogo de metadados estruturados dos documentos (cidade, endereço,
proprietários, valor, data, matrícula etc.), usado para responder perguntas
de agregação/contagem que a busca semântica por chunk não resolve bem.
"""
from __future__ import annotations

import json
import re
import unicodedata

from src import config

# Palavras que indicam que a pergunta é sobre o conjunto de documentos como
# um todo (contagem, listagem, soma), e não sobre um trecho específico de um
# documento. Nesses casos, usamos o catálogo completo em vez de retrieval.
_AGGREGATE_KEYWORDS = [
    "quantos",
    "quantas",
    "quantidade",
    "todos os",
    "todas as",
    "liste",
    "listar",
    "lista de",
    "total de",
    "quais imoveis",
    "quais imóveis",
    "quais os imoveis",
    "quais os imóveis",
    "soma",
    "somatorio",
    "somatório",
]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_accents.lower()


def is_aggregate_question(query: str) -> bool:
    """Detecta heuristicamente se a pergunta é de agregação/contagem."""
    normalized = _normalize(query)
    return any(keyword in normalized for keyword in (_normalize(k) for k in _AGGREGATE_KEYWORDS))


def load_catalog() -> list[dict]:
    """Carrega os metadados estruturados de todos os documentos indexados."""
    entries = []
    for json_path in sorted(config.CATALOG_DIR.glob("*.json")):
        try:
            entries.append(json.loads(json_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def format_catalog_for_prompt(entries: list[dict]) -> str:
    """Formata o catálogo completo como texto compacto para o LLM.

    Cada documento vira um bloco curto com seus campos extraídos, para que
    o modelo possa contar/filtrar/listar sobre o conjunto inteiro, em vez de
    depender de trechos recuperados por similaridade semântica.
    """
    blocks = []
    for entry in entries:
        proprietarios = ", ".join(entry.get("proprietarios") or []) or "não informado"
        blocks.append(
            f"- Arquivo: {entry.get('source_file')}\n"
            f"  Tipo: {entry.get('tipo_documento') or 'não informado'}\n"
            f"  Cidade: {entry.get('cidade') or 'não informado'}\n"
            f"  Endereço: {entry.get('endereco') or 'não informado'}\n"
            f"  Proprietários: {proprietarios}\n"
            f"  Valor: {entry.get('valor') or 'não informado'}\n"
            f"  Data: {entry.get('data') or 'não informado'}\n"
            f"  Matrícula: {entry.get('matricula') or 'não informado'}"
        )
    return "\n".join(blocks)


def _normalized_value(value: str) -> str:
    value = _normalize(value)
    return " ".join(value.split())


def count_properties_in_city(query: str, entries: list[dict]) -> str | None:
    """Responde deterministicamente a contagens explícitas por cidade.

    O LLM continua sendo usado para perguntas mais abertas, mas não deve
    decidir sozinho uma contagem: erros de omissão ou dupla contagem são
    especialmente difíceis de perceber em documentos jurídicos.
    """
    normalized_query = _normalize(query)
    if "quant" not in normalized_query or "cidade" not in normalized_query:
        return None

    cities = {
        _normalized_value(str(entry["cidade"]))
        for entry in entries
        if entry.get("cidade")
    }
    mentioned_city = next(
        (city for city in sorted(cities, key=len, reverse=True) if city in normalized_query),
        None,
    )
    if not mentioned_city:
        return None

    matching = [
        entry
        for entry in entries
        if entry.get("cidade")
        and mentioned_city in _normalized_value(str(entry["cidade"]))
        and entry.get("endereco")
    ]
    distinct_addresses = {
        _normalized_value(str(entry["endereco"])) for entry in matching
    }
    files = ", ".join(entry["source_file"] for entry in matching)
    return (
        f"Encontrei {len(matching)} documento(s) com imóvel/endereço "
        f"registrado em {mentioned_city.title()} e "
        f"{len(distinct_addresses)} endereço(s) distinto(s).\n\n"
        f"Documentos: {files}.\n\n"
        "A contagem de documentos pode ser maior que a de endereços porque "
        "um mesmo imóvel pode aparecer em escritura, certidão ou outro "
        "documento relacionado."
    )
