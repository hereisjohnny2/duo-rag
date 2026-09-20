"""Interface Streamlit do Duo RAG: chat para consultar os documentos e
upload de novos documentos (com processamento/indexação incremental)."""
from __future__ import annotations

import sys
from pathlib import Path

# Garante que "src" seja importável tanto rodando via `streamlit run` quanto
# a partir do executável empacotado com PyInstaller.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from datetime import datetime

import streamlit as st

from src import config
from src.ingestion.chunk_and_index import index_processed_document
from src.ingestion.extract import extract_and_save
from src.ingestion.metadata_extractor import extract_and_save_metadata
from src.rag.pipeline import ask

st.set_page_config(page_title="Duo RAG", page_icon="🏛️", layout="wide")


def _list_processed_documents() -> list[dict]:
    docs = []
    for json_path in sorted(config.PROCESSED_DIR.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        docs.append(data)
    return docs


def _load_catalog_entry(stem: str) -> dict | None:
    catalog_path = config.CATALOG_DIR / f"{stem}.json"
    if not catalog_path.exists():
        return None
    try:
        return json.loads(catalog_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _process_uploaded_file(uploaded_file) -> None:
    dest_path = config.RAW_DIR / uploaded_file.name
    dest_path.write_bytes(uploaded_file.getvalue())

    with st.spinner(f"Extraindo texto de {uploaded_file.name}..."):
        processed_path = extract_and_save(dest_path, force=True)

    with st.spinner(f"Indexando {uploaded_file.name} no RAG..."):
        num_chunks = index_processed_document(processed_path)

    with st.spinner(f"Extraindo metadados (cidade, proprietários, valor...) de {uploaded_file.name}..."):
        extract_and_save_metadata(processed_path, force=True)

    st.success(f"'{uploaded_file.name}' processado e indexado ({num_chunks} chunks).")


def render_chat_tab() -> None:
    st.subheader("Converse com os documentos da Duo")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("sources"):
                with st.expander("Fontes"):
                    for source in entry["sources"]:
                        st.markdown(
                            f"- **{source.source_file}**, página {source.page_number}"
                            f"{' (via OCR)' if source.used_ocr else ''}"
                        )

    question = st.chat_input("Pergunte algo sobre os documentos indexados...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Consultando os documentos..."):
                result = ask(question)
            st.markdown(result.answer)
            if result.sources:
                with st.expander("Fontes"):
                    for source in result.sources:
                        st.markdown(
                            f"- **{source.source_file}**, página {source.page_number}"
                            f"{' (via OCR)' if source.used_ocr else ''}"
                        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
            }
        )


def render_upload_tab() -> None:
    st.subheader("Adicionar novos documentos")
    st.caption(
        "Envie PDFs de escrituras (ou outros documentos da firma). Cada arquivo é "
        "extraído automaticamente (com OCR se for um scan) e indexado no RAG."
    )

    uploaded_files = st.file_uploader(
        "Selecione um ou mais PDFs", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Processar e indexar", type="primary"):
        for uploaded_file in uploaded_files:
            _process_uploaded_file(uploaded_file)


def render_documents_tab() -> None:
    st.subheader("Documentos indexados")
    docs = _list_processed_documents()
    if not docs:
        st.info("Nenhum documento processado ainda.")
        return

    for doc in docs:
        processed_at = doc.get("processed_at", "")
        try:
            processed_at = datetime.fromisoformat(processed_at).strftime(
                "%d/%m/%Y %H:%M"
            )
        except ValueError:
            pass
        with st.expander(f"📄 {doc['source_file']} ({doc.get('num_pages', 0)} páginas)"):
            st.write(f"Processado em: {processed_at}")
            st.write(f"OCR utilizado: {'Sim' if doc.get('used_ocr') else 'Não'}")

            catalog_entry = _load_catalog_entry(Path(doc["source_file"]).stem)
            if catalog_entry:
                st.markdown("**Metadados extraídos:**")
                st.write(f"Tipo: {catalog_entry.get('tipo_documento') or 'não informado'}")
                st.write(f"Cidade: {catalog_entry.get('cidade') or 'não informado'}")
                st.write(f"Endereço: {catalog_entry.get('endereco') or 'não informado'}")
                proprietarios = catalog_entry.get("proprietarios") or []
                st.write(f"Proprietários: {', '.join(proprietarios) or 'não informado'}")
                st.write(f"Valor: {catalog_entry.get('valor') or 'não informado'}")
                st.write(f"Data: {catalog_entry.get('data') or 'não informado'}")
                st.write(f"Matrícula: {catalog_entry.get('matricula') or 'não informado'}")
            else:
                st.caption("Metadados ainda não extraídos para este documento.")


def main() -> None:
    st.title("🏛️ Duo RAG — Documentos da firma")
    st.caption(
        "Assistente local (100% offline) para consultar documentos da Duo, "
        "como escrituras de imóveis."
    )

    tab_chat, tab_upload, tab_docs = st.tabs(
        ["💬 Consultar", "📤 Adicionar documentos", "📚 Documentos indexados"]
    )
    with tab_chat:
        render_chat_tab()
    with tab_upload:
        render_upload_tab()
    with tab_docs:
        render_documents_tab()


if __name__ == "__main__":
    main()
