FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Instala o torch em sua variante CPU-only antes do resto: a variante padrão
# do PyPI traz dependências de CUDA (GBs) desnecessárias em uma instância de
# nuvem pequena sem GPU. Isso reduz bastante o tamanho final da imagem.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Baixa o modelo de embeddings local durante o build (evita esperar/baixar
# no primeiro request em produção e permite rodar 100% offline depois).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY src ./src
COPY scripts ./scripts
COPY README.md .

RUN mkdir -p data/raw data/processed data/catalog data/chroma

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
