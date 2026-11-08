"""
Criação dos embeddings e indexação no ChromaDB.

Decisões técnicas:
- Modelo intfloat/multilingual-e5-small: multilíngue (bom em português),
  384 dimensões, leve o suficiente para rodar em CPU/Apple Silicon.
- O e5 foi treinado para busca ASSIMÉTRICA: documentos recebem o prefixo
  "passage: " e perguntas o prefixo "query: ". Sem isso a qualidade cai.
- normalize_embeddings=True: vetores com norma 1, então produto interno
  = similaridade cosseno (usado também no MMR do retriever).
- ChromaDB persistente com espaço "cosine": distância = 1 - similaridade.
  Não precisa de servidor e permite filtrar por metadados (cláusula where).
- id do vetor = review_id: cada evidência é rastreável até a avaliação.

Execução:  python -m src.indexing --limit 2000   (amostra rápida para testar)
           python -m src.indexing                (base completa)
"""
import argparse

import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src import config

# Metadados guardados junto de cada vetor: permitem filtros (nota, categoria,
# entrega, período) e mostram ao usuário de onde veio cada evidência
METADATA_COLUMNS = ["review_id", "order_id", "review_score", "sentiment", "category",
                    "delivery_status", "delay_days", "order_status",
                    "review_date", "year_month", "year"]


def get_embedder() -> SentenceTransformer:
    """Carrega o modelo de embeddings (baixado do Hugging Face na 1ª execução)."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def get_collection(reset: bool = False):
    """Abre (ou cria) a coleção persistente no disco."""
    client = chromadb.PersistentClient(
        path=str(config.VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    if reset:
        # Reindexação do zero: evita documentos duplicados ou desatualizados
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass  # a coleção ainda não existia
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _to_native(record: dict) -> dict:
    """Converte tipos numpy (int64, float64) em tipos Python, exigidos pelo Chroma."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}


def build_index(batch_size: int = 256, limit: int | None = None) -> int:
    df = pd.read_parquet(config.DOCUMENTS_PATH)
    if limit:
        # Amostra estratificada pela nota: mantém a proporção de cada nota
        # (uma amostra simples poderia sub-representar as notas raras)
        df = df.groupby("review_score").sample(
            frac=min(1.0, limit / len(df)), random_state=42
        )

    model = get_embedder()
    collection = get_collection(reset=True)

    # Lotes: controlam o uso de memória e respeitam o limite de inserção do Chroma
    for start in tqdm(range(0, len(df), batch_size), desc="Indexando"):
        batch = df.iloc[start:start + batch_size]
        passages = [f"passage: {text}" for text in batch["text"]]
        embeddings = model.encode(passages, normalize_embeddings=True,
                                  show_progress_bar=False)
        collection.add(
            ids=batch["review_id"].tolist(),
            documents=batch["text"].tolist(),
            embeddings=embeddings.tolist(),
            metadatas=[_to_native(r) for r in batch[METADATA_COLUMNS].to_dict("records")],
        )
    return collection.count()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa as avaliações no ChromaDB")
    parser.add_argument("--limit", type=int, default=None,
                        help="indexa só uma amostra (útil para testes rápidos)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    total = build_index(batch_size=args.batch_size, limit=args.limit)
    print(f"Coleção '{config.COLLECTION_NAME}' com {total:,} documentos em {config.VECTORSTORE_DIR}")