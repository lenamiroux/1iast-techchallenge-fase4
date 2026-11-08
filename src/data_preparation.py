"""
Construção da base de conhecimento a partir do dataset da Olist.

Decisões (documentadas no README), com base na inspeção dos dados:
- SELEÇÃO: avaliações (texto + nota) enriquecidas com metadados de pedidos
  (prazo de entrega), itens e produtos (categoria).
- DUPLICADOS: 814 review_id repetidos no CSV; mantemos a 1ª ocorrência.
- AUSENTES: 58,7% das avaliações não têm mensagem e 88,3% não têm título.
  Avaliações sem nenhum texto são descartadas (não há o que recuperar).
  Metadados ausentes recebem rótulos explícitos ("desconhecido",
  "sem categoria"), pois o Chroma não aceita None em metadados.
- LIMPEZA: removemos quebras de linha, espaços repetidos e pontuação
  exagerada. NÃO removemos acentos/stopwords nem aplicamos lowercase:
  o modelo de embeddings é multilíngue e se beneficia do texto natural.
- TEXTOS CURTOS: mensagens com menos de MIN_CHARS caracteres ("ok", "bom")
  não explicam o porquê da experiência e poluem a busca.
- DOCUMENTO: mensagens têm mediana de 53 caracteres (máx. 208), então
  1 avaliação = 1 documento (título + mensagem), sem chunking.

Execução:  python -m src.data_preparation
"""
import re

import numpy as np
import pandas as pd

from src import config

# Apenas os arquivos do Kaggle que a solução utiliza
RAW_FILES = {
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
}


# ---------------------------------------------------------------------------
# Funções puras de texto (sem dependência de arquivos: fáceis de testar)
# ---------------------------------------------------------------------------
def clean_text(text) -> str:
    """Normaliza um texto de avaliação, preservando acentos e pontuação útil."""
    # Valores ausentes (NaN/None) viram string vazia para simplificar o fluxo
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    # Quebras de linha e tabulações viram espaço ("Ótimo\r\nproduto")
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Colapsa espaços múltiplos em um só
    text = re.sub(r"\s+", " ", text)
    # "!!!!!" -> "!"  /  "....." -> "."  (reduz ruído sem perder o tom)
    text = re.sub(r"([!?.])\1{2,}", r"\1", text)
    return text.strip()


def combine_title_message(title: str, message: str) -> str:
    """Une título e mensagem em um único texto de documento."""
    if title and message:
        # Evita repetir quando a mensagem já começa com o título
        if message.lower().startswith(title.lower()):
            return message
        return f"{title}. {message}"
    return title or message


def is_informative(text: str, min_chars: int = config.MIN_CHARS) -> bool:
    """Mantém apenas textos com tamanho mínimo e ao menos uma letra."""
    return len(text) >= min_chars and re.search(r"[A-Za-zÀ-ÿ]", text) is not None


def score_to_sentiment(score: int) -> str:
    """Converte a nota (1 a 5) em rótulo de sentimento para filtros e análises."""
    if score <= 2:
        return "negativo"
    if score == 3:
        return "neutro"
    return "positivo"


# ---------------------------------------------------------------------------
# Carga e transformação
# ---------------------------------------------------------------------------
def load_raw_data() -> dict:
    """Lê os CSVs necessários, com mensagem clara caso algum arquivo falte."""
    data = {}
    for name, filename in RAW_FILES.items():
        path = config.RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}. Baixe o dataset (veja o README)."
            )
        data[name] = pd.read_csv(path)
    return data


def prepare_reviews(reviews: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicados, limpa os textos e descarta avaliações sem conteúdo."""
    stats = {"avaliacoes_brutas": len(reviews)}

    df = reviews.drop_duplicates(subset="review_id", keep="first").copy()
    stats["apos_remover_duplicados"] = len(df)

    df["title_clean"] = df["review_comment_title"].apply(clean_text)
    df["message_clean"] = df["review_comment_message"].apply(clean_text)
    df["text"] = [
        combine_title_message(t, m)
        for t, m in zip(df["title_clean"], df["message_clean"])
    ]

    stats["com_algum_texto"] = int((df["text"] != "").sum())
    df = df[df["text"].apply(is_informative)]
    stats["com_texto_informativo"] = len(df)
    return df, stats


def build_delivery_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Calcula o atraso de entrega (em dias) e um status categórico por pedido."""
    df = orders[[
        "order_id", "order_status", "order_purchase_timestamp",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ]].copy()
    for col in ["order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Positivo = entregue depois da data estimada; negativo = antes do prazo
    df["delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    # np.select avalia as condições em ordem: primeiro "não entregue"
    df["delivery_status"] = np.select(
        [df["order_delivered_customer_date"].isna(), df["delay_days"] > 0],
        ["nao_entregue", "atrasado"],
        default="no_prazo",
    )
    return df[["order_id", "order_status", "order_purchase_timestamp",
               "delay_days", "delivery_status"]]


def build_order_category(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Define UMA categoria por pedido: a mais frequente entre seus itens."""
    df = items[["order_id", "product_id"]].merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    )
    df["product_category_name"] = df["product_category_name"].fillna("sem_categoria")

    # Conta itens por (pedido, categoria) e fica com a categoria mais frequente.
    # É bem mais rápido do que groupby + lambda com mode() em ~100 mil pedidos.
    counts = (
        df.groupby(["order_id", "product_category_name"]).size()
        .reset_index(name="n")
        .sort_values(["order_id", "n"], ascending=[True, False])
        .drop_duplicates("order_id")
    )
    counts["category"] = counts["product_category_name"].str.replace("_", " ")
    return counts[["order_id", "category"]]


def build_documents(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Pipeline completo: CSVs brutos -> documentos limpos com metadados."""
    raw = load_raw_data()
    reviews, stats = prepare_reviews(raw["reviews"])
    delivery = build_delivery_features(raw["orders"])
    category = build_order_category(raw["items"], raw["products"])

    df = (reviews
          .merge(delivery, on="order_id", how="left")
          .merge(category, on="order_id", how="left"))

    # --- Tratamento de metadados ausentes (o Chroma não aceita None) ---
    df["category"] = df["category"].fillna("sem categoria")
    df["delivery_status"] = df["delivery_status"].fillna("desconhecido")
    df["order_status"] = df["order_status"].fillna("desconhecido")
    # Atraso ausente (pedido não entregue) vira 0; o delivery_status desambigua
    df["delay_days"] = df["delay_days"].fillna(0).astype(int)

    # --- Período: data da compra; se ausente, usa a data da avaliação ---
    review_date = pd.to_datetime(df["review_creation_date"], errors="coerce")
    period = df["order_purchase_timestamp"].fillna(review_date)
    df["review_date"] = review_date.dt.strftime("%Y-%m-%d").fillna("desconhecida")
    df["year_month"] = period.dt.strftime("%Y-%m").fillna("desconhecido")
    df["year"] = period.dt.year.fillna(0).astype(int)

    df["review_score"] = df["review_score"].astype(int)
    df["sentiment"] = df["review_score"].apply(score_to_sentiment)
    df["n_chars"] = df["text"].str.len()

    columns = ["review_id", "order_id", "text", "review_score", "sentiment",
               "category", "delivery_status", "delay_days", "order_status",
               "review_date", "year_month", "year", "n_chars"]
    docs = df[columns].reset_index(drop=True)
    stats["documentos_finais"] = len(docs)

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        docs.to_parquet(config.DOCUMENTS_PATH, index=False)
    return docs, stats


if __name__ == "__main__":
    docs, stats = build_documents()
    print("\n=== Estatísticas da preparação (use no README) ===")
    for key, value in stats.items():
        print(f"{key:>28}: {value:,}")
    print("\nDistribuição das notas:\n", docs["review_score"].value_counts().sort_index())
    print("\nSentimento:\n", docs["sentiment"].value_counts())
    print("\nStatus de entrega:\n", docs["delivery_status"].value_counts())
    print("\nTamanho dos textos (caracteres):\n", docs["n_chars"].describe().round(1))
    print(f"\nSalvo em: {config.DOCUMENTS_PATH}")