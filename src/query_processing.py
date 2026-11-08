"""
Processamento da pergunta: normalização + inferência de filtros.

Regras simples e transparentes (fáceis de explicar no vídeo):
- termos de insatisfação  -> filtra notas <= 2
- termos de elogio        -> filtra notas >= 4
Filtros de entrega e categoria NÃO são inferidos automaticamente: há
reclamações de atraso em pedidos marcados como "no prazo" (o cliente
reclama do prazo estimado longo, por exemplo). Esses filtros ficam a
cargo do usuário na interface.
"""
import re

# Expressões regulares em minúsculas; "notas? baixas?" cobre singular e plural
NEGATIVE_PATTERNS = [r"insatisf", r"notas? baixas?", r"reclama", r"negativ", r"detrator"]
POSITIVE_PATTERNS = [r"elogi", r"notas? altas?", r"positiv", r"satisfeit", r"promotor"]

# Perguntas muito longas não melhoram a busca e aumentam o custo do LLM
MAX_QUESTION_CHARS = 500


def normalize_question(question: str) -> str:
    """Remove espaços extras e limita o tamanho da pergunta."""
    question = re.sub(r"\s+", " ", question or "").strip()
    if not question:
        raise ValueError("A pergunta não pode ser vazia.")
    return question[:MAX_QUESTION_CHARS]


def infer_filters(question: str) -> dict:
    """Infere filtros de nota a partir de palavras-chave da pergunta."""
    q = question.lower()
    # Negativo é testado primeiro: "insatisfeito" também contém "satisfeit"
    if any(re.search(p, q) for p in NEGATIVE_PATTERNS):
        return {"score_max": 2}
    if any(re.search(p, q) for p in POSITIVE_PATTERNS):
        return {"score_min": 4}
    return {}


def build_where(filters: dict) -> dict | None:
    """Traduz o dicionário de filtros para a sintaxe 'where' do ChromaDB."""
    conditions = []
    if filters.get("score_min") is not None:
        conditions.append({"review_score": {"$gte": int(filters["score_min"])}})
    if filters.get("score_max") is not None:
        conditions.append({"review_score": {"$lte": int(filters["score_max"])}})
    if filters.get("category"):
        conditions.append({"category": {"$eq": filters["category"]}})
    if filters.get("delivery_status"):
        conditions.append({"delivery_status": {"$eq": filters["delivery_status"]}})
    if filters.get("year"):
        conditions.append({"year": {"$eq": int(filters["year"])}})

    if not conditions:
        return None  # o Chroma não aceita um where vazio
    # Uma condição vai direto; duas ou mais precisam ser combinadas com $and
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}