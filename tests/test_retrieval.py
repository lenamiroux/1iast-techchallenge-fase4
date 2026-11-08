"""
Testes do processamento da pergunta e do MMR.

Não precisam do modelo de embeddings nem do ChromaDB: testam a lógica
de forma isolada, com vetores pequenos montados à mão.
"""
import numpy as np
import pytest

from src.query_processing import build_where, infer_filters, normalize_question
from src.retriever import mmr


# --- Processamento da pergunta ------------------------------------------------
def test_normalize_question_remove_espacos_extras():
    assert normalize_question("  O que   dizem\nsobre a entrega? ") == "O que dizem sobre a entrega?"


def test_normalize_question_rejeita_pergunta_vazia():
    with pytest.raises(ValueError):
        normalize_question("   ")


def test_infer_filters_insatisfeitos():
    assert infer_filters("Quais padrões nas avaliações de clientes insatisfeitos?") == {"score_max": 2}


def test_infer_filters_nota_baixa():
    assert infer_filters("Principais problemas nas avaliações com nota baixa") == {"score_max": 2}


def test_infer_filters_elogios():
    assert infer_filters("O que os clientes mais elogiam?") == {"score_min": 4}


def test_infer_filters_pergunta_neutra():
    assert infer_filters("O que os clientes relatam sobre a entrega?") == {}


# --- Tradução para a cláusula where do ChromaDB ----------------------------------
def test_build_where_sem_filtros():
    assert build_where({}) is None


def test_build_where_um_filtro():
    assert build_where({"score_max": 2}) == {"review_score": {"$lte": 2}}


def test_build_where_varios_filtros_usa_and():
    where = build_where({"score_max": 2, "delivery_status": "atrasado", "year": 2018})
    assert set(where) == {"$and"}
    assert {"delivery_status": {"$eq": "atrasado"}} in where["$and"]
    assert len(where["$and"]) == 3


# --- MMR -----------------------------------------------------------------------
def test_mmr_prefere_documento_diferente_a_duplicado():
    query = np.array([1.0, 0.0])
    docs = np.array([
        [0.95, 0.312],   # documento A
        [0.95, 0.312],   # duplicado de A
        [0.95, -0.312],  # tão relevante quanto A, mas com conteúdo diferente
    ])
    # Sem diversidade escolheríamos [0, 1]; o MMR troca o duplicado pelo diferente
    assert mmr(query, docs, top_k=2, lambda_mult=0.7) == [0, 2]


def test_mmr_respeita_top_k():
    rng = np.random.default_rng(42)
    docs = rng.normal(size=(10, 4))
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    assert len(mmr(docs[0], docs, top_k=3, lambda_mult=0.7)) == 3