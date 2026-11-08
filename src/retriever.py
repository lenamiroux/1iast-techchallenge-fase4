"""
Recuperação semântica das avaliações.

Fluxo: pergunta -> embedding ("query: ...") -> busca no ChromaDB
(CANDIDATES_K candidatos, com filtros de metadados) -> limiar de
similaridade -> MMR (diversidade) -> TOP_K evidências.
"""
from dataclasses import dataclass

import numpy as np

from src import config
from src.indexing import get_collection, get_embedder


@dataclass
class Evidence:
    """Uma avaliação recuperada, com tudo o que é preciso para rastreá-la."""
    review_id: str
    text: str
    similarity: float
    metadata: dict


def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int,
        lambda_mult: float) -> list[int]:
    """
    Maximal Marginal Relevance.

    A cada passo escolhe o candidato que maximiza:
        λ * sim(candidato, pergunta) - (1 - λ) * max sim(candidato, já escolhidos)

    Muitas avaliações dizem quase a mesma coisa ("chegou no prazo"). Sem MMR,
    as 8 evidências enviadas ao LLM poderiam ser variações de um único tema;
    com MMR, cobrimos temas distintos, o que é essencial para perguntas como
    "quais são os PRINCIPAIS problemas?".
    """
    # Vetores normalizados: produto interno = similaridade cosseno
    relevance = doc_vecs @ query_vec
    candidates = list(range(len(doc_vecs)))
    selected: list[int] = []

    while candidates and len(selected) < top_k:
        if not selected:
            # 1ª escolha: simplesmente o mais relevante
            best = max(candidates, key=lambda i: relevance[i])
        else:
            chosen = doc_vecs[selected]
            best = max(
                candidates,
                key=lambda i: lambda_mult * relevance[i]
                - (1 - lambda_mult) * float(np.max(chosen @ doc_vecs[i])),
            )
        selected.append(best)
        candidates.remove(best)
    return selected


class ReviewRetriever:
    def __init__(self, embedder=None, collection=None):
        # Injeção de dependências: facilita os testes e permite reaproveitar
        # o modelo já carregado (carregar o modelo leva alguns segundos)
        self.embedder = embedder if embedder is not None else get_embedder()
        self.collection = collection if collection is not None else get_collection()
        if self.collection.count() == 0:
            raise RuntimeError("Base vetorial vazia. Rode: python -m src.indexing")

    def embed_query(self, question: str) -> np.ndarray:
        """Gera o embedding da pergunta com o prefixo exigido pelo e5."""
        return self.embedder.encode([f"query: {question}"], normalize_embeddings=True)[0]

    def search(self, question: str, where: dict | None = None, top_k: int | None = None,
               candidates_k: int | None = None, threshold: float | None = None,
               use_mmr: bool = True) -> list[Evidence]:
        # "is None" em vez de "or": permite passar threshold=0.0 de propósito
        top_k = config.TOP_K if top_k is None else top_k
        candidates_k = config.CANDIDATES_K if candidates_k is None else candidates_k
        threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold

        query_vec = self.embed_query(question)
        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=candidates_k,
            where=where,
            # embeddings são necessários para calcular o MMR
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = results["ids"][0]
        if not ids:
            return []  # nenhum documento satisfaz os filtros

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        sims = 1.0 - np.array(results["distances"][0])  # distância cosseno -> similaridade
        embs = np.array(results["embeddings"][0])

        # Guardrail 1: descarta candidatos pouco similares à pergunta
        keep = np.where(sims >= threshold)[0]
        if len(keep) == 0:
            return []

        if use_mmr:
            order = [keep[i] for i in mmr(query_vec, embs[keep], top_k, config.MMR_LAMBDA)]
        else:
            order = keep[:top_k]  # o Chroma já devolve ordenado por similaridade

        return [Evidence(ids[i], docs[i], float(sims[i]), metas[i]) for i in order]