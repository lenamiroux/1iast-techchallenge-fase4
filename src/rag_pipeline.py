"""
Pipeline RAG completo, com os guardrails de evidência.

Fluxo:
  1. processamento da pergunta (normalização + filtros de metadados)
  2. recuperação (embedding + ChromaDB + limiar + MMR)
  3. guardrail: nenhuma evidência        -> responde sem chamar o LLM
  4. guardrail: poucas evidências        -> mostra as avaliações, sem concluir
  5. construção do contexto e geração da resposta
  6. guardrail: o LLM declara evidência insuficiente
  7. rastreabilidade: valida as citações [E#] da resposta

Execução (terminal):
  python -m src.rag_pipeline "O que os clientes relatam sobre a entrega?"
"""
import argparse
import re
from dataclasses import dataclass, field

from src import config
from src.llm import get_llm
from src.prompts import INSUFFICIENT_PREFIX, SYSTEM_PROMPT, build_context, build_user_prompt
from src.query_processing import build_where, infer_filters, normalize_question
from src.retriever import Evidence, ReviewRetriever

NO_EVIDENCE_MSG = (
    f"{INSUFFICIENT_PREFIX} não encontrei avaliações suficientemente relacionadas à "
    "pergunta na base de conhecimento. Reformule a pergunta, amplie os filtros ou "
    "verifique se ela trata da experiência de compra dos clientes."
)


@dataclass
class RAGResponse:
    """Resultado de uma pergunta, com tudo o que a interface precisa exibir."""
    question: str
    answer: str
    # ok | sem_evidencias | evidencias_insuficientes | verificar_citacoes
    status: str
    evidences: list[Evidence] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    cited_ids: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)


def extract_citations(answer: str, n_evidences: int) -> tuple[list[int], list[int]]:
    """
    Extrai as citações da resposta ([E1], [E2, E5], [E3][E4]...) e separa as
    válidas das inexistentes (ex.: [E9] quando só há 8 evidências, um sinal
    de que o modelo inventou uma referência).
    """
    numbers = set()
    for group in re.findall(r"\[([^\]]+)\]", answer):
        numbers.update(int(n) for n in re.findall(r"E(\d+)", group))
    valid = sorted(n for n in numbers if 1 <= n <= n_evidences)
    invalid = sorted(n for n in numbers if n not in valid)
    return valid, invalid


class VoCRAGPipeline:
    def __init__(self, retriever=None, llm=None):
        # Injeção de dependências: nos testes passamos dublês (sem modelo nem LLM)
        self.retriever = retriever if retriever is not None else ReviewRetriever()
        self.llm = llm if llm is not None else get_llm()

    def ask(self, question: str, filters: dict | None = None, top_k: int | None = None,
            threshold: float | None = None, auto_filters: bool = True) -> RAGResponse:
        # 1) Processamento da pergunta
        question = normalize_question(question)
        inferred = infer_filters(question) if auto_filters else {}
        explicit = {k: v for k, v in (filters or {}).items() if v is not None}
        final_filters = {**inferred, **explicit}  # filtros escolhidos pelo usuário vencem

        # 2) Recuperação
        evidences = self.retriever.search(
            question, where=build_where(final_filters), top_k=top_k, threshold=threshold
        )

        # 3) Guardrail: nenhuma evidência relevante -> não chama o LLM (evita alucinação)
        if not evidences:
            return RAGResponse(question, NO_EVIDENCE_MSG, "sem_evidencias", [], final_filters)

        # 4) Guardrail: poucas evidências -> mostra as avaliações, sem tirar conclusões
        if len(evidences) < config.MIN_EVIDENCES:
            msg = (f"{INSUFFICIENT_PREFIX} apenas {len(evidences)} avaliação(ões) relevante(s) "
                   "encontrada(s), o que não basta para identificar padrões. As avaliações "
                   "estão listadas abaixo para consulta.")
            return RAGResponse(question, msg, "evidencias_insuficientes", evidences, final_filters)

        # 5) Construção do contexto e geração
        prompt = build_user_prompt(question, build_context(evidences))
        answer = self.llm.generate(SYSTEM_PROMPT, prompt)

        # 6) Guardrail: o próprio LLM julgou que as evidências não sustentam uma resposta
        if INSUFFICIENT_PREFIX in answer[:200]:
            return RAGResponse(question, answer, "evidencias_insuficientes", evidences, final_filters)

        # 7) Rastreabilidade: a resposta precisa citar evidências que existem
        cited, invalid = extract_citations(answer, len(evidences))
        status = "ok" if cited and not invalid else "verificar_citacoes"
        return RAGResponse(question, answer, status, evidences, final_filters, cited, invalid)


def main():
    parser = argparse.ArgumentParser(description="Pergunte à base de avaliações da Olist")
    parser.add_argument("question", help="pergunta em linguagem natural")
    parser.add_argument("--top-k", type=int, default=None, help="nº de evidências enviadas ao LLM")
    parser.add_argument("--no-auto-filters", action="store_true",
                        help="não inferir filtros de nota a partir da pergunta")
    args = parser.parse_args()

    result = VoCRAGPipeline().ask(args.question, top_k=args.top_k,
                                  auto_filters=not args.no_auto_filters)

    print(f"\nStatus: {result.status} | Filtros: {result.filters or 'nenhum'}\n")
    print(result.answer)
    print("\n--- Evidências (* = citada na resposta) ---")
    for i, ev in enumerate(result.evidences, start=1):
        mark = "*" if i in result.cited_ids else " "
        print(f"{mark}[E{i}] sim={ev.similarity:.3f} nota={ev.metadata['review_score']} "
              f"id={ev.review_id}\n     {ev.text[:150]}")


if __name__ == "__main__":
    main()