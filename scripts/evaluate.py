"""
Avaliação do pipeline RAG e calibração do limiar de similaridade.

Modo padrão (rápido, sem LLM): mede só a RECUPERAÇÃO. Para cada pergunta,
mostra a similaridade máxima/média dos candidatos e quantos passam do limiar.
Serve para justificar o SIMILARITY_THRESHOLD escolhido.

Modo --with-llm (lento, ~1 min por pergunta com Ollama): roda o pipeline
completo e registra o status final, as citações e a resposta. Serve para
verificar se cada pergunta teve o comportamento esperado:
  - perguntas DENTRO do escopo -> status "ok" (resposta com citações)
  - perguntas FORA do escopo   -> "sem_evidencias" ou "evidencias_insuficientes"

Saídas: reports/evaluation.csv e reports/evaluation.md (tabela para o README).

Execução:  python -m scripts.evaluate
           python -m scripts.evaluate --with-llm
"""
import argparse
import time

import pandas as pd

from src import config
from src.query_processing import build_where, infer_filters
from src.retriever import ReviewRetriever

# (pergunta, escopo esperado). Inclui as perguntas do enunciado, perguntas
# com metadados (desafio adicional) e perguntas fora do escopo.
QUESTIONS = [
    ("Quais são os principais problemas relatados pelos clientes em suas avaliações?", "dentro"),
    ("O que os clientes relatam sobre problemas relacionados à entrega?", "dentro"),
    ("Quais aspectos da experiência de compra são mais elogiados pelos clientes?", "dentro"),
    ("Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?", "dentro"),
    ("Quais avaliações sustentam a conclusão de que existem problemas recorrentes "
     "relacionados ao prazo de entrega?", "dentro"),
    ("Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?", "dentro"),
    ("Quais são os principais problemas relatados nas avaliações com nota baixa?", "dentro"),
    ("Qual é a capital da Austrália?", "fora"),
    ("Quem ganhou a Copa do Mundo de 2002?", "fora"),
    ("Qual foi o faturamento da Olist em 2018?", "fora"),
    ("Qual é a receita de bolo de cenoura?", "fora"),
]

OUT_OF_SCOPE_OK = {"sem_evidencias", "evidencias_insuficientes"}


def to_markdown(df: pd.DataFrame) -> str:
    """Gera uma tabela Markdown simples (sem depender de bibliotecas extras)."""
    header = "| " + " | ".join(df.columns) + " |"
    separator = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, separator, *rows])


def main():
    parser = argparse.ArgumentParser(description="Avalia recuperação e respostas do RAG")
    parser.add_argument("--with-llm", action="store_true",
                        help="roda o pipeline completo (lento) além da recuperação")
    args = parser.parse_args()

    retriever = ReviewRetriever()
    pipeline = None
    if args.with_llm:
        from src.rag_pipeline import VoCRAGPipeline
        pipeline = VoCRAGPipeline(retriever=retriever)  # reaproveita o modelo já carregado

    rows = []
    for question, scope in QUESTIONS:
        where = build_where(infer_filters(question))
        # threshold=0 e sem MMR: queremos ver as similaridades "cruas" dos candidatos
        raw = retriever.search(question, where=where, threshold=0.0, use_mmr=False)
        sims = [e.similarity for e in raw]
        row = {
            "pergunta": question[:70] + ("..." if len(question) > 70 else ""),
            "escopo": scope,
            "sim_max": round(max(sims, default=0.0), 3),
            "sim_media": round(sum(sims) / len(sims), 3) if sims else 0.0,
            "acima_limiar": sum(s >= config.SIMILARITY_THRESHOLD for s in sims),
        }

        if pipeline:
            start = time.time()
            result = pipeline.ask(question)
            expected_ok = (result.status == "ok") if scope == "dentro" \
                else (result.status in OUT_OF_SCOPE_OK)
            row.update({
                "status": result.status,
                "citadas": len(result.cited_ids),
                "esperado": "✅" if expected_ok else "❌",
                "tempo_s": round(time.time() - start, 1),
                "resposta": result.answer.replace("\n", " ")[:400],
            })
        rows.append(row)
        print(f"[{scope}] {row['sim_max']:.3f} {row.get('status', '')} | {question[:60]}")

    df = pd.DataFrame(rows)
    out_dir = config.ROOT_DIR / "reports"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "evaluation.csv", index=False)
    (out_dir / "evaluation.md").write_text(
        to_markdown(df.drop(columns=["resposta"], errors="ignore")) + "\n", encoding="utf-8"
    )

    print(f"\nLimiar atual: {config.SIMILARITY_THRESHOLD}")
    if pipeline:
        hits = (df["esperado"] == "✅").sum()
        print(f"Comportamento esperado em {hits} de {len(df)} perguntas")
    print(f"Resultados salvos em {out_dir}/evaluation.csv e evaluation.md")


if __name__ == "__main__":
    main()