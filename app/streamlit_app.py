"""
Interface de demonstração do Voice of Customer Intelligence.

Execução:  streamlit run app/streamlit_app.py

A interface permite:
- fazer perguntas em linguagem natural (ou escolher um exemplo);
- aplicar filtros de metadados (nota, categoria, entrega, ano);
- ajustar top-k e limiar de similaridade (útil para demonstrar o guardrail);
- ver a resposta, o status do guardrail e TODAS as evidências recuperadas,
  com destaque para as que foram citadas na resposta.
"""
import sys
from pathlib import Path

# O Streamlit executa este arquivo a partir de app/; adicionamos a raiz do
# projeto ao caminho de importação para que "from src import ..." funcione
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.rag_pipeline import VoCRAGPipeline  # noqa: E402

st.set_page_config(page_title="VoC Intelligence – Olist", page_icon="🛒", layout="wide")

# Perguntas do enunciado + uma fora do escopo para demonstrar o guardrail
EXAMPLES = [
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes "
    "relacionados ao prazo de entrega?",
    "Qual é a receita de bolo de cenoura?",
]

# status do pipeline -> (tipo de alerta do Streamlit, texto exibido)
STATUS_LABEL = {
    "ok": ("success", "✅ Resposta fundamentada em evidências"),
    "verificar_citacoes": ("warning", "⚠️ Resposta gerada, mas confira as citações"),
    "evidencias_insuficientes": ("warning", "⚠️ Evidências insuficientes para concluir"),
    "sem_evidencias": ("error", "⛔ Nenhuma avaliação relevante encontrada"),
}


@st.cache_resource(show_spinner="Carregando modelo de embeddings e base vetorial...")
def load_pipeline() -> VoCRAGPipeline:
    """Carrega o pipeline uma única vez (o modelo leva alguns segundos para subir)."""
    return VoCRAGPipeline()


@st.cache_data
def load_categories() -> list[str]:
    """Lista as categorias existentes na base, para o filtro da barra lateral."""
    df = pd.read_parquet(config.DOCUMENTS_PATH, columns=["category"])
    return sorted(df["category"].unique())


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title("🛒 Voice of Customer Intelligence")
st.caption("RAG sobre as avaliações do Brazilian E-Commerce Public Dataset (Olist)")

# ---------------------------------------------------------------------------
# Barra lateral: filtros e parâmetros
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filtros por metadados")
    score_range = st.slider("Nota da avaliação", 1, 5, (1, 5))
    category = st.selectbox("Categoria do produto", ["Todas"] + load_categories())
    delivery = st.selectbox("Status da entrega", ["Todos", "atrasado", "no_prazo", "nao_entregue"])
    year = st.selectbox("Ano da compra", ["Todos", 2016, 2017, 2018])

    st.header("Parâmetros de recuperação")
    top_k = st.slider("Evidências enviadas ao LLM (top-k)", 3, 15, config.TOP_K)
    threshold = st.slider("Similaridade mínima", 0.75, 0.95,
                          config.SIMILARITY_THRESHOLD, 0.01)
    auto_filters = st.checkbox("Inferir filtros a partir da pergunta", value=True,
                               help="Ex.: 'insatisfeitos' filtra notas ≤ 2; 'elogiam' filtra notas ≥ 4")

    st.divider()
    model_name = config.OLLAMA_MODEL if config.LLM_PROVIDER == "ollama" else config.ANTHROPIC_MODEL
    st.caption(f"LLM: {config.LLM_PROVIDER} ({model_name})")
    st.caption(f"Embeddings: {config.EMBEDDING_MODEL}")

# ---------------------------------------------------------------------------
# Pergunta
# ---------------------------------------------------------------------------
example = st.selectbox("Exemplos de perguntas", ["(escreva a sua abaixo)"] + EXAMPLES)
default_text = "" if example.startswith("(") else example
question = st.text_area("Sua pergunta", value=default_text, height=90)

if st.button("Perguntar", type="primary", disabled=not question.strip()):
    # Faixa completa de notas = sem filtro (deixa a inferência automática agir)
    full_range = score_range == (1, 5)
    filters = {
        "score_min": None if full_range else score_range[0],
        "score_max": None if full_range else score_range[1],
        "category": None if category == "Todas" else category,
        "delivery_status": None if delivery == "Todos" else delivery,
        "year": None if year == "Todos" else year,
    }

    try:
        pipeline = load_pipeline()
        with st.spinner("Recuperando avaliações e gerando a resposta (pode levar ~1 min)..."):
            result = pipeline.ask(question, filters=filters, top_k=top_k,
                                  threshold=threshold, auto_filters=auto_filters)
    except Exception as exc:  # erros de configuração/Ollama/API mostrados de forma amigável
        st.error(f"Erro: {exc}")
        st.stop()

    # --- Resposta ---------------------------------------------------------------
    kind, label = STATUS_LABEL[result.status]
    getattr(st, kind)(label)
    applied = {k: v for k, v in result.filters.items() if v is not None}
    st.caption(f"Filtros aplicados: {applied or 'nenhum'}")
    st.markdown(result.answer)
    if result.invalid_citations:
        st.warning(f"Citações inexistentes na resposta: {result.invalid_citations}")

    # --- Evidências (rastreabilidade) ---------------------------------------------
    st.subheader(f"Evidências recuperadas ({len(result.evidences)})")
    if result.evidences:
        st.caption("✅ = citada na resposta · ordem definida pelo MMR (relevância + diversidade)")
    for i, ev in enumerate(result.evidences, start=1):
        meta = ev.metadata
        cited = i in result.cited_ids
        title = (f"{'✅' if cited else '▫️'} [E{i}] nota {meta['review_score']} · "
                 f"{meta['category']} · {meta['delivery_status']} · "
                 f"similaridade {ev.similarity:.3f}")
        with st.expander(title, expanded=cited):
            st.write(ev.text)
            st.caption(f"review_id: {ev.review_id} · pedido: {meta['order_id']} · "
                       f"compra: {meta['year_month']}")