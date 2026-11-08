"""
Testes dos guardrails do pipeline RAG.

Usam "dublês" (FakeRetriever e FakeLLM) no lugar do ChromaDB e do modelo
de linguagem: os testes ficam rápidos, determinísticos e não dependem do
Ollama nem de chave de API. O que importa aqui é a LÓGICA de decisão.
"""
from src.prompts import build_context
from src.rag_pipeline import VoCRAGPipeline, extract_citations
from src.retriever import Evidence


class FakeRetriever:
    """Devolve sempre as mesmas evidências e guarda o último filtro recebido."""

    def __init__(self, evidences):
        self.evidences = evidences
        self.last_where = None

    def search(self, question, where=None, top_k=None, threshold=None):
        self.last_where = where
        return self.evidences


class FakeLLM:
    """Devolve uma resposta fixa e registra se foi chamado."""

    def __init__(self, answer):
        self.answer = answer
        self.called = False

    def generate(self, system, user):
        self.called = True
        return self.answer


def make_evidences(n):
    meta = {"review_score": 1, "category": "cama mesa banho", "year_month": "2018-01",
            "delivery_status": "atrasado", "delay_days": 3}
    return [Evidence(f"r{i}", f"Produto atrasou {i}", 0.9, meta) for i in range(n)]


# --- Guardrails ----------------------------------------------------------------
def test_sem_evidencias_nao_chama_llm():
    llm = FakeLLM("qualquer")
    result = VoCRAGPipeline(FakeRetriever([]), llm).ask("Qual é a capital da Austrália?")
    assert result.status == "sem_evidencias"
    assert not llm.called  # nada de gerar texto sem base nos dados


def test_poucas_evidencias_nao_chama_llm():
    llm = FakeLLM("qualquer")
    result = VoCRAGPipeline(FakeRetriever(make_evidences(2)), llm).ask("Pergunta")
    assert result.status == "evidencias_insuficientes"
    assert len(result.evidences) == 2  # as avaliações continuam visíveis ao usuário
    assert not llm.called


def test_llm_declara_evidencias_insuficientes():
    llm = FakeLLM("EVIDÊNCIAS INSUFICIENTES: a pergunta está fora do escopo.")
    result = VoCRAGPipeline(FakeRetriever(make_evidences(5)), llm).ask("Receita de bolo?")
    assert result.status == "evidencias_insuficientes"


def test_resposta_com_citacoes_validas():
    llm = FakeLLM("**Resumo:** atrasos frequentes [E1][E2].")
    result = VoCRAGPipeline(FakeRetriever(make_evidences(5)), llm).ask("Pergunta")
    assert result.status == "ok"
    assert result.cited_ids == [1, 2]


def test_citacao_inexistente_e_sinalizada():
    llm = FakeLLM("Atrasos frequentes [E1]. Outro ponto [E9].")
    result = VoCRAGPipeline(FakeRetriever(make_evidences(5)), llm).ask("Pergunta")
    assert result.invalid_citations == [9]
    assert result.status == "verificar_citacoes"


def test_resposta_sem_citacoes_e_sinalizada():
    llm = FakeLLM("Os clientes reclamam de atrasos.")
    result = VoCRAGPipeline(FakeRetriever(make_evidences(5)), llm).ask("Pergunta")
    assert result.status == "verificar_citacoes"


# --- Filtros ---------------------------------------------------------------------
def test_filtro_inferido_e_repassado_ao_retriever():
    retriever = FakeRetriever(make_evidences(5))
    VoCRAGPipeline(retriever, FakeLLM("ok [E1]")).ask("O que dizem os clientes insatisfeitos?")
    assert retriever.last_where == {"review_score": {"$lte": 2}}


def test_filtro_do_usuario_tem_prioridade():
    retriever = FakeRetriever(make_evidences(5))
    VoCRAGPipeline(retriever, FakeLLM("ok [E1]")).ask(
        "O que dizem os clientes insatisfeitos?", filters={"score_max": 1}
    )
    assert retriever.last_where == {"review_score": {"$lte": 1}}


# --- Funções auxiliares ------------------------------------------------------------
def test_extract_citations_formatos_variados():
    assert extract_citations("texto [E1, E3] e [E2]", 5) == ([1, 2, 3], [])


def test_build_context_numera_e_inclui_metadados():
    context = build_context(make_evidences(2))
    assert context.startswith("[E1]")
    assert "[E2]" in context
    assert "nota: 1/5" in context and "atrasado (3 dias)" in context