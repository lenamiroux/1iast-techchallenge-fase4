"""
Configurações centrais do projeto.

Todos os caminhos e hiperparâmetros ficam aqui para que:
  1. nenhum valor "mágico" fique espalhado pelo código;
  2. os experimentos possam ser ajustados pelo .env sem editar código;
  3. o README possa documentar um único ponto de configuração.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env (se existir) para os.environ
# override=True: o .env é a fonte da verdade, mesmo que o terminal
# tenha variáveis antigas carregadas (ex.: pelo VS Code)
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
# parents[1] = pasta raiz do projeto (este arquivo está em src/config.py)
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"              # CSVs originais do Kaggle
PROCESSED_DIR = ROOT_DIR / "data" / "processed"  # base limpa
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"       # índice persistente do Chroma
DOCUMENTS_PATH = PROCESSED_DIR / "reviews_documents.parquet"

# ---------------------------------------------------------------------------
# Preparação dos dados
# ---------------------------------------------------------------------------
# Textos com menos caracteres que isso ("ok", "bom") quase não carregam
# informação sobre o *porquê* da satisfação/insatisfação e poluem a busca.
MIN_CHARS = int(os.getenv("MIN_CHARS", 10))

# ---------------------------------------------------------------------------
# Embeddings e base vetorial
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
COLLECTION_NAME = "olist_reviews"

# ---------------------------------------------------------------------------
# Recuperação
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", 8))                     # evidências enviadas ao LLM
CANDIDATES_K = int(os.getenv("CANDIDATES_K", 40))      # candidatos antes do MMR
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.84))
MIN_EVIDENCES = int(os.getenv("MIN_EVIDENCES", 3))     # mínimo para gerar resposta
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", 0.7))       # 1 = só relevância; 0 = só diversidade

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "medium")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
