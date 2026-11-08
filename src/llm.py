"""
Camada de LLM plugável: Ollama (local, gratuito) ou Claude (API da Anthropic).

As duas classes expõem o mesmo método generate(system, user) -> str, então
o restante do pipeline não precisa saber qual modelo está em uso. A escolha
é feita pela variável LLM_PROVIDER no .env.
"""
import requests

from src import config
from src.prompts import INSUFFICIENT_PREFIX


class OllamaLLM:
    """Modelo local servido pelo Ollama (https://ollama.com)."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_URL

    def generate(self, system: str, user: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,  # resposta completa de uma vez
                    # Temperatura baixa: menos criatividade, mais fidelidade ao contexto
                    "options": {"temperature": 0.1},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=300,  # modelos locais podem levar alguns segundos por resposta
            )
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Não foi possível conectar ao Ollama. Rode: brew services start ollama"
            ) from exc
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Erro do Ollama ({exc.response.status_code}). "
                f"O modelo foi baixado? Rode: ollama pull {self.model}"
            ) from exc
        return response.json()["message"]["content"].strip()


class AnthropicLLM:
    """Claude via API da Anthropic (requer ANTHROPIC_API_KEY no .env)."""

    def __init__(self, model: str | None = None):
        import anthropic  # import tardio: quem usa Ollama não precisa de chave

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente
        self.model = model or config.ANTHROPIC_MODEL

    def generate(self, system: str, user: str) -> str:
        a = self._anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                # "effort" controla a profundidade do raciocínio (e o custo)
                output_config={"effort": config.ANTHROPIC_EFFORT},
                # Se o modelo recusar por política, a API tenta outro modelo
                # automaticamente dentro da mesma chamada
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
            )
        except a.AuthenticationError as exc:
            raise RuntimeError("ANTHROPIC_API_KEY ausente ou inválida (verifique o .env).") from exc
        except a.RateLimitError as exc:
            raise RuntimeError("Limite de requisições atingido; aguarde e tente novamente.") from exc
        except a.APIStatusError as exc:
            raise RuntimeError(f"Erro na API da Anthropic (HTTP {exc.status_code}).") from exc
        except a.APIConnectionError as exc:
            raise RuntimeError("Falha de conexão com a API da Anthropic.") from exc

        if response.stop_reason == "refusal":
            return f"{INSUFFICIENT_PREFIX} o modelo não gerou resposta para esta pergunta."
        # A resposta é uma lista de blocos; ficamos apenas com os de texto
        return "".join(b.text for b in response.content if b.type == "text").strip()


def get_llm(provider: str | None = None):
    """Devolve o cliente de LLM configurado em LLM_PROVIDER."""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "ollama":
        return OllamaLLM()
    if provider == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"LLM_PROVIDER inválido: {provider!r} (use 'ollama' ou 'anthropic')")