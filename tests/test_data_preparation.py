"""
Testes das funções puras de limpeza.

Não dependem do dataset: validam as regras de preparação de forma
rápida e reproduzível (rodam em qualquer máquina, até sem os CSVs).
"""
import numpy as np

from src.data_preparation import (clean_text, combine_title_message,
                                  is_informative, score_to_sentiment)


def test_clean_text_remove_quebras_espacos_e_pontuacao_repetida():
    # \r\n vira espaço, espaços duplos colapsam e "!!!" vira "!"
    assert clean_text("Muito  bom\r\nrecomendo!!!") == "Muito bom recomendo!"


def test_clean_text_preserva_acentos():
    # Não normalizamos acentos: o modelo de embeddings é multilíngue
    assert clean_text("Ótima qualidade, não recomendo") == "Ótima qualidade, não recomendo"


def test_clean_text_trata_nulos():
    # Valores ausentes do CSV chegam como NaN ou None
    assert clean_text(None) == ""
    assert clean_text(np.nan) == ""


def test_combine_title_message():
    assert combine_title_message("Ótimo", "Chegou antes do prazo") == "Ótimo. Chegou antes do prazo"
    # Só mensagem ou só título
    assert combine_title_message("", "Chegou rápido") == "Chegou rápido"
    assert combine_title_message("Recomendo", "") == "Recomendo"
    # Não duplica quando a mensagem já começa com o título
    assert combine_title_message("Bom", "Bom produto") == "Bom produto"


def test_is_informative():
    assert not is_informative("ok")            # curto demais
    assert not is_informative("...........")   # sem nenhuma letra
    assert is_informative("Produto chegou quebrado")


def test_score_to_sentiment():
    assert score_to_sentiment(1) == "negativo"
    assert score_to_sentiment(2) == "negativo"
    assert score_to_sentiment(3) == "neutro"
    assert score_to_sentiment(4) == "positivo"
    assert score_to_sentiment(5) == "positivo"