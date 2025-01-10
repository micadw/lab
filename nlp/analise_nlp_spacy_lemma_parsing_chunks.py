''' rodar direto no prompt se necessário
pip install spacy==3.5.0 pydantic==1.10.7
python -m spacy download pt_core_news_sm
'''

#%%
"""
Script de Processamento de Texto com spaCy

Este script utiliza o spaCy para processar um arquivo de texto em português e gerar análises linguísticas diversas.
As funcionalidades incluem:

1. Extração de tokens com informações detalhadas:
   - Texto original
   - Lema (forma base da palavra)
   - POS (parte do discurso)
   - Tag (etiqueta detalhada da parte do discurso)
   - Dep (função de dependência sintática)
   - Shape (formato do token)
   - Verificação se é alfabético
   - Verificação se é uma stopword

2. Identificação de noun chunks (grupos nominais).
3. Reconhecimento de entidades nomeadas (NER).
4. Visualização de dependências sintáticas em formato HTML.

Arquivos de saída:
- "analise_tokens.csv": Informações detalhadas sobre tokens.
- "noun_chunks.txt": Lista de grupos nominais extraídos.
- "entidades_nomeadas.txt": Lista de entidades nomeadas identificadas.
- "visualizacao_dependencias.html": Arquivo HTML com a visualização das dependências sintáticas.

Requisitos:
- Python 3.6 ou superior
- Biblioteca spaCy instalada
- Modelo de linguagem em português do spaCy (pt_core_news_sm)

Instalação do spaCy e do modelo:
- Instalar o spaCy: `pip install spacy`
- Baixar o modelo: `python -m spacy download pt_core_news_sm`

Instalações adicionais:
- Pandas (para manipulação de dados): `pip install pandas`

Como usar:
1. Execute o script.
2. Insira o caminho do arquivo de texto a ser analisado quando solicitado.
3. Os resultados serão salvos na mesma pasta do script.
"""

import spacy
from spacy import displacy
import os
import pandas as pd
import subprocess
import sys

def ensure_dependencies():
    """Certifica-se de que todas as dependências necessárias estão instaladas."""
    try:
        import spacy
        import pandas
        spacy.load("pt_core_news_sm")
    except ImportError:
        print("Instalando dependências...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy", "pandas"])
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "pt_core_news_sm"])
    except OSError:
        print("Baixando modelo 'pt_core_news_sm'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "pt_core_news_sm"])

def process_text_file(file_path):
    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        print("O arquivo especificado não foi encontrado.")
        return

    # Carregar o modelo do spaCy em português
    nlp = spacy.load("pt_core_news_sm")

    # Ler o conteúdo do arquivo
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Processar o texto com o spaCy
    doc = nlp(text)

    # Coletar análises
    analyses = {
        "Texto": [],
        "Lema": [],
        "POS": [],
        "Tag": [],
        "Dep": [],
        "Shape": [],
        "Alfabético": [],
        "Stopword": [],
    }

    for token in doc:
        analyses["Texto"].append(token.text)
        analyses["Lema"].append(token.lemma_)
        analyses["POS"].append(token.pos_)
        analyses["Tag"].append(token.tag_)
        analyses["Dep"].append(token.dep_)
        analyses["Shape"].append(token.shape_)
        analyses["Alfabético"].append(token.is_alpha)
        analyses["Stopword"].append(token.is_stop)

    # Criar um DataFrame para exportar
    df = pd.DataFrame(analyses)
    df.to_csv("analise_tokens.csv", index=False, encoding="utf-8")

    # Coletar noun chunks
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    with open("noun_chunks.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(noun_chunks))

    # Coletar entidades nomeadas
    named_entities = []
    for ent in doc.ents:
        named_entities.append(f"{ent.text} ({ent.label_})")

    with open("entidades_nomeadas.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(named_entities))

    # Visualização de dependências
    html = displacy.render(doc, style="dep", page=True)
    with open("visualizacao_dependencias.html", "w", encoding="utf-8") as file:
        file.write(html)

    print("Análise concluída! Os resultados foram salvos como arquivos.")

# Exemplo de uso
if __name__ == "__main__":
    ensure_dependencies()
    caminho_arquivo = input("Digite o caminho do arquivo de texto: ")
    process_text_file(caminho_arquivo)
