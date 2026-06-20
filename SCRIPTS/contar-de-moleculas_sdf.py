import sys

def contar_moleculas_sdf(caminho_arquivo):
    """
    Conta o número de moléculas em um arquivo SDF procurando pelo delimitador '$$$$'.

    Argumentos:
        caminho_arquivo (str): O caminho para o arquivo SDF.

    Retorna:
        int: O número de moléculas encontradas, ou None se ocorrer um erro ao ler o arquivo.
    """
    contador_moleculas = 0
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8', errors='ignore') as arquivo:
            for linha in arquivo:
                # O delimitador "$$$$" geralmente marca o fim de uma entrada de molécula
                if '$$$$' in linha.strip():
                    contador_moleculas += 1
        return contador_moleculas
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao ler o arquivo '{caminho_arquivo}': {e}")
        return None

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    # Verifica se um nome de arquivo foi passado como argumento na linha de comando
    if len(sys.argv) > 1:
        caminho_do_arquivo_sdf = sys.argv[1]
    else:
        # Se não, pede ao usuário para digitar o caminho
        caminho_do_arquivo_sdf = input("Digite o caminho completo para o arquivo SDF: ")

    # Chama a função para contar as moléculas
    numero_de_moleculas = contar_moleculas_sdf(caminho_do_arquivo_sdf)

    # Imprime o resultado se a contagem foi bem-sucedida
    if numero_de_moleculas is not None:
        print(f"\nO arquivo '{caminho_do_arquivo_sdf}' contém {numero_de_moleculas} moléculas.")
