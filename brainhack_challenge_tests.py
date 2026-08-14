#!/usr/bin/env python3
"""Roda brainhack_challenge.py para varias combinacoes de --input.

Para cada combinacao da lista COMBINACOES abaixo, executa

    python3 brainhack_challenge.py --input <componentes>

e acrescenta a saida em test_results.txt, no mesmo formato dos blocos que ja
estao la (comando, resultados, separador).

Uso:

    python3 brainhack_challenge_tests.py

Cada combinacao roda o pipeline inteiro (preprocess + treino + eval + teste),
entao isso demora. A saida de cada run aparece ao vivo no terminal e so o
trecho de resultados vai para o arquivo.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# Configuracao
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT = "brainhack_challenge.py"
RESULTS_FILE = REPO_ROOT / "test_results.txt"

# Combinacoes de --input a testar, uma lista de componentes por run.
# Os componentes disponiveis sao os de INPUT_COMPONENTS em brainhack_challenge:
# fa, md, ad, rd, tensor, b0, t1.
COMBINACOES = [
    ["fa"],
    ["tensor"],
    ["fa", "tensor"],
    #["fa", "md"],
    #["fa", "md", "rd"],
    #["fa", "t1"],
    #["fa", "tensor", "t1"],
]

# Argumentos extras aplicados a TODAS as combinacoes (ex.: ["--epochs", "50"]).
EXTRA_ARGS: list[str] = []

# Pular combinacoes cujo comando ja aparece em test_results.txt. Util para
# retomar uma bateria interrompida sem repetir horas de treino.
PULAR_JA_TESTADOS = False

SEPARADOR = "-" * 79

# O bloco de resultados comeca na varredura de threshold; sem varredura
# (--no-threshold-sweep) o primeiro marcador e o relatorio da validacao.
MARCADORES_INICIO = (
    "Varredura de threshold na validacao",
    "[val] threshold=",
)


# ============================================================================
# Execucao
# ============================================================================


def display_command(inputs):
    """Comando como ele aparece no arquivo (python3, igual aos blocos antigos)."""
    return "python3 " + shlex.join([SCRIPT, "--input", *inputs, *EXTRA_ARGS])


def run(inputs):
    """Roda uma combinacao, ecoando a saida ao vivo, e devolve (ok, saida)."""
    cmd = [sys.executable, SCRIPT, "--input", *inputs, *EXTRA_ARGS]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # junta stderr para nao perder tracebacks
    )

    # Le em blocos crus (e nao linha a linha) para as barras de progresso, que
    # usam \r sem \n, continuarem se atualizando no lugar em vez de encher a
    # tela de linhas soltas.
    pedacos = []
    while True:
        data = os.read(proc.stdout.fileno(), 8192)
        if not data:
            break
        texto = data.decode("utf-8", errors="replace")
        sys.stdout.write(texto)
        sys.stdout.flush()
        pedacos.append(texto)

    proc.stdout.close()
    return proc.wait() == 0, "".join(pedacos)


def extract_results(saida):
    """Recorta o trecho de resultados; None se ele nao aparecer na saida."""
    # As barras de progresso deixam varias "linhas" grudadas num \r so; tratar
    # \r como quebra evita que o marcador fique escondido no meio de uma delas.
    linhas = saida.replace("\r", "\n").splitlines()
    for i, linha in enumerate(linhas):
        if any(linha.lstrip().startswith(m) for m in MARCADORES_INICIO):
            return "\n".join(linhas[i:]).strip("\n")
    return None


def bloco(comando, resultados):
    """Monta um bloco no formato do test_results.txt."""
    return (
        f"Input command:\n\n{comando}\n\n"
        f"Results:\n\n{resultados}\n\n"
        f"{SEPARADOR}\n\n"
    )


def append_results(texto):
    # "a" para nunca sobrescrever o que ja foi medido.
    with open(RESULTS_FILE, "a") as f:
        f.write(texto)


def ja_testado(comando):
    if not RESULTS_FILE.exists():
        return False
    return comando in RESULTS_FILE.read_text()


def resumo_linha(resultados):
    """Uma linha de resumo para o terminal (o melhor threshold do run)."""
    for linha in resultados.splitlines():
        if linha.startswith("Melhor threshold:"):
            return linha
    return "(sem linha de melhor threshold)"


def main():
    total = len(COMBINACOES)
    resumo = []

    for n, inputs in enumerate(COMBINACOES, 1):
        comando = display_command(inputs)
        cabecalho = f"[{n}/{total}] {comando}"
        print(f"\n{'=' * 79}\n{cabecalho}\n{'=' * 79}", flush=True)

        if PULAR_JA_TESTADOS and ja_testado(comando):
            print("Ja esta em test_results.txt: pulando.", flush=True)
            resumo.append((comando, "pulado"))
            continue

        inicio = time.time()
        ok, saida = run(inputs)
        minutos = (time.time() - inicio) / 60

        resultados = extract_results(saida) if ok else None

        if resultados is None:
            # Guarda o fim da saida para o erro nao se perder: o bloco fica no
            # arquivo com o mesmo formato dos demais, so que com o traceback.
            cauda = "\n".join(saida.replace("\r", "\n").splitlines()[-25:]).strip()
            motivo = (
                "comando terminou com erro"
                if not ok
                else "nao encontrei o trecho de resultados na saida"
            )
            resultados = f"FALHOU: {motivo}.\n\nFim da saida:\n\n{cauda}"
            estado = "FALHOU"
        else:
            estado = resumo_linha(resultados)

        append_results(bloco(comando, resultados))
        print(f"\n--> {estado}  ({minutos:.1f} min)  -> {RESULTS_FILE.name}", flush=True)
        resumo.append((comando, estado))

    print(f"\n{'=' * 79}\nResumo\n{'=' * 79}")
    for comando, estado in resumo:
        print(f"  {comando}\n      {estado}")


if __name__ == "__main__":
    main()
