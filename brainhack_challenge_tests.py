#!/usr/bin/env python3
"""Roda brainhack_challenge.py para varias combinacoes de --input.

A bateria e o PRODUTO de duas listas: COMBINACOES (o que entra como canal) e
WM_THRESHOLDS (o quanto a mascara de substancia branca aperta). Para cada par
executa

    python3 brainhack_challenge.py --input <componentes> [--wm-threshold <v>]

e acrescenta a saida em test_results.txt, no mesmo formato dos blocos que ja
estao la (comando, resultados, separador). Sao
len(COMBINACOES) * len(WM_THRESHOLDS) rodadas — conferir a conta antes de
comecar, porque cada uma e um pipeline inteiro.

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
RESULTS_FILE = REPO_ROOT / "logs/test_results.txt"


def _python_do_projeto():
    """Interpretador que roda o pipeline: o do .venv do repo, se existir.

    NOTA: nao usar sys.executable direto. Chamar `python3 brainhack_challenge_tests.py`
    sem ativar o .venv faz cada run herdar o python do sistema, que nao tem as
    dependencias, e a bateria inteira morre no import de albumentations.
    Dentro do venv os dois caminhos dao no mesmo interpretador.
    """
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


PYTHON = _python_do_projeto()

# Modulos de terceiros importados no topo de brainhack_challenge.py; usados so
# para conferir o ambiente antes de comecar (ver checar_ambiente).
DEPENDENCIAS = "albumentations, monai, nibabel, pytorch_lightning, SimpleITK, torch"

# Combinacoes de --input a testar, uma lista de componentes por run.
# Os componentes disponiveis sao os de INPUT_COMPONENTS em brainhack_challenge:
# fa, md, ad, rd, tensor, tensor_inv, b0, t1.
COMBINACOES = [
    ["fa"],
    ["tensor"],
    ["tensor_inv"],
    ["fa", "tensor_inv"],
    ["fa", "tensor_inv", "t1"]
    #["fa", "tensor"],
    # tensor_inv contra tensor: a mesma informacao do tensor, mas invariante a
    # rotacao da augmentation. O par a comparar e (tensor, tensor_inv) e
    # (fa+tensor, fa+tensor_inv) — ver tensor_invariants_from_channels.
    #["tensor_inv"],
    #["fa", "tensor_inv"],
    #["fa", "md"],
    #["fa", "md", "rd"],
    #["fa", "t1"],
    #["fa", "tensor", "t1"],
]

# Limiares de mascara de WM a testar. Cada valor e cruzado com CADA item de
# COMBINACOES, entao a bateria tem len(COMBINACOES) * len(WM_THRESHOLDS) runs —
# cuidado, cada run e um pipeline inteiro.
#
# None = sem mascara (a flag nem e passada), que e a linha de base para dizer se
# mascarar ajudou. Limiar menor = mascara mais larga: em 0.3 sobra ~0.8% do CC
# de fora da mascara, em 0.5 sobra ~2.2% — e o que fica de fora vira teto de
# Dice, porque a rede nunca ve esse voxel.
#
# A segmentacao do FSL FAST e cacheada por sujeito (ver extract_wm.py), entao
# trocar de limiar nao paga o FAST de novo, so o pre-processamento.
WM_THRESHOLDS: list[float | None] = [None, 0.3, 0.5]

# Argumentos extras aplicados a TODAS as combinacoes (ex.: ["--epochs", "50"]).
EXTRA_ARGS: list[str] = ["--epochs", "1000", "--lr", "1e-5"]

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


def rodadas():
    """Todas as (inputs, wm_threshold) da bateria: o produto das duas listas.

    A ordem varre os limiares por dentro, entao as rodadas do mesmo --input
    ficam vizinhas e da para comparar o efeito da mascara sem procurar no meio
    do arquivo.
    """
    return [(inputs, wm) for inputs in COMBINACOES for wm in WM_THRESHOLDS]


def script_args(inputs, wm_threshold):
    """Argumentos do brainhack_challenge.py para uma rodada."""
    args = ["--input", *inputs, *EXTRA_ARGS]
    if wm_threshold is not None:
        args += ["--wm-threshold", f"{wm_threshold:g}"]
    return args


def display_command(inputs, wm_threshold=None):
    """Comando como ele aparece no arquivo (python3, igual aos blocos antigos)."""
    return "python3 " + shlex.join([SCRIPT, *script_args(inputs, wm_threshold)])


def checar_ambiente():
    """Aborta antes da bateria se o interpretador nao tiver as dependencias.

    Sem isto um ambiente errado so aparece run a run, cada combinacao gravando
    um bloco FALHOU identico em test_results.txt.
    """
    proc = subprocess.run(
        [PYTHON, "-c", f"import {DEPENDENCIAS}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return

    print(f"Ambiente incompleto em {PYTHON}:\n", file=sys.stderr)
    print(proc.stderr.strip(), file=sys.stderr)
    print(
        "\nInstale as dependencias (pip install -r requirements.txt) ou rode a "
        "bateria com o python do .venv.",
        file=sys.stderr,
    )
    sys.exit(1)


def run(inputs, wm_threshold=None):
    """Roda uma combinacao, ecoando a saida ao vivo, e devolve (ok, saida)."""
    cmd = [PYTHON, SCRIPT, *script_args(inputs, wm_threshold)]
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
    # Cria a pasta antes: o append acontece no FIM de cada rodada, entao um
    # diretorio faltando (RESULTS_FILE aponta para logs/) so apareceria depois
    # do treino inteiro — e levaria junto o resultado que acabou de sair.
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # "a" para nunca sobrescrever o que ja foi medido.
    with open(RESULTS_FILE, "a") as f:
        f.write(texto)


def ja_testado(comando):
    """O comando ja aparece, EXATO, em test_results.txt?

    Compara linha inteira, e nao substring. Com wm_threshold=None a flag nem e
    passada, entao o comando da rodada sem mascara e PREFIXO do comando da
    rodada com mascara:

        python3 brainhack_challenge.py --input fa --epochs 500
        python3 brainhack_challenge.py --input fa --epochs 500 --wm-threshold 0.3

    Um `comando in texto` daria True para o primeiro assim que o segundo
    estivesse no arquivo, e retomar uma bateria pularia justamente a linha de
    base que da sentido a comparacao. O mesmo vale para "--input fa" como
    prefixo de "--input fa tensor".
    """
    if not RESULTS_FILE.exists():
        return False
    return any(
        linha.strip() == comando for linha in RESULTS_FILE.read_text().splitlines()
    )


def resumo_linha(resultados):
    """Uma linha de resumo para o terminal (o melhor threshold do run)."""
    for linha in resultados.splitlines():
        if linha.startswith("Melhor threshold:"):
            return linha
    return "(sem linha de melhor threshold)"


def main():
    checar_ambiente()

    lista = rodadas()
    total = len(lista)
    resumo = []
    print(f"Interpretador: {PYTHON}", flush=True)
    print(
        f"Bateria: {len(COMBINACOES)} entradas x {len(WM_THRESHOLDS)} limiares de WM "
        f"= {total} rodadas",
        flush=True,
    )

    for n, (inputs, wm) in enumerate(lista, 1):
        comando = display_command(inputs, wm)
        cabecalho = f"[{n}/{total}] {comando}"
        print(f"\n{'=' * 79}\n{cabecalho}\n{'=' * 79}", flush=True)

        if PULAR_JA_TESTADOS and ja_testado(comando):
            print("Ja esta em test_results.txt: pulando.", flush=True)
            resumo.append((comando, "pulado"))
            continue

        inicio = time.time()
        ok, saida = run(inputs, wm)
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
