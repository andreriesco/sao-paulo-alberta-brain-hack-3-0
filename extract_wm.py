#!/usr/bin/env python3
"""Extrai a mascara de substancia branca de todos os sujeitos com o FSL FAST.

Por que um comando separado: o FAST leva ~40 s por sujeito e nao depende de
nada do treino. Rodando dentro do `--stages preprocess`, ele segura a barra de
progresso por ~90 min na primeira vez, um sujeito de cada vez. Aqui roda em
paralelo e uma vez so; o pre-processamento seguinte encontra tudo em cache e
passa direto, para qualquer --input e qualquer --wm-threshold.

O FAST NAO roda por fatia: ele ve o volume 3D inteiro do T1, uma vez por
sujeito. Quem recorta a fatia e o pipeline, depois, sobre a mascara pronta.

Uso:

    python extract_wm.py                  # todos os sujeitos, 4 em paralelo
    python extract_wm.py --jobs 8         # mais paralelo (1 processo por job)
    python extract_wm.py --force          # re-segmenta quem ja esta em cache

Saida, por sujeito, em <cache>/<subject_id>/:

    t1_pve_2.nii.gz     mapa PVE de branca do FAST — e o que o pipeline consome
    t1_pve_0/1, t1_seg  o resto do que o FAST escreve (LCR, cinzenta, rotulos)
    wm_mask_las.nii.gz  mascara binaria no limiar --threshold, ja reorientada

Depois disto, o pipeline so le o cache:

    python brainhack_challenge.py --input fa --wm-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

# Reaproveita o pipeline em vez de repetir a chamada do FAST e a reorientacao:
# se a convencao mudar la, muda aqui junto.
from brainhack_challenge import (
    DEFAULT_WM_CACHE,
    MODES,
    REPO_ROOT,
    BrainHack3Data,
    find_data_root,
    find_fsl_fast,
    load_las,
    wm_pve_from_t1,
)


def subject_ids(split_dir):
    """Sujeitos dos tres splits, sem repetir e em ordem estavel."""
    ids = []
    for mode in MODES:
        with open(Path(split_dir) / f"{mode}.json") as f:
            ids.extend(json.load(f))
    return list(dict.fromkeys(ids))  # dict preserva a ordem e remove repetidos


def extract_one(subject_dir, cache_root, fast_bin, threshold, force):
    """Segmenta um sujeito e escreve a mascara binaria. Devolve (id, estado, voxels)."""
    subject_dir = Path(subject_dir)
    destino = Path(cache_root) / subject_dir.name

    ja_estava = bool(sorted(destino.glob("t1_pve_2.nii*")))
    if force and ja_estava:
        # Apaga a saida antiga: o FAST nao sobrescreve de forma confiavel
        # quando o tipo de saida do FSL muda entre as rodadas.
        for antigo in destino.glob("t1*"):
            antigo.unlink()
        ja_estava = False

    pve = wm_pve_from_t1(subject_dir / BrainHack3Data.T1_FILE, destino, fast_bin)

    # Mascara binaria ja em LAS, para dar para abrir por cima da FA sem
    # reorientar de novo. O affine vem da FA de proposito: e a grade em que o
    # pipeline de fato usa a mascara (ver load_las em brainhack_challenge).
    fa = nib.load(subject_dir / BrainHack3Data.FA_FILE)
    wm = (load_las(pve) >= threshold).astype(np.uint8)
    nib.save(nib.Nifti1Image(wm, fa.affine), destino / "wm_mask_las.nii.gz")

    return subject_dir.name, ("cache" if ja_estava else "novo"), int(wm.sum())


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Roda o FSL FAST em todos os sujeitos e guarda a mascara de WM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default=None,
                   help="Pasta com as subpastas de sujeito (padrao: descoberta em --search-root).")
    p.add_argument("--search-root", default=str(REPO_ROOT),
                   help="Onde procurar as pastas de sujeito.")
    p.add_argument("--split-dir", default=str(REPO_ROOT),
                   help="Pasta com train.json, val.json e test.json.")
    p.add_argument("--cache-dir", default=DEFAULT_WM_CACHE,
                   help="Onde guardar as segmentacoes (o mesmo --wm-cache-dir do pipeline).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Limiar do PVE para a mascara binaria salva. Nao afeta o "
                        "pipeline, que aplica o proprio --wm-threshold sobre o PVE.")
    p.add_argument("--jobs", type=int, default=4,
                   help="Quantos FAST rodar ao mesmo tempo (um processo cada).")
    p.add_argument("--force", action="store_true",
                   help="Re-segmenta mesmo quem ja esta em cache.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    fast_bin = find_fsl_fast()
    if fast_bin is None:
        raise SystemExit(
            "FSL nao encontrado: nao ha `fast` no PATH nem em $FSLDIR. Instale o FSL."
        )

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Sujeito = pasta que tem FA, mascara de CC e T1 (o T1 e o que interessa aqui).
        data_dir = find_data_root(
            Path(args.search_root), BrainHack3Data.required_files((), wm_mask=True)
        )

    ids = subject_ids(args.split_dir)
    print(f"FAST: {fast_bin}")
    print(f"Sujeitos: {len(ids)} em {data_dir}")
    print(f"Cache: {args.cache_dir}  |  {args.jobs} em paralelo")

    inicio = time.time()
    novos, em_cache, erros = 0, 0, []

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        # ThreadPool e nao ProcessPool: cada tarefa so espera um processo
        # externo (o FAST), entao as threads ficam bloqueadas em I/O e o GIL
        # nao atrapalha.
        futuros = {
            pool.submit(
                extract_one, data_dir / str(sid), args.cache_dir, fast_bin,
                args.threshold, args.force,
            ): sid
            for sid in ids
        }
        for futuro in tqdm(as_completed(futuros), total=len(futuros), desc="FAST"):
            sid = futuros[futuro]
            try:
                _, estado, voxels = futuro.result()
            except Exception as exc:  # um sujeito quebrado nao derruba a bateria
                erros.append((sid, exc))
                continue
            if estado == "novo":
                novos += 1
            else:
                em_cache += 1
            if voxels == 0:
                erros.append((sid, "mascara de WM vazia"))

    minutos = (time.time() - inicio) / 60
    print(f"\nSegmentados agora: {novos}  |  ja em cache: {em_cache}  "
          f"|  falhas: {len(erros)}  ({minutos:.1f} min)")
    for sid, motivo in erros:
        print(f"  FALHOU {sid}: {motivo}")

    if erros:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
