#!/usr/bin/env python3
"""brainhack_challenge — versao script do notebook do hands-on (BrainHack 3.0).

Segmentacao do corpo caloso na fatia sagital media com uma UNet 2D treinada
com PyTorch Lightning.

A entrada da rede e uma LISTA de componentes (`--input`), concatenados como
canais na ordem pedida — de "so a FA" (o notebook original) a combinacoes como
`--input fa tensor md`. Os componentes disponiveis estao em INPUT_COMPONENTS:
metricas escalares (fa, md, ad, rd, b0, t1) valem 1 canal cada, `tensor` vale
6, as componentes unicas de D (que e simetrico, entao suas 9 componentes
carregam apenas 6 numeros independentes: Dxx Dxy Dxz Dyy Dyz Dzz), e
`tensor_inv` vale 5.

Cada componente se normaliza sozinho, porque as unidades nao sao comparaveis:
FA e adimensional em [0, 1], as difusividades estao em mm^2/s e T1/b0 vem em
unidades arbitrarias de scanner. Diferente das metricas escalares, o tensor
preserva a orientacao das fibras — o que exige o cuidado documentado em
`RotateCropTensor` (um campo tensorial nao gira como um campo escalar).

`tensor_inv` e a alternativa ao `tensor`: os mesmos 6 numeros reescritos como 5
invariantes sob a rotacao que a augmentation aplica (ver
`tensor_invariants_from_channels`). Carrega a orientacao das fibras, que as
metricas escalares jogam fora, mas sem depender do referencial, que e o que
obriga o `tensor` a girar junto com a imagem.

O notebook `brainhack-3-0-ii-encontro-ismrm-brasil-full.ipynb` foi reorganizado
em estagios executaveis, na mesma ordem das celulas:

    1. preprocess : le os volumes 3D (entrada + mascara CC), normaliza, extrai
                    a fatia sagital media e salva um .npz por sujeito.
    2. train      : treina a UNet 2D sobre as fatias salvas.
    3. eval       : inferencia na validacao + varredura de threshold.
    4. test       : inferencia no teste com o melhor threshold.

Uso tipico (tudo de uma vez, dados descobertos automaticamente):

    python brainhack_challenge.py

Combinando metricas (7 canais: 1 de FA + 6 do tensor):

    python brainhack_challenge.py --input fa tensor

Somente treino, com os .npz ja gerados:

    python brainhack_challenge.py --stages train --epochs 50

Smoke test rapido (1 batch de treino e 1 de validacao):

    python brainhack_challenge.py --debug

Diferencas em relacao ao notebook estao marcadas com `# NOTA:`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from glob import glob
from math import nan
from pathlib import Path

import albumentations as A
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.transforms import Orientation
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# Configuracao
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_LOGS_ROOT = "logs"
DEFAULT_EXPERIMENT = "BrainhackCC"
# Onde ficam as segmentacoes do FSL FAST (ver MaskByWM). Uma por sujeito,
# reaproveitada por todas as combinacoes de --input.
DEFAULT_WM_CACHE = "fsl_fast_wm"
SAGITTAL_AXIS = 0
MODES = ("train", "val", "test")
# Estagios do pipeline, na ordem em que rodam (ver --stages).
STAGES = ("preprocess", "train", "eval", "test")

DEFAULT_INPUT = ("fa",)

# Indices das 6 componentes unicas de D (simetrico), na ordem em que viram
# canais: Dxx Dxy Dxz Dyy Dyz Dzz. Listas (e nao np.triu_indices) porque
# servem para indexar tanto arrays numpy quanto tensores torch.
TRIU = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
N_TENSOR_CHANNELS = 6

# Difusividade de referencia (mm^2/s): a da agua livre a 37 C. Escala FIXA
# usada para normalizar difusividades — ver norm_diffusivity.
D_REF = 3e-3


def find_data_root(start: Path, targets) -> Path:
    """Retorna o diretorio pai das pastas de sujeito, a qualquer profundidade.

    Identifica sujeito pelo conteudo (presenca dos NIfTI exigidos pela entrada
    escolhida), nao por profundidade nem por formato do nome.
    """
    targets = set(targets)
    # os.walk para com o primeiro acerto; rglob("*") teria que listar a arvore
    # inteira (dezenas de milhares de arquivos) antes de testar o primeiro candidato.
    for root, dirs, files in os.walk(start):
        dirs.sort()  # ordem deterministica, como o sorted() original
        if targets <= set(files):
            return Path(root).parent
    raise FileNotFoundError(
        f"Nenhuma pasta contendo {sorted(targets)} encontrada em {start}"
    )


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(path).get_fdata()


# ============================================================================
# Tensor de difusao: componentes, invariantes e rotacao
# ============================================================================


def tensor_to_channels(D):
    """(..., 3, 3) -> (..., 6): as componentes unicas de um tensor simetrico."""
    return D[..., TRIU[0], TRIU[1]]


def channels_to_tensor(comp):
    """(..., 6) -> (..., 3, 3): reconstroi a matriz simetrica completa."""
    D = np.empty(comp.shape[:-1] + (3, 3), dtype=comp.dtype)
    D[..., TRIU[0], TRIU[1]] = comp
    D[..., TRIU[1], TRIU[0]] = comp
    return D


def fa_from_channels(comp):
    """FA a partir das 6 componentes, sem autodecomposicao.

    FA^2 = 3/2 * ||D - (tr D / 3) I||^2 / ||D||^2, e as duas normas de
    Frobenius saem direto das componentes. Confere com o FA.nii do dataset ate
    ~2e-5 dentro do cerebro. E invariante a escala, entao da o mesmo resultado
    antes ou depois da normalizacao.
    """
    dxx, dxy, dxz, dyy, dyz, dzz = (comp[..., k] for k in range(6))
    tr = dxx + dyy + dzz
    norm2 = dxx**2 + dyy**2 + dzz**2 + 2 * (dxy**2 + dxz**2 + dyz**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = np.sqrt(np.clip(1.5 * (1.0 - (tr**2 / 3.0) / norm2), 0.0, 1.0))
    return np.nan_to_num(fa)


def tensor_invariants_from_channels(comp):
    """(..., 6) -> (..., 5): invariantes de D sob rotacao em torno do eixo x.

    Motivacao: as 6 componentes de D dependem do referencial. A augmentation
    gira a fatia no plano (y, z), ou seja D -> R D R^T com R = rotation_matrix_x,
    e ai as componentes se misturam — a rede precisa aprender essa invariancia
    pelos exemplos. Estes 5 canais ja sao invariantes por construcao.

    Escrevendo D em blocos, com c = (dxy, dxz) e B = [[dyy, dyz], [dyz, dzz]],
    uma rotacao em torno de x fixa dxx, gira c como um vetor 2D e leva B em
    R2 B R2^T. Logo dxx, tr D, |c|, det B e o quociente de Rayleigh
    c^T B c / |c|^2 nao mudam. Sao 5 funcoes independentes = os 6 graus de
    liberdade de D menos o parametro da rotacao, entao nao se perde nada alem
    da fase arbitraria.

    Por que dxx importa aqui: na fatia sagital media as fibras do CC cruzam a
    linha media, entao a direcao principal aponta para fora do plano (eixo x).
    As outras estruturas de FA alta da fatia (fornix, cingulo) correm no plano.
    O canal dxx/tr separa justamente esses casos — e nenhuma metrica escalar
    (fa, md, ad, rd) consegue, porque todas dependem so dos autovalores.

    Todos os canais sao de primeira ordem em difusividade (dai as raizes: um
    termo quadratico ficaria espremido perto de zero) e normalizados pelo traco,
    para cairem na mesma faixa ~[0, 1]. Canais com escalas muito diferentes
    entram desequilibrados na primeira convolucao.
    """
    dxx, dxy, dxz, dyy, dyz, dzz = (comp[..., k] for k in range(6))
    tr = dxx + dyy + dzz
    # tr <= 0 so acontece fora do cerebro (ou em voxel de ruido): divide por 1
    # ali e deixa o clip final zerar o canal.
    safe_tr = np.where(tr > 0, tr, 1.0)

    c2 = dxy**2 + dxz**2  # |c|^2
    det_b = dyy * dzz - dyz**2  # >= 0 quando D e positivo definido

    # c^T B c / |c|^2 fica entre os autovalores de B, entao ainda e uma
    # difusividade: dividido pelo traco vira adimensional como os outros.
    with np.errstate(invalid="ignore", divide="ignore"):
        rayleigh = np.where(
            c2 > 0,
            (dxy**2 * dyy + 2 * dxy * dxz * dyz + dxz**2 * dzz) / np.where(c2 > 0, c2, 1.0),
            0.0,
        )

        inv = np.stack(
            [
                dxx / safe_tr,  # fracao da difusao que atravessa o plano
                tr / (3.0 * D_REF),  # escala (= MD normalizada como norm_diffusivity)
                np.sqrt(c2) / safe_tr,  # acoplamento entre o plano e o eixo x
                np.sqrt(np.clip(det_b, 0.0, None)) / safe_tr,  # media geom. dos autovals de B
                rayleigh / safe_tr,  # orientacao de c no referencial proprio de B
            ],
            axis=-1,
        )

    # Com D positivo definido todos os canais ja caem em [0, 1] (dxx <= tr,
    # e |c|, sqrt(det B) <= tr/2). O clip so apara ruido que torna D nao-SPD.
    return np.nan_to_num(np.clip(inv, 0.0, 1.0))


def rotation_matrix_x(angle_rad):
    """Rotacao em torno do eixo sagital (eixo 0) = o plano (y, z) da fatia."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rotate_tensor_channels(comp, R):
    """Gira um campo tensorial: D' = R D R^T em cada voxel. `comp` e (..., 6).

    Isso NAO e opcional quando se gira a imagem. Um campo escalar (FA) so
    precisa que o grid gire; num campo tensorial as componentes precisam girar
    junto, senao a orientacao codificada em D deixa de corresponder a anatomia
    que aparece na imagem.
    """
    D = channels_to_tensor(np.asarray(comp, dtype=np.float32))
    D = np.einsum("ij,...jk,lk->...il", R, D, R)
    return np.ascontiguousarray(tensor_to_channels(D), dtype=np.float32)


# ============================================================================
# Componentes de entrada: cada metrica vira um ou mais canais
# ============================================================================
#
# A entrada da rede e uma LISTA de componentes (--input fa tensor), concatenados
# na ordem pedida. Cada componente sabe (a) de quais NIfTI depende, (b) quantos
# canais produz e (c) como se normaliza — porque metricas em unidades
# diferentes nao podem compartilhar uma normalizacao so.


def norm_unit(x):
    """Ja adimensional em [0, 1] (FA): so garante a faixa.

    NOTA: substitui a MinMaxNormalize do notebook para a FA. No dataset a FA
    tem min exatamente 0 e max 0.999999..1.0, entao min-max era identidade a
    menos de 1e-6 — mas com escala FIXA os sujeitos ficam comparaveis entre si.
    """
    return np.clip(x, 0.0, 1.0)


def norm_diffusivity(x):
    """Difusividades (mm^2/s: MD, AD, RD e as componentes de D) por D_REF.

    Escala fixa, nao min-max: um voxel ruidoso nao pode reescalar o sujeito
    inteiro, e o zero continua significando "sem difusao" em todos eles.
    """
    return np.clip(x / D_REF, -1.0, 1.0)


def norm_robust(x, pct=99.0):
    """Intensidades em unidades arbitrarias (T1, b0): escala pelo percentil 99.

    Nao min-max: essas imagens tem cauda longa (no dataset, max ~3.4x o p99),
    entao dividir pelo maximo comprime o tecido todo na parte de baixo da
    faixa. O p99 tambem varia ~1.4x entre sujeitos, o que descarta usar uma
    escala fixa: aqui a normalizacao por sujeito e a escolha certa.
    """
    ref = np.percentile(x, pct)
    return np.clip(x / ref, 0.0, 1.0) if ref > 0 else x


def load_las(path):
    """Carrega um NIfTI e reorienta para LAS, a orientacao das metricas de difusao.

    NOTA: isto NAO e defensivo, e obrigatorio. Neste dataset FA, MD, AD, RD,
    evals, evecs e mean_b0 estao todos em LAS, mas o T1 e a mascara de CC vem em
    orientacoes que MUDAM de sujeito para sujeito — nos 20 primeiros sujeitos
    aparecem dez combinacoes diferentes (SAL, IPR, LPS, SPR, ...). Empilhar um
    T1 cru como canal ao lado da FA junta dois volumes que nao estao no mesmo
    espaco: o canal fica espelhado/transposto em relacao aos outros.

    Para volumes que ja estao em LAS a reorientacao nao faz nada, entao da para
    aplicar em todos os escalares sem pensar em qual precisa.

    So serve para campos ESCALARES. Um campo vetorial ou tensorial (evecs, as
    componentes de D) precisaria girar as componentes junto com o grid — mas
    esses ja vem em LAS, entao nao passam por aqui.
    """
    nii = nib.load(path)
    vol = np.nan_to_num(np.asarray(nii.dataobj, dtype=np.float32))
    las = Orientation(axcodes="LAS")(MetaTensor(np.expand_dims(vol, 0), affine=nii.affine))
    return np.ascontiguousarray(las.numpy()[0], dtype=np.float32)


def _scalar_loader(filename, normalize):
    """Componente de 1 canal lido de um NIfTI escalar."""

    def load(subject_dir):
        vol = load_las(Path(subject_dir) / filename)
        return normalize(np.expand_dims(vol, 0))

    return load


def _tensor_components(subject_dir):
    """(X, Y, Z, 6): as componentes unicas de D, CRUAS (em mm^2/s).

    Sem normalizar de proposito: quem consome escolhe a escala. O `tensor` usa
    norm_diffusivity; o `tensor_inv` precisa do traco na unidade original para
    normalizar o canal de escala do mesmo jeito que as outras difusividades.
    """
    subject_dir = Path(subject_dir)
    evals = nib.load(subject_dir / "evals.nii").get_fdata(dtype=np.float32)
    evecs = nib.load(subject_dir / "evecs.nii").get_fdata(dtype=np.float32)
    if evecs.shape[-1] == 9:  # achatado (X, Y, Z, 9)
        evecs = evecs.reshape(*evecs.shape[:-1], 3, 3)

    # D = V diag(lambda) V^T. Os autovetores estao nas COLUNAS de V, ou seja
    # evecs[..., :, i] corresponde a evals[..., i] — por isso o '...kj' (a
    # transposta) no final. Trocar por '...jk' produz um tensor simetrico e
    # plausivel, mas errado: neste dataset as duas convencoes diferem por ~40
    # graus na direcao principal.
    D = np.einsum("...ij,...j,...kj->...ik", evecs, evals, evecs)

    return np.nan_to_num(tensor_to_channels(D))  # (X, Y, Z, 6)


def _tensor_loader(subject_dir):
    """Componente de 6 canais: as componentes unicas de D."""
    comp = _tensor_components(subject_dir)
    return norm_diffusivity(np.moveaxis(comp, -1, 0))


def _tensor_inv_loader(subject_dir):
    """Componente de 5 canais: invariantes de D sob a rotacao da augmentation.

    Alternativa ao `tensor`: a mesma informacao, menos a fase da rotacao no
    plano da fatia, num referencial que a augmentation nao mexe. Ver
    tensor_invariants_from_channels. Ja sai em [0, 1], entao nao passa pelas
    funcoes norm_*.
    """
    inv = tensor_invariants_from_channels(_tensor_components(subject_dir))
    return np.moveaxis(inv, -1, 0)


class InputComponent:
    """Uma metrica que pode virar canal(is) de entrada da rede."""

    def __init__(self, channels, files, load, descricao):
        self.channels = channels
        self.files = tuple(files)  # NIfTI que a pasta do sujeito precisa ter
        self.load = load  # (subject_dir) -> (canais, X, Y, Z) ja normalizado
        self.descricao = descricao


INPUT_COMPONENTS = {
    "fa": InputComponent(1, ["FA.nii"], _scalar_loader("FA.nii", norm_unit),
                         "anisotropia fracionada"),
    "md": InputComponent(1, ["MD.nii"], _scalar_loader("MD.nii", norm_diffusivity),
                         "difusividade media"),
    "ad": InputComponent(1, ["AD.nii"], _scalar_loader("AD.nii", norm_diffusivity),
                         "difusividade axial"),
    "rd": InputComponent(1, ["RD.nii"], _scalar_loader("RD.nii", norm_diffusivity),
                         "difusividade radial"),
    "tensor": InputComponent(6, ["evals.nii", "evecs.nii"], _tensor_loader,
                             "as 6 componentes unicas de D"),
    "tensor_inv": InputComponent(5, ["evals.nii", "evecs.nii"], _tensor_inv_loader,
                                 "5 invariantes de D sob a rotacao do plano da fatia"),
    "b0": InputComponent(1, ["mean_b0.nii"], _scalar_loader("mean_b0.nii", norm_robust),
                         "b0 medio"),
    "t1": InputComponent(1, ["T1_brain_1.25.nii"],
                         _scalar_loader("T1_brain_1.25.nii", norm_robust),
                         "T1 sem cranio, na grade da difusao"),
}


def normalize_components(names):
    """Valida a lista de --input e remove repeticoes, preservando a ordem."""
    vistos = []
    for name in names:
        if name not in INPUT_COMPONENTS:
            raise ValueError(
                f"Componente de entrada desconhecido: {name!r}. "
                f"Disponiveis: {', '.join(sorted(INPUT_COMPONENTS))}"
            )
        if name not in vistos:
            vistos.append(name)
    if not vistos:
        raise ValueError("--input precisa de pelo menos um componente.")
    return tuple(vistos)


def channel_layout(names):
    """{componente: (inicio, fim)} nos canais concatenados, na ordem de --input.

    Quem precisa disso: a augmentation, que tem que girar as componentes de D
    (e SO elas) quando gira a imagem, e a visualizacao, que precisa achar um
    canal escalar para mostrar.
    """
    layout, inicio = {}, 0
    for name in names:
        fim = inicio + INPUT_COMPONENTS[name].channels
        layout[name] = (inicio, fim)
        inicio = fim
    return layout


def n_input_channels(names):
    return sum(INPUT_COMPONENTS[n].channels for n in names)


def input_tag(names):
    """Nome curto do conjunto de entradas, para pastas e experimentos."""
    return "+".join(names)


# ============================================================================
# Dataset 3D
# ============================================================================


class BrainHack3Data(Dataset):
    """Le os volumes 3D dos sujeitos do split (mode) informado.

    Separar dados no nivel do paciente e essencial para evitar contaminacao
    (dados do mesmo paciente no treino e no teste, por exemplo).

    A classe tem uma unica responsabilidade: ler os volumes do disco. O
    pre-processamento fica em transformadas opcionais, passadas no construtor.
    """

    # Uso de propriedades da classe para constantes relacionadas ao dataset.
    FA_FILE = "FA.nii"
    EVALS_FILE = "evals.nii"
    EVECS_FILE = "evecs.nii"
    CC_FILE = "cc_mask_mricloud_1.25.nii"
    T1_FILE = "T1_brain_1.25.nii"

    @classmethod
    def required_files(cls, components, wm_mask=False):
        """NIfTI que uma pasta precisa ter para ser reconhecida como sujeito.

        A FA entra sempre: mesmo quando nao e canal de entrada, e ela que
        escolhe a fatia sagital media (ver ExtractMidSagittalSlice). O T1 entra
        quando a mascara de WM esta ligada, mesmo que ele nao seja canal de
        entrada — e dele que sai a segmentacao (ver MaskByWM).
        """
        arquivos = {cls.FA_FILE, cls.CC_FILE}
        for name in components:
            arquivos.update(INPUT_COMPONENTS[name].files)
        if wm_mask:
            arquivos.add(cls.T1_FILE)
        return arquivos

    def __init__(
        self, mode, data_dir, split_dir, components=DEFAULT_INPUT, transform=None, fix=True
    ):
        """mode: train, val ou test.

        NOTA: no notebook `DATA_DIR`/`DATA_JSON` eram globais; aqui sao
        argumentos explicitos, para o dataset nunca divergir do caminho que o
        pre-processamento realmente usou.
        """
        self.data_dir = Path(data_dir)
        self.components = normalize_components(components)

        # train/val/test tem indices distintos -> montamos o nome do JSON a partir de mode.
        split_path = Path(split_dir) / f"{mode}.json"
        with open(split_path) as f:
            self.subject_ids = json.load(f)

        self.transform = transform
        self.fix = fix

        if self.fix:
            # A mascara de CC vem em orientacao diferente da FA em parte dos
            # sujeitos; reorientamos para LAS antes de qualquer recorte.
            self.las = Orientation(axcodes="LAS")

    def __len__(self):
        return len(self.subject_ids)

    def _load_input(self, subject_dir):
        """Volume de entrada no formato [canal, X, Y, Z].

        Cada componente ja se normaliza no proprio loader: FA, difusividades e
        intensidades vivem em unidades diferentes e nao poderiam compartilhar
        uma normalizacao unica depois de concatenadas.
        """
        partes = [INPUT_COMPONENTS[n].load(subject_dir) for n in self.components]
        return np.ascontiguousarray(np.concatenate(partes, 0), dtype=np.float32)

    def _load_fa(self, subject_dir):
        """FA crua (sem normalizar), usada como criterio pelas transformadas."""
        return load_las(subject_dir / BrainHack3Data.FA_FILE)

    def __getitem__(self, i):
        subject_id = self.subject_ids[i]
        subject_dir = self.data_dir / subject_id

        cc_nii = nib.load(subject_dir / BrainHack3Data.CC_FILE)

        img = self._load_input(subject_dir)
        mask = np.expand_dims(cc_nii.get_fdata(), 0).astype(np.float32)

        if self.fix:
            mask = self.las(MetaTensor(mask, affine=cc_nii.affine)).numpy()

        if self.transform is not None:
            # ctx leva a FA junto com (x, y). Sem isso, transformadas que
            # dependem da FA (escolher a fatia, mascarar por limiar) teriam que
            # adivinhar onde ela esta nos canais — e nao existiria criterio
            # nenhum para uma entrada como "--input md rd".
            img, mask = self.transform(
                img, mask, {"fa": self._load_fa(subject_dir), "subject_dir": subject_dir}
            )

        metadata = {"subject_id": subject_id}
        return img, mask, metadata


# ============================================================================
# Transformadas do dataset 3D
# ============================================================================


class ComposeTransforms:
    """Aplica uma lista de transformadas em sequencia.

    Alem de (x, y), carrega um `ctx` opcional com dados do sujeito que nao sao
    nem entrada nem alvo — hoje a FA, que serve de criterio para escolher a
    fatia e para mascarar por limiar. As transformadas podem reescrever o ctx
    (a extracao de fatia troca o volume de FA pela fatia correspondente), de
    modo que ele sempre acompanha o x atual.
    """

    def __init__(self, transforms):
        # transforms e uma lista de objetos "chamaveis" (implementam __call__).
        self.transforms = transforms

    def __call__(self, x, y=None, ctx=None):
        # y=None e o caminho de inferencia (volume novo, sem mascara).
        ctx = {} if ctx is None else dict(ctx)
        for t in self.transforms:
            x, y = t(x, y, ctx)
        return x, y


# NOTA: a normalizacao deixou de ser uma transformada da cadeia e passou a
# viver em cada componente de entrada (norm_unit / norm_diffusivity /
# norm_robust, la em INPUT_COMPONENTS). Com "--input fa tensor t1" nao existe
# uma normalizacao unica que sirva: sao tres unidades diferentes no mesmo
# empilhamento de canais, e cada uma precisa da sua.


class ExtractMidSagittalSlice:
    """Extrai a fatia sagital media de volumes 3D.

    Criterio: menor FA medio entre as fatias com tecido suficiente segundo a
    mascara de CEREBRO (aqui, fa_vol > 0). A mascara do corpo caloso (o alvo y)
    NAO e usada: ela e o que a rede deve prever, e num volume novo nem existe.

    A FA vem do `ctx` (a FA crua do sujeito), nunca dos canais de x. Assim a
    fatia escolhida e a MESMA independente do que foi pedido em --input — se o
    criterio saisse dos canais, "--input md rd" nao teria criterio nenhum, e
    "--input t1" escolheria a fatia por intensidade de T1.
    """

    def __init__(self, sagittal_axis=SAGITTAL_AXIS):
        self.sagittal_axis = sagittal_axis

    def _brain_mask(self, fa_vol, subject_dir=None):
        # Mascara de cerebro: vem da ENTRADA, nunca do alvo.
        # NOTA: existe um T1_brain_mask_1.25.nii por sujeito, mas usa-lo aqui
        # mudaria a fatia escolhida em ~1 a cada 12 sujeitos em relacao ao que
        # o notebook faz. Mantido em fa_vol > 0 de proposito; passar
        # subject_dir ativa o outro caminho para quem quiser comparar.
        if subject_dir is not None:
            hits = sorted(Path(subject_dir).glob("T1_brain_mask_1.25.nii*"))
            if hits:
                return load_nifti(hits[0]) > 0
        return fa_vol > 0

    def _find_slice_index(self, fa_vol):
        other_axes = tuple(i for i in range(fa_vol.ndim) if i != self.sagittal_axis)
        mask_count = self._brain_mask(fa_vol).sum(axis=other_axes)
        fa_mean = fa_vol.mean(axis=other_axes)
        fa_mean[mask_count <= 0.90 * mask_count.max()] = 1
        return int(np.argmin(fa_mean))

    def __call__(self, x, y=None, ctx=None):
        ctx = {} if ctx is None else ctx
        fa_vol = ctx.get("fa")
        if fa_vol is None:
            # Sem ctx (uso solto da transformada): cai para o primeiro canal.
            fa_vol = x[0]
        slice_idx = self._find_slice_index(np.array(fa_vol, copy=True))

        # +1 porque o eixo 0 de x e o de canais: recorta TODOS os canais.
        x = np.take(x, slice_idx, axis=self.sagittal_axis + 1).astype(np.float32)
        # O ctx acompanha o x: daqui para frente "fa" e a FATIA de FA.
        ctx["fa"] = np.take(fa_vol, slice_idx, axis=self.sagittal_axis)
        ctx["slice_idx"] = slice_idx

        if y is None:
            return x, None  # inferencia: nao ha mascara a recortar
        cc_slice = np.take(y[0], slice_idx, axis=self.sagittal_axis)
        y = np.expand_dims((cc_slice > 0).astype(np.float32), 0)
        return x, y


class ThresholdByFA:
    """Zera os voxels cuja FA fica ABAIXO de `threshold`, mantendo o resto.

    A mascara vem da ENTRADA, nunca do alvo, e sempre da FA do `ctx` — entao o
    mesmo limiar seleciona exatamente os mesmos voxels seja qual for o
    --input, inclusive num que nem inclua a FA como canal. Como FA e
    adimensional e vive em [0, 1], o limiar e diretamente interpretavel (~0.2
    costuma separar substancia branca do resto).

    NOTA: roda DEPOIS da extracao da fatia, de proposito. Se rodasse antes,
    mudaria os dois criterios que a extracao usa — a FA media por fatia e a
    mascara de cerebro `fa_vol > 0`, que viraria `fa >= limiar`, encolhendo o
    "cerebro" para so a substancia branca. Mascarar a entrada nao deveria
    mudar QUAL fatia e vista.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x, y=None, ctx=None):
        ctx = {} if ctx is None else ctx
        fa = ctx.get("fa")
        if fa is None:
            fa = x[0] if x.shape[0] == 1 else fa_from_channels(np.moveaxis(x, 0, -1))
        # x e [canal, ...] e a FA e um mapa escalar: o broadcast mascara todos
        # os canais de forma coerente entre si.
        return (x * (fa >= self.threshold)).astype(np.float32), y


def find_fsl_fast():
    """Caminho do executavel `fast` do FSL, ou None se nao der para achar.

    Procura no PATH e, se nao achar, nos dois layouts de instalacao do FSL sob
    $FSLDIR (o de 6.0.6+ e o anterior) — o FSLDIR costuma estar definido mesmo
    quando o PATH nao foi exportado para a sessao.
    """
    caminho = shutil.which("fast")
    if caminho:
        return caminho

    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        for relativo in ("share/fsl/bin/fast", "bin/fast"):
            candidato = Path(fsldir) / relativo
            if candidato.exists():
                return str(candidato)
    return None


def wm_pve_from_t1(t1_path, out_dir, fast_bin):
    """Segmenta o T1 com o FSL FAST e devolve o mapa PVE de substancia branca.

    Classes do FAST com `-t 1` (T1), em ordem crescente de intensidade:
    pve_0 = LCR, pve_1 = cinzenta, pve_2 = BRANCA — e por isso que o arquivo
    procurado e o _pve_2.

    O resultado fica em cache no disco: o FAST leva cerca de um minuto por
    sujeito, e o pipeline le cada sujeito de novo a cada combinacao de --input.
    Como o cache e por sujeito (e nao por experimento), a segmentacao acontece
    uma vez so e serve a todas as combinacoes.

    NOTA: `-N` desliga a correcao de campo de bias, que e a parte cara do FAST.
    O T1 do HCP ja vem corrigido pelo pipeline deles; num dataset que nao venha,
    tire o -N (custa varias vezes mais tempo por sujeito).
    """
    out_dir = Path(out_dir)
    # glob e nao um nome fixo: a extensao da saida depende do FSLOUTPUTTYPE
    # (.nii.gz por padrao, .nii se o ambiente pedir).
    existentes = sorted(out_dir.glob("t1_pve_2.nii*"))
    if existentes:
        return existentes[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [fast_bin, "-t", "1", "-n", "3", "-N", "-o", str(out_dir / "t1"), str(t1_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FSL fast falhou em {t1_path} (codigo {proc.returncode}):\n"
            f"{(proc.stderr or proc.stdout).strip()}"
        )

    saidas = sorted(out_dir.glob("t1_pve_2.nii*"))
    if not saidas:
        gerados = sorted(p.name for p in out_dir.glob("t1*"))
        raise RuntimeError(
            f"FSL fast rodou mas nao gerou t1_pve_2 em {out_dir}. Gerados: {gerados}"
        )
    return saidas[0]


class MaskByWM:
    """Zera os voxels FORA da substancia branca, segundo o FSL FAST no T1.

    Mesma ideia do ThresholdByFA — restringir a entrada ao tecido onde o CC pode
    estar — com um criterio ANATOMICO em vez de um limiar de anisotropia. A
    diferenca pratica: a FA tambem e baixa dentro da substancia branca onde ha
    cruzamento de fibras (o limiar de FA come justamente essas regioes), e alta
    em bordas ruidosas fora dela; a segmentacao do T1 nao depende da difusao,
    entao erra em lugares diferentes.

    O limiar e aplicado ao mapa PVE (fracao de branca no voxel, em [0, 1]):
    0.5 fica com o voxel majoritariamente branco, valores menores alargam a
    mascara para incluir a borda parcial.

    A mascara vem da ENTRADA (o T1), nunca do alvo, e o criterio e o mesmo para
    qualquer --input — inclusive um que nem inclua o T1 como canal.

    NOTA: roda DEPOIS da extracao da fatia, pelo mesmo motivo do ThresholdByFA
    (mascarar a entrada nao deveria mudar QUAL fatia e vista). Por isso recorta
    o mapa de WM na MESMA fatia, usando o ctx["slice_idx"] que a extracao deixou.
    """

    def __init__(self, threshold, cache_root=DEFAULT_WM_CACHE, sagittal_axis=SAGITTAL_AXIS):
        self.threshold = threshold
        self.cache_root = Path(cache_root)
        self.sagittal_axis = sagittal_axis

        # Falha agora, e nao no meio do pre-processamento do 40o sujeito.
        self.fast_bin = find_fsl_fast()
        if self.fast_bin is None:
            raise RuntimeError(
                "A mascara de WM precisa do FSL: nao encontrei o executavel `fast` "
                "no PATH nem em $FSLDIR. Instale o FSL (ou rode sem --wm-threshold)."
            )
        print(
            f"Mascara de WM pelo FSL FAST ({self.fast_bin}), cache em {self.cache_root}. "
            f"A primeira passada segmenta cada sujeito e demora ~1 min por sujeito."
        )

    def __call__(self, x, y=None, ctx=None):
        ctx = {} if ctx is None else ctx
        subject_dir = ctx.get("subject_dir")
        if subject_dir is None:
            raise RuntimeError(
                "MaskByWM precisa de ctx['subject_dir'] para achar o T1 do sujeito."
            )

        subject_dir = Path(subject_dir)
        t1 = subject_dir / BrainHack3Data.T1_FILE
        if not t1.exists():
            raise FileNotFoundError(
                f"{t1} nao existe: a mascara de WM depende do T1 do sujeito."
            )

        pve = wm_pve_from_t1(t1, self.cache_root / subject_dir.name, self.fast_bin)
        # O FAST herda o header do T1, e a orientacao do T1 varia de sujeito
        # para sujeito neste dataset: sem o load_las a mascara sai espelhada em
        # relacao a FA e zera o hemisferio errado (ver load_las).
        wm = load_las(pve)

        # A extracao de fatia ja rodou: pega a MESMA fatia do mapa de WM.
        slice_idx = ctx.get("slice_idx")
        if slice_idx is not None:
            wm = np.take(wm, slice_idx, axis=self.sagittal_axis)

        if wm.shape != x.shape[1:]:
            raise RuntimeError(
                f"Mapa de WM {wm.shape} nao bate com a entrada {x.shape[1:]}: o T1 "
                f"precisa estar na mesma grade das metricas de difusao."
            )

        # x e [canal, ...] e a WM e um mapa escalar: o broadcast mascara todos
        # os canais de forma coerente entre si.
        ctx["wm"] = wm
        return (x * (wm >= self.threshold)).astype(np.float32), y


def build_preprocess_3d(sagittal_axis=SAGITTAL_AXIS, fa_threshold=None,
                        wm_threshold=None, wm_cache_dir=DEFAULT_WM_CACHE):
    """Cadeia salva em disco: extrair a fatia e (opcionalmente) mascarar.

    As duas mascaras sao independentes e podem ser combinadas: com as duas
    ligadas sobra a interseccao (branca segundo o T1 E anisotropica segundo a FA).

    A normalizacao nao aparece aqui: cada componente ja saiu normalizado do
    seu proprio loader (ver INPUT_COMPONENTS).
    """
    steps = [ExtractMidSagittalSlice(sagittal_axis=sagittal_axis)]
    if fa_threshold is not None:
        steps.append(ThresholdByFA(fa_threshold))
    if wm_threshold is not None:
        steps.append(MaskByWM(wm_threshold, wm_cache_dir, sagittal_axis))
    return ComposeTransforms(steps)


# ============================================================================
# Pre-processamento: fatias 2D em disco
# ============================================================================


def preprocess(preprocess_fn, data_dir, split_dir, out_root, components=DEFAULT_INPUT, force=False):
    """Extrai uma fatia sagital media por volume e salva em .npz.

    Salvar em disco acelera o treino: o codigo passa a ler fatias 2D ja
    normalizadas em vez do volume NIfTI original.
    """
    for mode in MODES:
        out_dir = Path(out_root) / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset = BrainHack3Data(
            mode,
            data_dir=data_dir,
            split_dir=split_dir,
            components=components,
            transform=preprocess_fn,
        )

        # NOTA: o notebook reprocessa sempre; aqui pulamos quando o split ja
        # esta completo, para poder rodar so o treino sem refazer tudo.
        existing = len(list(out_dir.glob("*.npz")))
        if not force and existing == len(dataset):
            print(f"Preprocess {mode}: {existing} arquivos ja existem, pulando.")
            continue

        for img, tgt, metadata in tqdm(dataset, desc=f"Preprocess {mode}"):
            subject_id = metadata["subject_id"]
            save_path = out_dir / f"{subject_id}.npz"
            np.savez_compressed(save_path, img=img, tgt=tgt)


# ============================================================================
# Dataset 2D, augmentation e DataModule
# ============================================================================


class BrainHack3Data2D(Dataset):
    """Tarefa bidimensional: segmentar o CC na fatia sagital media.

    A entrada pode ter 1 canal (FA) ou 6 canais (componentes de D) — o codigo
    abaixo e o mesmo nos dois casos.
    """

    def __init__(self, mode, processed_dir, transform=None):
        # Indexa os .npz gerados pelo pre-processamento.
        # CUIDADO: a ordem de "glob" nao e deterministica! Por isso o sorted.
        self.dataset = sorted(glob(os.path.join(str(processed_dir), mode, "*.npz")))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        npz = np.load(self.dataset[i])
        # A imagem normalizada e float (com o tensor, inclusive negativa): ler
        # como uint8 zeraria tudo. A mascara continua binaria.
        img = npz["img"].astype(np.float32)  # [canal, altura, largura]
        tgt = npz["tgt"].astype(np.uint8).squeeze()

        # NOTA: o Albumentations trabalha em HWC; os .npz estao em CHW.
        img = np.moveaxis(img, 0, -1)

        if self.transform is not None:
            out = self.transform(image=img, mask=tgt)
            img, tgt = out["image"], out["mask"]

        # Formato esperado pela rede: [canal, altura, largura]
        img = torch.from_numpy(np.ascontiguousarray(np.moveaxis(img, -1, 0))).float()
        tgt = torch.from_numpy(np.ascontiguousarray(tgt)).float().unsqueeze(0)
        return img, tgt


class RotateCropTensor:
    """Rotacao + crop aleatorio consistentes com um campo tensorial.

    Por que nao usar A.Rotate direto: o Albumentations gira o GRID, mas nao tem
    como saber que os 6 canais sao as componentes de um tensor e que elas
    precisam girar junto (D' = R D R^T). Sem isso a rede treina com tensores
    apontando para direcoes que nao existem na imagem — um campo escalar como a
    FA nao tem esse problema, um campo tensorial tem.

    A rotacao acontece no plano (y, z) da fatia sagital, entao a matriz 3D
    correspondente e uma rotacao em torno do eixo x (rotation_matrix_x).
    """

    def __init__(self, limit_deg=10, crop=64, p=0.5, tensor_span=None):
        self.limit_deg = limit_deg
        self.crop = crop
        self.p = p
        # (inicio, fim) dos canais que sao componentes de D, ou None se o
        # --input nao inclui o tensor. Com "--input fa tensor" o tensor vive em
        # (1, 7): girar o bloco errado corromperia a entrada em silencio.
        self.tensor_span = tensor_span

    def __call__(self, image, mask):
        if random.random() < self.p:
            # O angulo e sorteado AQUI (e nao dentro do A.Rotate) para podermos
            # aplicar exatamente a mesma rotacao nas componentes do tensor.
            angle = random.uniform(-self.limit_deg, self.limit_deg)
            out = A.Rotate(limit=(angle, angle), p=1.0)(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
            if self.tensor_span is not None:
                # Verificado empiricamente: A.Rotate(+a) move o conteudo por R.
                # (O Rotate do MONAI, por comparacao, move por R^T — a convencao
                # nao e a mesma entre bibliotecas, entao confira antes de
                # reusar isto em outro lugar.)
                i, j = self.tensor_span
                image = np.ascontiguousarray(image)
                image[..., i:j] = rotate_tensor_channels(
                    image[..., i:j], rotation_matrix_x(np.deg2rad(angle))
                )
        return A.RandomCrop(width=self.crop, height=self.crop, p=1.0)(
            image=image, mask=mask
        )


def get_transform(transform_str: str, components=DEFAULT_INPUT):
    """Factory de transformacoes (None = sem augmentation).

    A string de entrada controla qual objeto de transformada e instanciado, o
    que facilita a reproducibilidade: `transform_str` e um hiperparametro.
    `components` diz onde estao os canais do tensor, quando ha algum.
    """
    if transform_str == "rotate_crop":
        # NOTA: era A.Compose([A.Rotate(...), A.RandomCrop(...)]). Virou uma
        # classe propria porque a rotacao precisa girar tambem as componentes.
        return RotateCropTensor(
            limit_deg=10, crop=64, p=0.5,
            tensor_span=channel_layout(components).get("tensor"),
        )
    if transform_str == "center_crop":
        return A.Compose(
            [
                A.CenterCrop(width=128, height=128),
            ]
        )
    return None


class BrainHack3DataModule(pl.LightningDataModule):
    """Centraliza a criacao dos Datasets e dos DataLoaders.

    DataLoaders iteram sobre os dados de forma eficiente e criam batches. Redes
    convolucionais como a UNet processam batches inteiros de uma vez, em vez de
    aprender de amostra em amostra.
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage=None):
        componentes = tuple(self.hparams.input_components)
        train_t = get_transform(self.hparams.train_transform_str, componentes)
        eval_t = get_transform(self.hparams.eval_transform_str, componentes)
        processed_dir = self.hparams.processed_dir
        self.train = BrainHack3Data2D("train", processed_dir, transform=train_t)
        self.val = BrainHack3Data2D("val", processed_dir, transform=eval_t)
        self.test = BrainHack3Data2D("test", processed_dir, transform=eval_t)

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.nworkers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test, shuffle=False)


# ============================================================================
# Funcao de perda: Dice
# ============================================================================


def dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    """Coeficiente de Dice entre a entrada e o alvo: 2x overlap / uniao.

    Durante o treino podemos calcular o Dice por imagem e tirar a media, ou
    considerar o batch inteiro de uma vez (reduce_batch_first).
    """
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f"Dice: tensor sem batch (shape {input.shape})")

    # Duas dimensoes (ou batch inteiro de uma vez): Dice sobre os valores linearizados.
    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)

    # Mais de duas dimensoes: Dice para cada elemento do batch.
    dice = 0
    for i in range(input.shape[0]):
        dice += dice_coeff(input[i, ...], target[i, ...])

    return dice / input.shape[0]


# ============================================================================
# Arquitetura: UNet
# ============================================================================


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm, reduce, dim):
        """Bloco dinamico 2D/3D: duas convolucoes com batch norm e leaky ReLU.

        `reduce` controla o stride da segunda convolucao, reduzindo a resolucao.
        """
        super().__init__()
        if norm:
            norms = [getattr(nn, f"BatchNorm{dim}")(out_ch) for _ in range(2)]
        else:
            norms = [nn.Identity(), nn.Identity()]

        self.conv = nn.Sequential(
            getattr(nn, f"Conv{dim}")(
                in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False
            ),
            norms[0],
            nn.LeakyReLU(inplace=True),
            getattr(nn, f"Conv{dim}")(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                stride=2 if reduce else 1,
                bias=False,
            ),
            norms[1],
            nn.LeakyReLU(inplace=True),
        )
        self.residual_connection = getattr(nn, f"Conv{dim}")(
            in_ch, out_ch, kernel_size=1, padding=0, stride=2 if reduce else 1, bias=False
        )

    def forward(self, x):
        return self.conv(x) + self.residual_connection(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")
        self.conv = DoubleConv(in_ch, out_ch, norm, reduce=False, dim=dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ajuste de tamanho quando as dimensoes nao batem apos o upsample.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2))
        return self.conv(torch.cat([x2, x1], dim=1))


class UNetEncoder(nn.Module):
    def __init__(self, n_channels, init_channel, norm, dim):
        super().__init__()
        self.inc = DoubleConv(n_channels, init_channel, norm=norm, reduce=False, dim=dim)
        self.down1 = DoubleConv(init_channel, init_channel * 2, norm=norm, reduce=True, dim=dim)
        self.down2 = DoubleConv(init_channel * 2, init_channel * 4, norm=norm, reduce=True, dim=dim)
        self.down3 = DoubleConv(init_channel * 4, init_channel * 8, norm=norm, reduce=True, dim=dim)
        self.down4 = DoubleConv(init_channel * 8, init_channel * 8, norm=norm, reduce=True, dim=dim)

    def forward(self, x):
        out_1 = self.inc(x)
        out_2 = self.down1(out_1)
        out_3 = self.down2(out_2)
        out_4 = self.down3(out_3)
        y = self.down4(out_4)
        return y, out_1, out_2, out_3, out_4


class UNetDecoder(nn.Module):
    def __init__(self, n_classes, init_channel, norm, dim):
        super().__init__()
        self.up1 = Up(16 * init_channel, 4 * init_channel, norm, dim=dim)
        self.up2 = Up(8 * init_channel, 2 * init_channel, norm, dim=dim)
        self.up3 = Up(4 * init_channel, init_channel, norm, dim=dim)
        self.up4 = Up(2 * init_channel, init_channel, norm, dim=dim)
        self.outc = getattr(nn, f"Conv{dim}")(init_channel, n_classes, kernel_size=1, bias=False)

    def forward(self, y, out_1, out_2, out_3, out_4):
        y = self.up1(y, out_4)
        y = self.up2(y, out_3)
        y = self.up3(y, out_2)
        y = self.up4(y, out_1)
        return self.outc(y)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, norm, dim, init_channel):
        super().__init__()
        # Normaliza os canais de ENTRADA entre si. O BatchNorm dos DoubleConv so
        # age depois da primeira convolucao, entao sem isto canais de escalas
        # diferentes (FA em [0, 1] junto com componentes de D concentradas perto
        # de zero) entram desequilibrados na primeira conv.
        #
        # NOTA: BatchNorm e nao InstanceNorm. As normalizacoes dos loaders usam
        # escala FIXA de proposito, para os sujeitos ficarem comparaveis entre si
        # (ver norm_unit/norm_diffusivity); o BatchNorm usa estatisticas do
        # dataset (running stats na validacao) e preserva isso, enquanto um
        # InstanceNorm reescalaria por sujeito e desfaria exatamente essa escolha.
        self.in_norm = (
            getattr(nn, f"BatchNorm{dim}")(n_channels) if norm else nn.Identity()
        )
        self.enc = UNetEncoder(n_channels, init_channel, norm, dim)
        self.dec = UNetDecoder(n_classes, init_channel, norm, dim)
        print(f"UNet: in={n_channels} out={n_classes} dim={dim} init_ch={init_channel}")

    def forward(self, x):
        return self.dec(*self.enc(self.in_norm(x)))


# ============================================================================
# Lightning Module
# ============================================================================


class CCSegmentation(pl.LightningModule):
    """Abstracao do modelo completo: arquitetura + passos de treino/validacao."""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = UNet(
            n_channels=self.hparams.nin,
            n_classes=self.hparams.nout,
            norm=True,
            dim="2d",
            init_channel=self.hparams.init_channels,
        )
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)          # logits, no sigmoid

    def step(self, mode, batch):
        x, y = batch
        logits = self.forward(x)
        probs = torch.sigmoid(logits)

        dice_loss = 1 - dice_coeff(probs, y)
        ce_loss = self.bce(logits, y.float())
        loss = dice_loss + ce_loss

        self.log_dict(
            {f"{mode}_dice_loss": dice_loss, f"{mode}_ce_loss": ce_loss},
            on_epoch=True, on_step=(mode == "train"),
        )
        if mode == "train":
            self.log("loss", loss, on_epoch=True, on_step=True)
            return loss
        else:
            self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            return loss

    '''
    def step(self, mode, batch):
    """Passa o batch pela rede e calcula a perda (1 - Dice).

    O Lightning cuida do otimizador, de mover para a GPU, etc.
    """
    x, y = batch

    y_hat = self.forward(x)

        loss = 1 - dice_coeff(y_hat, y)

        if mode == "train":
            self.log("loss", loss, on_epoch=True, on_step=True)
            return loss
        elif mode == "val":
            self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
    '''

    def training_step(self, batch, batch_idx):
        return self.step("train", batch)

    def validation_step(self, batch, batch_idx):
        self.step("val", batch)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.hparams.lr)


# ============================================================================
# Treino
# ============================================================================


def build_logger(hparams):
    """Logger do Lightning que salva as metricas do experimento.

    NOTA: o notebook usa TensorBoard direto; aqui caimos para CSV quando o
    tensorboard nao esta instalado, para o treino nao morrer por causa do log.
    """
    try:
        return TensorBoardLogger(
            save_dir=hparams["logs_root"], name=hparams["experiment_name"]
        )
    except ModuleNotFoundError:
        print("tensorboard nao instalado: usando CSVLogger (pip install tensorboard).")
        return CSVLogger(save_dir=hparams["logs_root"], name=hparams["experiment_name"])


def train(hparams, data_module, device):
    """Roda o treinamento e devolve o caminho do melhor checkpoint."""
    model = CCSegmentation(hparams)

    experiment_dir = hparams["experiment_dir"]
    os.makedirs(experiment_dir, exist_ok=True)

    logger = build_logger(hparams)

    # Callback que salva o modelo com o menor loss de validacao.
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        devices=1,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        precision=hparams["precision"],
        fast_dev_run=hparams["debug"],
        logger=logger,
        default_root_dir=experiment_dir,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    trainer.fit(model, data_module)

    return checkpoint_callback.best_model_path or None


def find_checkpoint(experiment_dir):
    """Checkpoint mais recente do experimento.

    NOTA: o notebook usa `sorted(glob(...))[-1]`, que ordena por string e
    escolheria `epoch=9` em vez de `epoch=19`. Aqui ordenamos por data.
    """
    ckpts = glob(os.path.join(str(experiment_dir), "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"Nenhum checkpoint em {experiment_dir}: rode o estagio 'train' antes."
        )
    return max(ckpts, key=os.path.getmtime)


# ============================================================================
# Inferencia e visualizacao
# ============================================================================


def load_trained(ckpt, device):
    print(f"Checkpoint: {ckpt}")
    # Modo eval desabilita partes que nao devem rodar fora do treino (ex.: dropout).
    return CCSegmentation.load_from_checkpoint(ckpt, map_location=device).eval().to(device)


def load_pyplot():
    """Importa o matplotlib sob demanda (backend Agg); None se nao estiver instalado."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nao instalado: pulando figuras (pip install matplotlib).")
        return None
    return plt


def to_display(img, components=DEFAULT_INPUT):
    """Mapa escalar para a figura, escolhido pelo layout de canais.

    Com mais de um canal nao existe "a imagem" para mostrar — um img.squeeze()
    daria (C, H, W) e quebraria o imshow. Preferimos a FA (se for um dos
    canais), depois a FA recalculada do tensor, e por ultimo o primeiro canal.
    """
    arr = img.numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    layout = channel_layout(components)
    if "fa" in layout:
        return arr[layout["fa"][0]]
    if "tensor" in layout:
        i, j = layout["tensor"]
        return fa_from_channels(np.moveaxis(arr[i:j], 0, -1))
    if "tensor_inv" in layout:
        # Canal 0 = dxx/tr, a fracao da difusao que atravessa o plano sagital —
        # justamente o que destaca o CC na fatia.
        return arr[layout["tensor_inv"][0]]
    return arr[0]


def paineis(img, tgt, pred, threshold, index, split):
    """Os quatro paineis de uma amostra: (array, titulo) na ordem de leitura.

    O quarto e a predicao ja BINARIZADA no threshold escolhido — que e a
    predicao de fato, a que as metricas medem. Os outros tres sao os do
    notebook: entrada, alvo e a predicao continua antes do corte.
    """
    return [
        (img, f"FA (entrada), {split} {index}"),
        (tgt, "CC (alvo)"),
        (pred, "Predicao (continua)"),
        # `>` e nao `>=`, o mesmo criterio da compute_metrics: a figura tem que
        # mostrar exatamente os voxels que entraram no Dice.
        ((pred > threshold).astype(np.float32), f"Predicao (threshold={threshold:.2f})"),
    ]


def save_panels(plt, img, tgt, pred, threshold, index, split, out_dir):
    """Salva os quatro paineis de UMA amostra, lado a lado."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colunas = paineis(img, tgt, pred, threshold, index, split)
    plt.figure(figsize=(4 * len(colunas), 4))
    for j, (arr, title) in enumerate(colunas):
        plt.subplot(1, len(colunas), j + 1)
        plt.imshow(arr, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{split}_{index:03d}.png", dpi=110)
    plt.close()


def save_grid(plt, amostras, threshold, split, out_dir):
    """Salva as amostras num UNICO PNG: uma linha por amostra, uma coluna por painel.

    E o bloco de saida do notebook inteiro em um arquivo so, empilhado, para
    comparar os sujeitos de uma olhada em vez de abrir um PNG por vez.

    `amostras` e uma lista de (img, tgt, pred, index), na ordem do split.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_colunas = len(paineis(*amostras[0], threshold, split))
    # squeeze=False para axes ser sempre 2D, inclusive com uma amostra so.
    # A altura acompanha o numero de linhas (4 polegadas por painel, a mesma
    # escala da save_panels), senao a grade achata as imagens.
    fig, axes = plt.subplots(
        len(amostras), n_colunas,
        figsize=(4 * n_colunas, 4 * len(amostras)), squeeze=False,
    )

    for linha, (img, tgt, pred, index) in enumerate(amostras):
        for coluna, (arr, title) in enumerate(
            paineis(img, tgt, pred, threshold, index, split)
        ):
            ax = axes[linha][coluna]
            ax.imshow(arr, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

    fig.tight_layout()
    caminho = out_dir / f"{split}_grid.png"
    fig.savefig(caminho, dpi=110)
    plt.close(fig)
    print(f"Figura combinada de {split}: {caminho}")


def save_figures(dataset, tgts_np, preds_np, threshold, split, figures_dir,
                 max_figures=6, components=DEFAULT_INPUT):
    """Desenha as figuras do split, no threshold ja escolhido.

    NOTA: roda DEPOIS da varredura, e nao dentro da predict_split. O threshold
    da validacao so existe depois que a varredura terminou; desenhando junto com
    a inferencia, o painel binarizado da validacao teria que usar um limiar
    provisorio — diferente do que as metricas dela reportam.

    Reabre as `max_figures` primeiras amostras em vez de guardar o split inteiro
    na memoria. A transformada de avaliacao e deterministica (center crop), entao
    a imagem relida e a mesma que gerou a predicao.
    """
    if figures_dir is None:
        return
    plt = load_pyplot()  # se o matplotlib faltar, as figuras ficam desligadas
    if plt is None:
        return

    amostras = []
    for i in range(min(max_figures, len(dataset))):
        img, _ = dataset[i]
        display = to_display(img, components)
        save_panels(plt, display, tgts_np[i], preds_np[i], threshold, i, split, figures_dir)
        amostras.append((display, tgts_np[i], preds_np[i], i))

    if amostras:
        save_grid(plt, amostras, threshold, split, figures_dir)


def predict_split(model, dataset, device, split):
    """Roda a rede em todo o split e devolve (alvos, predicoes) como numpy."""
    tgts_np, preds_np = [], []

    for i in range(len(dataset)):
        img, tgt = dataset[i]

        # torch.no_grad() desabilita gradientes (so necessarios no treino):
        # economiza memoria e tempo.
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).cpu().squeeze().numpy()

        tgts_np.append(tgt.squeeze().numpy())
        preds_np.append(pred)

    print(f"{len(dataset)} amostras de {split} inferidas.")
    return tgts_np, preds_np


# ============================================================================
# Metricas
# ============================================================================

# Nomes legiveis das metricas coletadas por seg_metrics (so para exibicao: a
# ordem de impressao vem da ordem de insercao em seg_metrics).
METRIC_LABELS = {
    "dice": "Dice",
    "jaccard": "Jaccard",
    "hd": "Hausdorff",
}


def seg_metrics(gts, preds, metrics, struct_names=("cc",)):
    """Compara mascaras binarias com SimpleITK: Dice, Jaccard e Hausdorff."""
    for gt, pred, label in zip(gts, preds, struct_names):
        # O sitk implementa filtros que fornecem multiplas metricas.
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        hausdorff = sitk.HausdorffDistanceImageFilter()

        # Converte numpy para o formato do sitk.
        gt_img = sitk.GetImageFromArray(gt)
        pred_img = sitk.GetImageFromArray(pred)

        overlap.Execute(gt_img, pred_img)

        # Salva metricas em dicionario dinamico.
        metrics[label]["dice"].append(overlap.GetDiceCoefficient())
        metrics[label]["jaccard"].append(overlap.GetJaccardCoefficient())
        try:
            hausdorff.Execute(gt_img, pred_img)
            metrics[label]["hd"].append(hausdorff.GetHausdorffDistance())
        except Exception:
            metrics[label]["hd"].append(nan)


def compute_metrics(tgts_np, preds_np, threshold):
    """Binariza as predicoes com `threshold` e acumula as metricas por estrutura."""
    metrics = defaultdict(lambda: defaultdict(list))
    for tgt_np, pred in zip(tgts_np, preds_np):
        gt_u8 = (tgt_np > 0).astype(np.uint8)
        pred_u8 = (pred > threshold).astype(np.uint8)
        seg_metrics(gt_u8[None], pred_u8[None], metrics, struct_names=["cc"])
    return metrics


def metric_label(key):
    """Nome legivel da metrica; chaves desconhecidas aparecem como estao."""
    return METRIC_LABELS.get(key, key)


def metric_means(metrics, label="cc"):
    """Media de cada metrica coletada, ignorando nan (ex.: Hausdorff que falhou)."""
    means = {}
    for key, values in metrics[label].items():
        v = np.asarray(values, dtype=float)
        means[key] = nan if np.all(np.isnan(v)) else float(np.nanmean(v))
    return means


def report(metrics, threshold, split, label="cc"):
    """Imprime media, desvio, minimo e maximo de TODAS as metricas coletadas."""
    per_metric = metrics[label]
    if not per_metric:
        print(f"\n[{split}] threshold={threshold:.2f}  sem metricas (split vazio?)")
        return

    n = len(next(iter(per_metric.values())))
    print(f"\n[{split}] threshold={threshold:.2f}  n={n} amostras")

    for key, values in per_metric.items():
        v = np.asarray(values, dtype=float)
        n_nan = int(np.isnan(v).sum())
        if n_nan == len(v):
            print(f"    {metric_label(key):<10} sem valores validos ({n_nan} nan)")
            continue
        aviso = f"  ({n_nan} nan)" if n_nan else ""
        print(
            f"    {metric_label(key):<10}"
            f" media={np.nanmean(v):9.4f}"
            f"  dp={np.nanstd(v):8.4f}"
            f"  min={np.nanmin(v):9.4f}"
            f"  max={np.nanmax(v):9.4f}{aviso}"
        )


def sweep_threshold(tgts_np, preds_np, thresholds, label="cc"):
    """TAREFA do notebook: qual o melhor limiar de binarizacao na validacao?

    A rede produz valores continuos em [0, 1]; um limiar muito baixo gera falso
    positivo, um muito alto gera falso negativo. A escolha usa o Dice, mas a
    tabela mostra todas as metricas para o limiar ser julgado por inteiro.

    Devolve (melhor threshold, metricas ja calculadas nesse threshold), para o
    chamador imprimir o detalhamento sem repetir a conta do SimpleITK.
    """
    rows = [
        (float(th), compute_metrics(tgts_np, preds_np, th)) for th in thresholds
    ]
    means = [(th, metric_means(metrics, label)) for th, metrics in rows]

    keys = list(means[0][1])
    print("\nVarredura de threshold na validacao (media de cada metrica):")
    header = "  threshold" + "".join(f"  {metric_label(k):>12}" for k in keys)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for th, row_means in means:
        print(f"  {th:>9.2f}" + "".join(f"  {row_means[k]:>12.4f}" for k in keys))

    best_i = max(range(len(means)), key=lambda i: means[i][1]["dice"])
    best_th, best_means = means[best_i]
    resumo = "  ".join(f"{metric_label(k)}={best_means[k]:.4f}" for k in keys)
    print(f"\nMelhor threshold: {best_th:.2f}  ({resumo})")
    return best_th, rows[best_i][1]


# ============================================================================
# Registro dos resultados
# ============================================================================
#
# O mesmo relatorio que aparece na tela vai para <experimento>/results.txt, no
# MESMO formato de bloco que o brainhack_challenge_tests.py usa no
# test_results.txt (comando, resultados, separador). Assim o resultado fica
# junto do que o produziu — checkpoint, curvas do TensorBoard e figuras — e uma
# rodada solta, fora da bateria, tambem deixa registro.

RESULTS_FILENAME = "results.txt"
SEPARADOR = "-" * 79

# Onde o relatorio comeca: os mesmos marcadores que o test_results.txt usa. Sem
# varredura (--no-threshold-sweep) o primeiro e o relatorio da validacao.
RESULT_MARKERS = (
    "Varredura de threshold na validacao",
    "[val] threshold=",
)


class TeeText(io.TextIOBase):
    """Repassa o texto para a tela e guarda uma copia.

    O relatorio precisa ir aos dois lugares: continuar aparecendo ao vivo (a
    bateria de testes le a saida do processo) e virar arquivo no fim.
    """

    def __init__(self, destino):
        self.destino = destino
        self.pedacos = []

    def write(self, texto):
        self.pedacos.append(texto)
        return self.destino.write(texto)

    def flush(self):
        self.destino.flush()

    def getvalue(self):
        return "".join(self.pedacos)


def trim_to_report(texto):
    """Recorta o trecho de resultados, como o brainhack_challenge_tests.py faz.

    Sem isto o bloco comecaria em linhas de servico (checkpoint carregado,
    caminho das figuras) que nao sao resultado. Se nenhum marcador aparece
    (ex.: so `--stages test`), guarda o que houver.
    """
    linhas = texto.splitlines()
    for i, linha in enumerate(linhas):
        if any(linha.lstrip().startswith(m) for m in RESULT_MARKERS):
            return "\n".join(linhas[i:]).strip("\n")
    return texto.strip("\n")


def save_results(texto, experiment_dir, argv=None):
    """Anexa o bloco de resultados ao results.txt do experimento.

    Anexa, nunca sobrescreve: rodar o eval de novo acrescenta um bloco em vez de
    apagar o anterior, e os blocos ficam em ordem cronologica.
    """
    resultados = trim_to_report(texto)
    if not resultados:
        return None

    argv = sys.argv[1:] if argv is None else list(argv)
    comando = "python " + shlex.join([Path(sys.argv[0]).name, *argv])

    destino = Path(experiment_dir)
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / RESULTS_FILENAME
    with open(caminho, "a") as f:
        f.write(
            f"Input command:\n\n{comando}\n\n"
            f"Results:\n\n{resultados}\n\n"
            f"{SEPARADOR}\n\n"
        )
    return caminho


# ============================================================================
# CLI
# ============================================================================


def build_hparams(args, processed_dir):
    hparams = {
        "experiment_name": args.experiment_name,
        "train_transform_str": args.train_transform,
        "eval_transform_str": args.eval_transform,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "nworkers": args.workers,
        "input_components": list(args.input),
        "fa_threshold": args.fa_threshold,
        "wm_threshold": args.wm_threshold,
        # nin acompanha a entrada escolhida: 1 (FA) ou 6 (componentes de D).
        "nin": n_input_channels(args.input),
        "nout": 1,
        "lr": args.lr,
        "precision": args.precision,
        "debug": args.debug,
        "init_channels": args.init_channels,
        "processed_dir": str(processed_dir),
        "logs_root": str(args.logs_root),
    }
    hparams["experiment_dir"] = os.path.join(
        str(args.logs_root), hparams["experiment_name"]
    )
    return hparams


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Segmentacao do corpo caloso na fatia sagital media (BrainHack 3.0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stages",
        nargs="+",
        choices=[*STAGES, "all"],
        default=["all"],
        help="Etapas a executar; rodam sempre na ordem "
        + " -> ".join(STAGES)
        + ", independente da ordem digitada.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Pasta que contem as subpastas de sujeito. Se omitido, e descoberta em --search-root.",
    )
    p.add_argument(
        "--search-root",
        default=str(REPO_ROOT),
        help="Onde procurar as pastas de sujeito quando --data-dir nao e informado.",
    )
    p.add_argument(
        "--split-dir",
        default=str(REPO_ROOT),
        help="Pasta com train.json, val.json e test.json.",
    )
    p.add_argument(
        "--input",
        nargs="+",
        default=list(DEFAULT_INPUT),
        choices=sorted(INPUT_COMPONENTS),
        metavar="COMPONENTE",
        help="Um ou mais componentes, concatenados como canais na ordem dada. "
        + "Disponiveis: "
        + "; ".join(
            f"{k} ({v.channels}ch, {v.descricao})"
            for k, v in sorted(INPUT_COMPONENTS.items())
        )
        + ". Ex.: --input fa tensor",
    )
    p.add_argument(
        "--fa-threshold",
        type=float,
        default=None,
        help="Se informado, zera na entrada os voxels com FA abaixo deste valor "
        "(ex.: 0.2). Vale para os dois modos de --input.",
    )
    p.add_argument(
        "--wm-threshold",
        type=float,
        default=None,
        help="Se informado, zera na entrada os voxels fora da substancia branca "
        "(ex.: 0.5). A mascara sai do FSL FAST sobre o T1 do sujeito, no mapa "
        "PVE de branca; exige o FSL instalado. Combinavel com --fa-threshold.",
    )
    p.add_argument(
        "--wm-cache-dir",
        default=DEFAULT_WM_CACHE,
        help="Onde guardar as segmentacoes do FSL FAST (uma por sujeito, "
        "reaproveitada entre experimentos).",
    )
    p.add_argument(
        "--processed-dir",
        default=None,
        help="Saida das fatias .npz (padrao: depende de --input).",
    )
    p.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT, help="Raiz dos logs/checkpoints.")
    p.add_argument(
        "--experiment-name",
        default=None,
        help=f"Padrao: {DEFAULT_EXPERIMENT}_<input>.",
    )
    p.add_argument("--force-preprocess", action="store_true", help="Reprocessa mesmo se os .npz existirem.")
    p.add_argument("--sagittal-axis", type=int, default=SAGITTAL_AXIS)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--init-channels", type=int, default=32)
    p.add_argument("--precision", default="32")
    p.add_argument("--train-transform", default="rotate_crop")
    p.add_argument("--eval-transform", default="center_crop")
    p.add_argument("--debug", action="store_true", help="fast_dev_run: 1 batch de treino/validacao.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--threshold", type=float, default=0.5, help="Limiar usado quando o estagio 'eval' nao roda.")
    p.add_argument(
        "--no-threshold-sweep",
        action="store_true",
        help="Nao varre thresholds na validacao; usa --threshold.",
    )
    p.add_argument(
        "--figures-dir",
        default=None,
        help="Onde salvar os PNGs das predicoes. Padrao: <logs-root>/<experimento>/figures.",
    )
    p.add_argument("--no-figures", action="store_true", help="Nao salva figura nenhuma.")
    p.add_argument("--max-figures", type=int, default=6)
    p.add_argument("--checkpoint", default=None, help="Checkpoint para eval/test (padrao: o mais recente).")

    args = p.parse_args(argv)
    pedidos = set(STAGES) if "all" in args.stages else set(args.stages)
    # Ordem canonica: "--stages test eval" nao pode avaliar antes de escolher o threshold.
    args.stages = [stage for stage in STAGES if stage in pedidos]
    # precision aceita "32", "16-mixed", "bf16-mixed", ...
    if args.precision.isdigit():
        args.precision = int(args.precision)
    # Defaults que dependem de --input: pasta dos .npz e nome do experimento.
    # Separados por entrada para que trocar --input nao reaproveite fatias nem
    # checkpoints da entrada anterior (o numero de canais nem bate). O limiar
    # entra no nome pelo mesmo motivo: ele muda o conteudo dos .npz sem mudar a
    # contagem de arquivos, e o preprocess pula quando a contagem ja bate.
    args.input = normalize_components(args.input)
    sufixo = input_tag(args.input)
    if args.fa_threshold is not None:
        sufixo += f"_th{args.fa_threshold:g}"
    if args.wm_threshold is not None:
        sufixo += f"_wm{args.wm_threshold:g}"
    if args.processed_dir is None:
        args.processed_dir = f"preprocessed_cc_{sufixo}"
    if args.experiment_name is None:
        args.experiment_name = f"{DEFAULT_EXPERIMENT}_{sufixo}"

    # NOTA: as figuras vao, por padrao, para dentro do diretorio do experimento
    # — ao lado dos checkpoints e dos eventos do TensorBoard. Assim toda rodada
    # deixa registrado o que ela previu, sem depender de lembrar da flag, e as
    # figuras de entradas diferentes nao se misturam (o nome do experimento ja
    # carrega o --input e as mascaras).
    if args.no_figures:
        args.figures_dir = None
    elif args.figures_dir is None:
        args.figures_dir = str(Path(args.logs_root) / args.experiment_name / "figures")
    return args


def main(argv=None):
    args = parse_args(argv)

    # seed_everything ja semeia random, numpy e torch.
    pl.seed_everything(args.seed, workers=True)

    processed_dir = Path(args.processed_dir)
    hparams = build_hparams(args, processed_dir)

    print("Hiperparametros:")
    for k, v in hparams.items():
        print(f"  {k}: {v}")

    # ---- 1. Pre-processamento -------------------------------------------
    if "preprocess" in args.stages:
        if args.data_dir:
            data_dir = Path(args.data_dir)
        else:
            data_dir = find_data_root(
                Path(args.search_root),
                BrainHack3Data.required_files(
                    args.input, wm_mask=args.wm_threshold is not None
                ),
            )
        print(f"\nDATA_DIR: {data_dir}")
        preprocess(
            build_preprocess_3d(
                args.sagittal_axis, args.fa_threshold,
                args.wm_threshold, args.wm_cache_dir,
            ),
            data_dir=data_dir,
            split_dir=args.split_dir,
            out_root=processed_dir,
            components=args.input,
            force=args.force_preprocess,
        )

    # Os estagios que leem as fatias usam sempre os datasets do DataModule, para
    # a avaliacao nao poder divergir do pre-processamento visto no treino.
    stages_com_dados = [s for s in args.stages if s != "preprocess"]
    if not stages_com_dados:
        return

    data_module = BrainHack3DataModule(hparams)
    data_module.setup()

    # Sanidade: conferir os splits antes do treino para detectar erros cedo.
    for mode in MODES:
        n = len(getattr(data_module, mode))
        if n == 0:
            raise RuntimeError(
                f"Split '{mode}' vazio em {processed_dir}: rode o estagio 'preprocess'."
            )
        print(f"{mode}: {n} amostras")

    # Os .npz nao guardam qual entrada os gerou. Sem esta checagem, apontar
    # --processed-dir para a pasta da outra entrada so falharia la na frente,
    # como um erro de shape no primeiro conv.
    canais = data_module.train[0][0].shape[0]
    if canais != hparams["nin"]:
        raise RuntimeError(
            f"{processed_dir} tem {canais} canais, mas --input {input_tag(args.input)} espera "
            f"{hparams['nin']}. Rode 'preprocess' (com --force-preprocess se a pasta "
            f"ja existir) ou aponte --processed-dir para a pasta certa."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 2. Treino -------------------------------------------------------
    best_ckpt = None
    if "train" in args.stages:
        best_ckpt = train(hparams, data_module, device)

    if not {"eval", "test"} & set(args.stages):
        return

    if "train" in args.stages and best_ckpt is None and not args.checkpoint:
        # Deriva do fato (o treino nao salvou nada, tipicamente fast_dev_run) em
        # vez de reinterpretar --debug, para nao engolir um --stages eval avulso.
        print("\nTreino nao gerou checkpoint (--debug?): pulando eval/test.")
        return

    ckpt = args.checkpoint or best_ckpt or find_checkpoint(hparams["experiment_dir"])
    trained = load_trained(ckpt, device)

    # Daqui para baixo a saida vai para a tela E para uma copia, que no fim
    # vira o results.txt do experimento (ver save_results). O tqdm escreve em
    # stderr, entao as barras de progresso nao entram na copia.
    tee = TeeText(sys.stdout)
    with contextlib.redirect_stdout(tee):
        run_eval_and_test(trained, data_module, device, args, threshold=args.threshold)

    caminho = save_results(tee.getvalue(), hparams["experiment_dir"])
    if caminho is not None:
        print(f"\nResultados: {caminho}")


def run_eval_and_test(trained, data_module, device, args, threshold):
    """Estagios 3 e 4: validacao (com escolha do threshold) e teste."""
    # ---- 3. Validacao: predicoes, escolha do threshold e figuras ---------
    if "eval" in args.stages:
        tgts_np, preds_np = predict_split(trained, data_module.val, device, "val")
        if args.no_threshold_sweep:
            metrics = compute_metrics(tgts_np, preds_np, threshold)
        else:
            threshold, metrics = sweep_threshold(
                tgts_np, preds_np, np.arange(0.05, 1.0, 0.05)
            )
        report(metrics, threshold, "val")
        save_figures(data_module.val, tgts_np, preds_np, threshold, "val",
                     args.figures_dir, args.max_figures, args.input)

    # ---- 4. Avaliacao final no teste -------------------------------------
    if "test" in args.stages:
        tgts_np, preds_np = predict_split(trained, data_module.test, device, "test")
        report(compute_metrics(tgts_np, preds_np, threshold), threshold, "test")
        save_figures(data_module.test, tgts_np, preds_np, threshold, "test",
                     args.figures_dir, args.max_figures, args.input)


if __name__ == "__main__":
    main()
