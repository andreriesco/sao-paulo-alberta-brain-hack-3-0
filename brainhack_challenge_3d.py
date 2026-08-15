#!/usr/bin/env python3
"""brainhack_challenge_3d — versao 3D do script do hands-on (BrainHack 3.0).

Segmentacao do corpo caloso no VOLUME inteiro com uma UNet 3D treinada com
PyTorch Lightning. E a irma de `brainhack_challenge.py`, que resolve a mesma
tarefa numa unica fatia sagital media com uma UNet 2D; tudo que difere entre as
duas esta marcado com `# NOTA 3D:`.

O que muda, em uma frase cada:

  * o pre-processamento nao extrai mais uma fatia: recorta o volume para
    VOLUME_SIZE^3 = 128^3 (ver CropVolume);
  * o treino ve PATCHES de PATCH_SIZE^3 = 64^3 sorteados do volume (ver
    RotateCropTensor3D), porque o volume inteiro com largura de canal decente
    nao cabe numa GPU pequena; a avaliacao roda no volume 128^3 inteiro, que a
    rede aceita por ser totalmente convolucional;
  * a augmentation gira em torno do eixo sagital, como na versao 2D — la o giro
    era no plano da fatia, aqui e o mesmo giro aplicado a todas as fatias de uma
    vez, entao a correcao do campo tensorial (D -> R D R^T) vale palavra por
    palavra;
  * a UNet e a mesma classe com dim="3d": os blocos ja eram genericos, so o
    upsample e o ajuste de tamanho precisavam saber o numero de eixos.

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
`RotateCropTensor3D` (um campo tensorial nao gira como um campo escalar).

`tensor_inv` e a alternativa ao `tensor`: os mesmos 6 numeros reescritos como 5
invariantes sob a rotacao que a augmentation aplica (ver
`tensor_invariants_from_channels`). Carrega a orientacao das fibras, que as
metricas escalares jogam fora, mas sem depender do referencial, que e o que
obriga o `tensor` a girar junto com a imagem.

Os estagios sao os mesmos da versao 2D:

    1. preprocess : le os volumes 3D (entrada + mascara CC), normaliza, recorta
                    para 128^3 e salva um .npz por sujeito.
    2. train      : treina a UNet 3D em patches de 64^3.
    3. eval       : inferencia na validacao (volume inteiro) + varredura de
                    threshold.
    4. test       : inferencia no teste com o melhor threshold.

Uso tipico (tudo de uma vez, dados descobertos automaticamente):

    python brainhack_challenge_3d.py

Combinando metricas (7 canais: 1 de FA + 6 do tensor):

    python brainhack_challenge_3d.py --input fa tensor

Somente treino, com os .npz ja gerados:

    python brainhack_challenge_3d.py --stages train --epochs 50

Smoke test rapido (1 batch de treino e 1 de validacao):

    python brainhack_challenge_3d.py --debug

Custo, para dimensionar antes de rodar: cada .npz guarda 128^3 voxels em
float32 por canal (~8 MB por canal por sujeito antes da compressao), e a
memoria de GPU do treino e governada por --patch-size, --batch-size e
--init-channels, nessa ordem. Se faltar memoria, o primeiro knob e
`--precision 16-mixed`, depois --init-channels.

Diferencas em relacao ao notebook estao marcadas com `# NOTA:`, e as que
existem so por causa do 3D com `# NOTA 3D:`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from glob import glob
from math import nan
from pathlib import Path

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
# NOTA 3D: sai o Albumentations, que so trabalha em imagens 2D (HWC), e entra o
# scipy.ndimage para girar o volume em torno do eixo sagital.
from scipy.ndimage import rotate as ndimage_rotate
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# Configuracao
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_LOGS_ROOT = "logs"
# NOTA 3D: nome proprio para os checkpoints 3D nao caírem na mesma pasta de
# experimento dos 2D — o numero de canais bate, mas a arquitetura nao.
DEFAULT_EXPERIMENT = "BrainhackCC3D"
# Eixo sagital dos volumes (X). E em torno dele que a augmentation gira e que a
# correcao do campo tensorial e definida (ver rotation_matrix_x).
SAGITTAL_AXIS = 0
MODES = ("train", "val", "test")

# NOTA 3D: lado do cubo que sai do pre-processamento. Duas razoes para recortar:
# 145x174x145 nao e multiplo de 16 (a UNet reduz 4 vezes por 2, e um lado que
# nao seja multiplo de 16 volta do decoder com tamanho diferente do skip), e o
# volume inteiro na resolucao original nao cabe na GPU.
VOLUME_SIZE = 128
# Lado do patch de treino. A rede e totalmente convolucional, entao treinar em
# patch e avaliar no volume inteiro e a mesma rede — o patch existe so para o
# treino caber na memoria.
PATCH_SIZE = 16
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
    gira o volume no plano (y, z), ou seja D -> R D R^T com R = rotation_matrix_x,
    e ai as componentes se misturam — a rede precisa aprender essa invariancia
    pelos exemplos. Estes 5 canais ja sao invariantes por construcao.

    Escrevendo D em blocos, com c = (dxy, dxz) e B = [[dyy, dyz], [dyz, dzz]],
    uma rotacao em torno de x fixa dxx, gira c como um vetor 2D e leva B em
    R2 B R2^T. Logo dxx, tr D, |c|, det B e o quociente de Rayleigh
    c^T B c / |c|^2 nao mudam. Sao 5 funcoes independentes = os 6 graus de
    liberdade de D menos o parametro da rotacao, entao nao se perde nada alem
    da fase arbitraria.

    Por que dxx importa aqui: em torno da linha media as fibras do CC a cruzam,
    entao a direcao principal aponta ao longo do eixo x. As outras estruturas de
    FA alta da regiao (fornix, cingulo) correm no plano sagital. O canal dxx/tr
    separa justamente esses casos — e nenhuma metrica escalar (fa, md, ad, rd)
    consegue, porque todas dependem so dos autovalores.

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
    """Rotacao em torno do eixo sagital (eixo 0) = giro no plano (y, z)."""
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


def _scalar_loader(filename, normalize):
    """Componente de 1 canal lido de um NIfTI escalar."""

    def load(subject_dir):
        vol = nib.load(Path(subject_dir) / filename).get_fdata(dtype=np.float32)
        return normalize(np.expand_dims(np.nan_to_num(vol), 0))

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
    plano sagital, num referencial que a augmentation nao mexe. Ver
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
                                 "5 invariantes de D sob a rotacao do plano sagital"),
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

    @classmethod
    def required_files(cls, components):
        """NIfTI que uma pasta precisa ter para ser reconhecida como sujeito.

        A FA entra sempre: mesmo quando nao e canal de entrada, e ela que vai no
        `ctx` como criterio das transformadas (ver ThresholdByFA).
        """
        arquivos = {cls.FA_FILE, cls.CC_FILE}
        for name in components:
            arquivos.update(INPUT_COMPONENTS[name].files)
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
        fa = nib.load(subject_dir / BrainHack3Data.FA_FILE).get_fdata(dtype=np.float32)
        return np.nan_to_num(fa)

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
            # dependem da FA (mascarar por limiar) teriam que adivinhar onde ela
            # esta nos canais — e nao existiria criterio nenhum para uma entrada
            # como "--input md rd".
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
    nem entrada nem alvo — hoje a FA, que serve de criterio para mascarar por
    limiar. As transformadas podem reescrever o ctx (o recorte troca a FA pela
    FA recortada), de modo que ele sempre acompanha o x atual.
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


class CropVolume:
    """Recorta o volume, centrado, para um cubo de lado `size`.

    NOTA 3D: e o que substitui a ExtractMidSagittalSlice da versao 2D — em vez
    de escolher UMA fatia, o pipeline 3D fica com o volume todo, so aparado.

    Center crop, e nao um recorte guiado pelo alvo: a mascara de CC e o que a
    rede deve prever, e num volume novo ela nem existe. Medido nos 144 sujeitos
    dos tres splits, o CC ocupa [47..96] x [46..133] x [46..91] e o crop 128^3
    de um volume 145x174x145 comeca em (8, 23, 8) — sobram pelo menos 17 voxels
    de folga de cada lado, entao o recorte nao corta alvo de nenhum sujeito.

    Volumes menores que `size` sao completados com zero, para um dataset com
    outra grade falhar de forma visivel em vez de virar um erro de shape la no
    meio do treino.
    """

    def __init__(self, size=VOLUME_SIZE):
        self.size = int(size)

    def _crop_pad(self, n):
        """(fatia, padding) de um eixo de tamanho n para chegar em self.size."""
        if n >= self.size:
            inicio = (n - self.size) // 2
            return slice(inicio, inicio + self.size), (0, 0)
        falta = self.size - n
        return slice(0, n), (falta // 2, falta - falta // 2)

    def apply(self, arr, com_canal=True):
        """Recorta os eixos ESPACIAIS de arr, preservando o eixo de canal."""
        arr = np.asarray(arr)
        espacial = arr.shape[1:] if com_canal else arr.shape
        cortes, paddings = zip(*(self._crop_pad(n) for n in espacial))
        if com_canal:
            arr = arr[(slice(None), *cortes)]
            paddings = ((0, 0), *paddings)
        else:
            arr = arr[tuple(cortes)]
        return np.pad(arr, paddings) if any(p != (0, 0) for p in paddings) else arr

    def __call__(self, x, y=None, ctx=None):
        ctx = {} if ctx is None else ctx
        x = self.apply(x).astype(np.float32)

        fa_vol = ctx.get("fa")
        if fa_vol is not None:
            # O ctx acompanha o x: daqui para frente "fa" e a FA recortada, para
            # a ThresholdByFA seguinte poder fazer broadcast com os canais.
            ctx["fa"] = self.apply(fa_vol, com_canal=False)

        if y is None:
            return x, None  # inferencia: nao ha mascara a recortar
        return x, (self.apply(y) > 0).astype(np.float32)


class ThresholdByFA:
    """Zera os voxels cuja FA fica ABAIXO de `threshold`, mantendo o resto.

    A mascara vem da ENTRADA, nunca do alvo, e sempre da FA do `ctx` — entao o
    mesmo limiar seleciona exatamente os mesmos voxels seja qual for o
    --input, inclusive num que nem inclua a FA como canal. Como FA e
    adimensional e vive em [0, 1], o limiar e diretamente interpretavel (~0.2
    costuma separar substancia branca do resto).

    NOTA 3D: roda DEPOIS do recorte, de proposito — o `fa` do ctx so tem o mesmo
    shape dos canais depois que a CropVolume aparou os dois.
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


def build_preprocess_volume(volume_size=VOLUME_SIZE, fa_threshold=None):
    """Cadeia salva em disco: recortar o volume e (opcionalmente) mascarar.

    A normalizacao nao aparece aqui: cada componente ja saiu normalizado do
    seu proprio loader (ver INPUT_COMPONENTS).
    """
    steps = [CropVolume(volume_size)]
    if fa_threshold is not None:
        steps.append(ThresholdByFA(fa_threshold))
    return ComposeTransforms(steps)


# ============================================================================
# Pre-processamento: volumes recortados em disco
# ============================================================================


def preprocess(preprocess_fn, data_dir, split_dir, out_root, components=DEFAULT_INPUT, force=False):
    """Recorta um volume por sujeito e salva em .npz.

    Salvar em disco acelera o treino: o codigo passa a ler volumes ja
    normalizados e recortados em vez de remontar o tensor a cada epoca.
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
# Dataset 3D, augmentation e DataModule
# ============================================================================


class BrainHack3DataVolume(Dataset):
    """Tarefa tridimensional: segmentar o CC no volume recortado.

    NOTA 3D: substitui a BrainHack3Data2D. Como o Albumentations saiu, os
    arrays ficam em [canal, X, Y, Z] do disco ate a rede, sem a ida e volta
    para HWC que a versao 2D precisava fazer.

    A entrada pode ter 1 canal (FA) ou varios (componentes de D, combinacoes) —
    o codigo abaixo e o mesmo em todos os casos.
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
        img = npz["img"].astype(np.float32)  # [canal, X, Y, Z]
        tgt = npz["tgt"].astype(np.float32)  # [1, X, Y, Z]

        if self.transform is not None:
            img, tgt = self.transform(img, tgt)

        return (
            torch.from_numpy(np.ascontiguousarray(img)).float(),
            torch.from_numpy(np.ascontiguousarray(tgt)).float(),
        )


class RotateCropTensor3D:
    """Rotacao + crop aleatorio 3D consistentes com um campo tensorial.

    Mesma ideia da RotateCropTensor 2D, um eixo acima. A rotacao continua sendo
    em torno do eixo sagital — o que na versao 2D era girar a fatia no plano
    (y, z) e aqui e girar TODAS as fatias pelo mesmo angulo, que e exatamente o
    que o scipy.ndimage.rotate faz com axes=(y, z). Por isso a correcao do campo
    tensorial e a mesma de la, letra por letra: D' = R D R^T com
    R = rotation_matrix_x. Sem ela a rede treinaria com tensores apontando para
    direcoes que nao existem na imagem.

    Verificado empiricamente, como foi feito para o Albumentations: com
    axes=(y, z), rotate(+a) move o conteudo por R — a MESMA convencao do
    A.Rotate(+a). (O Rotate do MONAI, por comparacao, move por R^T; a convencao
    nao e a mesma entre bibliotecas, entao confira antes de reusar isto.)

    NOTA 3D: o crop e enviesado para o alvo (`fg_prob`). Um patch de 64^3
    sorteado uniformemente de um volume 128^3 quase nunca pega o CC, que ocupa
    ~1% dos voxels, e o treino gastaria quase todos os passos em patches vazios.
    """

    def __init__(self, patch=PATCH_SIZE, limit_deg=10, p=0.5, fg_prob=0.5,
                 tensor_span=None):
        self.patch = int(patch)
        self.limit_deg = limit_deg
        self.p = p
        self.fg_prob = fg_prob
        # (inicio, fim) dos canais que sao componentes de D, ou None se o
        # --input nao inclui o tensor. Com "--input fa tensor" o tensor vive em
        # (1, 7): girar o bloco errado corromperia a entrada em silencio.
        self.tensor_span = tensor_span

        # Margem para girar: um ponto do patch de lado L, depois de girar t, vem
        # de no maximo (L/2)(cos t + sin t) do centro. Recortar o bloco com essa
        # folga e girar SO o bloco custa ~5x menos que girar o volume inteiro, e
        # ainda evita que o canto do patch venha do preto de fora do volume.
        meia = self.patch / 2
        t = np.deg2rad(self.limit_deg)
        self.margin = int(np.ceil(meia * (np.cos(t) + np.sin(t)) - meia))

    def _origem(self, tgt, bloco):
        """Canto inicial do bloco a recortar, em cada eixo espacial."""
        shape = np.asarray(tgt.shape[1:])
        maximo = np.maximum(shape - np.asarray(bloco), 0)

        if random.random() < self.fg_prob:
            alvo = np.argwhere(tgt[0] > 0)
            if len(alvo):
                # Centra o bloco num voxel de CC sorteado; o clip devolve para
                # dentro do volume quando o voxel esta perto da borda.
                centro = alvo[random.randrange(len(alvo))]
                return np.clip(centro - np.asarray(bloco) // 2, 0, maximo)

        return np.array([random.randint(0, int(m)) for m in maximo])

    def __call__(self, img, tgt):
        girar = random.random() < self.p
        # O angulo e sorteado AQUI para podermos aplicar exatamente a mesma
        # rotacao as componentes do tensor.
        angle = random.uniform(-self.limit_deg, self.limit_deg) if girar else 0.0
        margem = self.margin if girar else 0

        # A rotacao e em torno de x, entao so os eixos y e z precisam da margem.
        bloco = (self.patch, self.patch + 2 * margem, self.patch + 2 * margem)
        origem = self._origem(tgt, bloco)
        corte = (slice(None), *(slice(i, i + b) for i, b in zip(origem, bloco)))
        img, tgt = img[corte], tgt[corte]

        if girar:
            # Plano do giro: os dois eixos espaciais que sobram tirando o
            # sagital, +1 por causa do eixo de canal do array [C, X, Y, Z].
            plano = tuple(i + 1 for i in range(3) if i != SAGITTAL_AXIS)
            # order=1 na imagem (e continua) e order=0 no alvo (tem que
            # continuar binario).
            img = ndimage_rotate(img, angle, axes=plano, reshape=False, order=1,
                                 mode="nearest")
            tgt = ndimage_rotate(tgt, angle, axes=plano, reshape=False, order=0,
                                 mode="nearest")
            if self.tensor_span is not None:
                i, j = self.tensor_span
                comp = rotate_tensor_channels(
                    np.moveaxis(img[i:j], 0, -1), rotation_matrix_x(np.deg2rad(angle))
                )
                img[i:j] = np.moveaxis(comp, -1, 0)
            # Joga a margem fora: o miolo e a parte do bloco que nao viu borda
            # durante a interpolacao.
            miolo = (slice(None), slice(None),
                     slice(margem, margem + self.patch),
                     slice(margem, margem + self.patch))
            img, tgt = img[miolo], tgt[miolo]

        return img, tgt


class CenterCrop3D:
    """Center crop de imagem e alvo para um cubo de lado `size`."""

    def __init__(self, size):
        self.crop = CropVolume(size)

    def __call__(self, img, tgt):
        return self.crop.apply(img), self.crop.apply(tgt)


def get_transform(transform_str: str, components=DEFAULT_INPUT, patch=PATCH_SIZE):
    """Factory de transformacoes (None = sem augmentation).

    A string de entrada controla qual objeto de transformada e instanciado, o
    que facilita a reproducibilidade: `transform_str` e um hiperparametro.
    `components` diz onde estao os canais do tensor, quando ha algum.

    NOTA 3D: o padrao da avaliacao e None, e nao um center crop como na versao
    2D — o volume ja sai do pre-processamento no tamanho final, e a rede, sendo
    totalmente convolucional, aceita o volume inteiro mesmo tendo treinado em
    patches. "center_crop" continua disponivel para avaliar em patch quando o
    volume inteiro nao couber na memoria.
    """
    if transform_str == "rotate_crop":
        return RotateCropTensor3D(
            patch=patch, limit_deg=10, p=0.5, fg_prob=0.5,
            tensor_span=channel_layout(components).get("tensor"),
        )
    if transform_str == "center_crop":
        return CenterCrop3D(patch)
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
        patch = self.hparams.patch_size
        train_t = get_transform(self.hparams.train_transform_str, componentes, patch)
        eval_t = get_transform(self.hparams.eval_transform_str, componentes, patch)
        processed_dir = self.hparams.processed_dir
        self.train = BrainHack3DataVolume("train", processed_dir, transform=train_t)
        self.val = BrainHack3DataVolume("val", processed_dir, transform=eval_t)
        self.test = BrainHack3DataVolume("test", processed_dir, transform=eval_t)

    def _loader(self, dataset, shuffle, batch_size=None, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=batch_size or self.hparams.batch_size,
            num_workers=self.hparams.nworkers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        # NOTA 3D: drop_last no TREINO. Com 115 amostras e batch 2, a ultima
        # batch da epoca tem uma amostra so, e o BatchNorm em modo treino exige
        # mais de um valor por canal: no fundo do encoder o patch ja esta
        # reduzido 16x, entao com --patch-size 16 sobra 1x1x1 por amostra e a
        # epoca morre na ultima batch. Descartar a sobra custa menos de uma
        # amostra por epoca, que o shuffle redistribui na epoca seguinte.
        # (So o treino: a avaliacao roda com o BatchNorm em modo eval, que usa
        # as running stats e aceita batch 1.)
        return self._loader(
            self.train, shuffle=True, drop_last=len(self.train) > self.hparams.batch_size
        )

    # NOTA 3D: avaliacao com batch 1. O treino ve patches de 64^3, mas aqui
    # passa o volume 128^3 inteiro — 8x mais voxels por amostra — e empilhar
    # dois deles no mesmo batch e o que estoura a memoria da GPU primeiro.
    def val_dataloader(self):
        return self._loader(self.val, shuffle=False, batch_size=1)

    def test_dataloader(self):
        return self._loader(self.test, shuffle=False, batch_size=1)


# ============================================================================
# Funcao de perda: Dice
# ============================================================================


def dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    """Coeficiente de Dice entre a entrada e o alvo: 2x overlap / uniao.

    Durante o treino podemos calcular o Dice por amostra e tirar a media, ou
    considerar o batch inteiro de uma vez (reduce_batch_first).

    NOTA 3D: a versao 2D descia recursivamente ate um tensor de DUAS dimensoes,
    o que aqui daria um Dice por FATIA do patch. Isso nao seria so uma media
    diferente: a maioria das fatias de um patch nao tem CC nenhum, e o ramo de
    "conjuntos vazios" devolve 1.0 para cada uma delas — a perda desapareceria
    debaixo de fatias vazias perfeitas. Achatar tudo menos o batch da
    exatamente o mesmo numero da versao 2D (la sobrava so o eixo de canal, de
    tamanho 1) e o numero certo em 3D.
    """
    assert input.size() == target.size()
    if input.dim() < 2 and reduce_batch_first:
        raise ValueError(f"Dice: tensor sem batch (shape {input.shape})")

    # Uma linha por amostra do batch (ou uma linha so, com reduce_batch_first).
    grupos = 1 if reduce_batch_first or input.dim() < 2 else input.shape[0]
    flat_input = input.reshape(grupos, -1)
    flat_target = target.reshape(grupos, -1)

    inter = (flat_input * flat_target).sum(dim=1)
    sets_sum = flat_input.sum(dim=1) + flat_target.sum(dim=1)
    # Alvo e predicao vazios: Dice 1, como no notebook (sets_sum = 2 * inter).
    sets_sum = torch.where(sets_sum == 0, 2 * inter, sets_sum)

    return ((2 * inter + epsilon) / (sets_sum + epsilon)).mean()


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
        # NOTA 3D: "bilinear" so vale para entradas de 4 dimensoes; o
        # equivalente para volumes e "trilinear".
        self.up = nn.Upsample(
            scale_factor=2, align_corners=True,
            mode="trilinear" if dim == "3d" else "bilinear",
        )
        self.conv = DoubleConv(in_ch, out_ch, norm, reduce=False, dim=dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ajuste de tamanho quando as dimensoes nao batem apos o upsample.
        # NOTA 3D: generico no numero de eixos espaciais (2 em 2D, 3 em 3D). O
        # F.pad consome os eixos de tras para frente, dai o reversed — a versao
        # 2D montava o par (diffY, diffX) na ordem trocada, o que passava
        # despercebido porque as fatias de treino e avaliacao eram quadradas.
        pad = []
        for eixo in reversed(range(2, x2.dim())):
            diff = x2.size(eixo) - x1.size(eixo)
            pad.extend([diff // 2, diff - diff // 2])
        x1 = F.pad(x1, pad)
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
            dim="3d",  # NOTA 3D: unica mudanca na arquitetura (Conv3d, BatchNorm3d).
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
    """Volume escalar (X, Y, Z) para a figura, escolhido pelo layout de canais.

    Com mais de um canal nao existe "a imagem" para mostrar. Preferimos a FA (se
    for um dos canais), depois a FA recalculada do tensor, e por ultimo o
    primeiro canal.
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
        # justamente o que destaca o CC.
        return arr[layout["tensor_inv"][0]]
    return arr[0]


def sagittal_index(tgt):
    """Fatia sagital a mostrar na figura: a de maior area de CC no alvo.

    NOTA 3D: olhar o alvo aqui e inofensivo porque SO a figura usa este indice —
    nenhuma decisao do pipeline (recorte, treino, metricas) depende dele. Sem
    alvo nenhum, cai no meio do volume.
    """
    area = np.asarray(tgt).sum(axis=(1, 2))
    return int(np.argmax(area)) if area.max() > 0 else len(area) // 2


def save_triptych(plt, img, tgt, pred, index, split, out_dir):
    """Salva entrada, alvo e predicao contigua lado a lado, numa fatia sagital.

    NOTA 3D: os tres paineis mostram a MESMA fatia do volume (a de maior CC no
    alvo), senao a comparacao visual nao diria nada.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = sagittal_index(tgt)

    plt.figure(figsize=(12, 4))
    for j, (arr, title) in enumerate(
        [
            (img[k], f"Entrada, {split} {index} (sagital {k})"),
            (tgt[k], "CC (alvo)"),
            (pred[k], "Predicao (continua)"),
        ]
    ):
        plt.subplot(1, 3, j + 1)
        plt.imshow(arr, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{split}_{index:03d}.png", dpi=110)
    plt.close()


def predict_split(model, dataset, device, split, figures_dir=None, max_figures=6,
                  components=DEFAULT_INPUT):
    """Roda a rede em todo o split e devolve (alvos, predicoes) como numpy."""
    tgts_np, preds_np = [], []
    # Resolve o matplotlib uma vez: se faltar, as figuras ficam desligadas.
    plt = load_pyplot() if figures_dir is not None else None

    for i in range(len(dataset)):
        img, tgt = dataset[i]

        # torch.no_grad() desabilita gradientes (so necessarios no treino):
        # economiza memoria e tempo.
        #
        # NOTA: o sigmoid nao esta na rede (CCSegmentation.forward devolve
        # LOGITS, para a BCEWithLogitsLoss do treino), entao ele precisa entrar
        # aqui: e o que poe a predicao em [0, 1], a faixa em que os thresholds
        # da varredura significam probabilidade. Sem ele, "threshold=0.05"
        # corta de fato em sigmoid(0.05) = 0.51, e a varredura de 0.05 a 0.95
        # cobre apenas 0.51..0.72 de probabilidade.
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            pred = torch.sigmoid(logits).cpu().squeeze().numpy()

        tgts_np.append(tgt.squeeze().numpy())
        preds_np.append(pred)

        # A entrada so e necessaria para as figuras, entao nao acumulamos o split inteiro.
        if plt is not None and i < max_figures:
            save_triptych(
                plt, to_display(img, components), tgts_np[-1], pred, i, split, figures_dir
            )

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
        "volume_size": args.volume_size,
        "patch_size": args.patch_size,
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
        description="Segmentacao 3D do corpo caloso no volume (BrainHack 3.0).",
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
        "--processed-dir",
        default=None,
        help="Saida dos volumes .npz (padrao: depende de --input).",
    )
    p.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT, help="Raiz dos logs/checkpoints.")
    p.add_argument(
        "--experiment-name",
        default=None,
        help=f"Padrao: {DEFAULT_EXPERIMENT}_<input>.",
    )
    p.add_argument("--force-preprocess", action="store_true", help="Reprocessa mesmo se os .npz existirem.")
    # NOTA 3D: sai o --sagittal-axis (nao ha mais fatia a escolher) e entram os
    # dois tamanhos que definem o pipeline 3D.
    p.add_argument(
        "--volume-size",
        type=int,
        default=VOLUME_SIZE,
        help="Lado do cubo salvo pelo pre-processamento. Precisa ser multiplo de 16.",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help="Lado do patch de treino, sorteado do volume. Multiplo de 16.",
    )

    p.add_argument("--epochs", type=int, default=50)
    # NOTA 3D: batch e largura menores que na versao 2D. Um patch 64^3 tem 16x
    # mais voxels que uma fatia 64^2, e a memoria da GPU e o limite.
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--init-channels", type=int, default=16)
    p.add_argument("--precision", default="32")
    p.add_argument("--train-transform", default="rotate_crop")
    p.add_argument("--eval-transform", default="none")
    p.add_argument("--debug", action="store_true", help="fast_dev_run: 1 batch de treino/validacao.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--threshold", type=float, default=0.5, help="Limiar usado quando o estagio 'eval' nao roda.")
    p.add_argument(
        "--no-threshold-sweep",
        action="store_true",
        help="Nao varre thresholds na validacao; usa --threshold.",
    )
    p.add_argument("--figures-dir", default=None, help="Se informado, salva PNGs das predicoes.")
    p.add_argument("--max-figures", type=int, default=6)
    p.add_argument("--checkpoint", default=None, help="Checkpoint para eval/test (padrao: o mais recente).")

    args = p.parse_args(argv)
    # A UNet reduz a resolucao 4 vezes por 2: um lado que nao seja multiplo de
    # 16 volta do decoder com tamanho diferente do skip.
    for nome, valor in (("--volume-size", args.volume_size), ("--patch-size", args.patch_size)):
        if valor % 16:
            p.error(f"{nome}={valor} precisa ser multiplo de 16.")
    if args.patch_size > args.volume_size:
        p.error(f"--patch-size ({args.patch_size}) nao cabe em --volume-size ({args.volume_size}).")
    # O BatchNorm em modo treino precisa de mais de um valor por canal. No fundo
    # do encoder (4 reducoes por 2) cada amostra contribui com (patch/16)^3
    # voxels, entao quem tem que passar de 1 e batch x (patch/16)^3. Conferir
    # aqui evita descobrir isso com um ValueError no meio da primeira epoca.
    voxels_no_fundo = (args.patch_size // 16) ** 3
    if args.batch_size * voxels_no_fundo < 2:
        p.error(
            f"--batch-size {args.batch_size} com --patch-size {args.patch_size} deixa so "
            f"{args.batch_size * voxels_no_fundo} valor por canal no fundo do encoder, e o "
            f"BatchNorm precisa de mais de um. Aumente --batch-size ou --patch-size."
        )

    pedidos = set(STAGES) if "all" in args.stages else set(args.stages)
    # Ordem canonica: "--stages test eval" nao pode avaliar antes de escolher o threshold.
    args.stages = [stage for stage in STAGES if stage in pedidos]
    # precision aceita "32", "16-mixed", "bf16-mixed", ...
    if args.precision.isdigit():
        args.precision = int(args.precision)
    # Defaults que dependem de --input: pasta dos .npz e nome do experimento.
    # Separados por entrada para que trocar --input nao reaproveite volumes nem
    # checkpoints da entrada anterior (o numero de canais nem bate). O limiar
    # entra no nome pelo mesmo motivo: ele muda o conteudo dos .npz sem mudar a
    # contagem de arquivos, e o preprocess pula quando a contagem ja bate.
    args.input = normalize_components(args.input)
    sufixo = input_tag(args.input)
    if args.fa_threshold is not None:
        sufixo += f"_th{args.fa_threshold:g}"
    if args.processed_dir is None:
        # NOTA 3D: prefixo proprio. As pastas preprocessed_cc_* guardam as
        # FATIAS da versao 2D; reusa-las aqui daria um erro de shape so no
        # primeiro conv (e a pasta preprocessed_cc_3d, de uma tentativa
        # anterior, guarda volumes inteiros, sem recorte).
        args.processed_dir = f"preprocessed_cc_3d_{sufixo}"
    if args.experiment_name is None:
        args.experiment_name = f"{DEFAULT_EXPERIMENT}_{sufixo}"
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
                Path(args.search_root), BrainHack3Data.required_files(args.input)
            )
        print(f"\nDATA_DIR: {data_dir}")
        preprocess(
            build_preprocess_volume(args.volume_size, args.fa_threshold),
            data_dir=data_dir,
            split_dir=args.split_dir,
            out_root=processed_dir,
            components=args.input,
            force=args.force_preprocess,
        )

    # Os estagios que leem os volumes usam sempre os datasets do DataModule, para
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

    # Os .npz nao guardam nem qual entrada nem qual recorte os gerou. Sem esta
    # checagem, apontar --processed-dir para a pasta de outra entrada (ou para
    # as fatias 2D do outro script) so falharia la na frente, como um erro de
    # shape no primeiro conv. Le do disco, e nao do dataset, para a resposta nao
    # depender da transformada de avaliacao em uso.
    img0 = np.load(data_module.val.dataset[0])["img"]
    canais, volume = img0.shape[0], tuple(img0.shape[1:])
    if canais != hparams["nin"]:
        raise RuntimeError(
            f"{processed_dir} tem {canais} canais, mas --input {input_tag(args.input)} espera "
            f"{hparams['nin']}. Rode 'preprocess' (com --force-preprocess se a pasta "
            f"ja existir) ou aponte --processed-dir para a pasta certa."
        )
    esperado = (args.volume_size,) * 3
    if volume != esperado:
        raise RuntimeError(
            f"{processed_dir} tem volumes {volume}, e nao {esperado}. Rode 'preprocess' "
            f"com --force-preprocess, ajuste --volume-size ou aponte --processed-dir "
            f"para a pasta certa."
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

    # ---- 3. Validacao: predicoes, figuras e escolha do threshold ---------
    threshold = args.threshold
    if "eval" in args.stages:
        tgts_np, preds_np = predict_split(
            trained, data_module.val, device, "val", args.figures_dir,
            args.max_figures, args.input
        )
        if args.no_threshold_sweep:
            metrics = compute_metrics(tgts_np, preds_np, threshold)
        else:
            threshold, metrics = sweep_threshold(
                tgts_np, preds_np, np.arange(0.05, 1.0, 0.05)
            )
        report(metrics, threshold, "val")

    # ---- 4. Avaliacao final no teste -------------------------------------
    if "test" in args.stages:
        tgts_np, preds_np = predict_split(
            trained, data_module.test, device, "test", args.figures_dir,
            args.max_figures, args.input
        )
        report(compute_metrics(tgts_np, preds_np, threshold), threshold, "test")


if __name__ == "__main__":
    main()
