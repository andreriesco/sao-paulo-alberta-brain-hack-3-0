#!/usr/bin/env python3
"""brainhack_challenge — versao script do notebook do hands-on (BrainHack 3.0).

Segmentacao do corpo caloso na fatia sagital media, a partir do mapa de FA,
com uma UNet 2D treinada com PyTorch Lightning.

O notebook `brainhack-3-0-ii-encontro-ismrm-brasil-full.ipynb` foi reorganizado
em estagios executaveis, na mesma ordem das celulas:

    1. preprocess : le os volumes 3D (FA + mascara CC), normaliza, extrai a
                    fatia sagital media e salva um .npz por sujeito.
    2. train      : treina a UNet 2D sobre as fatias salvas.
    3. eval       : inferencia na validacao + varredura de threshold.
    4. test       : inferencia no teste com o melhor threshold.

Uso tipico (tudo de uma vez, dados descobertos automaticamente):

    python brainhack_challenge.py

Somente treino, com os .npz ja gerados:

    python brainhack_challenge.py --stages train --epochs 50

Smoke test rapido (1 batch de treino e 1 de validacao):

    python brainhack_challenge.py --debug

Diferencas em relacao ao notebook estao marcadas com `# NOTA:`.
"""

from __future__ import annotations

import argparse
import json
import os
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

DEFAULT_PROCESSED_DIR = "preprocessed_cc"
DEFAULT_LOGS_ROOT = "logs"
DEFAULT_EXPERIMENT = "BrainhackCC"
SAGITTAL_AXIS = 0
MODES = ("train", "val", "test")
# Estagios do pipeline, na ordem em que rodam (ver --stages).
STAGES = ("preprocess", "train", "eval", "test")


def find_data_root(start: Path) -> Path:
    """Retorna o diretorio pai das pastas de sujeito, a qualquer profundidade.

    Identifica sujeito pelo conteudo (presenca dos NIfTI esperados),
    nao por profundidade nem por formato do nome.
    """
    # Os nomes vem do dataset (BrainHack3Data) para nao existirem duas verdades.
    targets = {BrainHack3Data.FA_FILE, BrainHack3Data.CC_FILE}
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
    CC_FILE = "cc_mask_mricloud_1.25.nii"

    def __init__(self, mode, data_dir, split_dir, transform=None, fix=True):
        """mode: train, val ou test.

        NOTA: no notebook `DATA_DIR`/`DATA_JSON` eram globais; aqui sao
        argumentos explicitos, para o dataset nunca divergir do caminho que o
        pre-processamento realmente usou.
        """
        self.data_dir = Path(data_dir)

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

    def __getitem__(self, i):
        subject_id = self.subject_ids[i]
        subject_dir = self.data_dir / subject_id

        fa_nii = nib.load(subject_dir / BrainHack3Data.FA_FILE)
        cc_nii = nib.load(subject_dir / BrainHack3Data.CC_FILE)

        img = np.expand_dims(fa_nii.get_fdata(), 0).astype(np.float32)
        mask = np.expand_dims(cc_nii.get_fdata(), 0).astype(np.float32)

        if self.fix:
            mask = self.las(MetaTensor(mask, affine=cc_nii.affine)).numpy()

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        metadata = {"subject_id": subject_id}
        return img, mask, metadata


# ============================================================================
# Transformadas do dataset 3D
# ============================================================================


class ComposeTransforms:
    """Aplica uma lista de transformadas em sequencia."""

    def __init__(self, transforms):
        # transforms e uma lista de objetos "chamaveis" (implementam __call__).
        self.transforms = transforms

    def __call__(self, x, y=None):
        # y=None e o caminho de inferencia (volume novo, sem mascara).
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class MinMaxNormalize:
    """Normaliza a FA para o intervalo [0, 1]."""

    def __call__(self, x, y=None):
        xmin, xmax = x.min(), x.max()
        if xmax > xmin:
            x = (x - xmin) / (xmax - xmin)
        return x, y


class ExtractMidSagittalSlice:
    """Extrai a fatia sagital media de volumes 3D.

    Criterio: menor FA medio entre as fatias com tecido suficiente segundo a
    mascara de CEREBRO (arquivo T1_brain_mask_1.25.nii* do sujeito ou, na falta
    dele, fa_vol > 0). A mascara do corpo caloso (o alvo y) NAO e usada aqui:
    ela e o que a rede deve prever, e num volume novo ela nem existe.
    """

    def __init__(self, sagittal_axis=SAGITTAL_AXIS):
        self.sagittal_axis = sagittal_axis

    def _brain_mask(self, fa_vol, subject_dir=None):
        # Mascara de cerebro: vem da ENTRADA, nunca do alvo.
        if subject_dir is not None:
            hits = sorted(Path(subject_dir).glob("T1_brain_mask_1.25.nii*"))
            if hits:
                return load_nifti(hits[0]) > 0
        # Sem arquivo: aproximacao a partir da propria FA (voxels com sinal).
        return fa_vol > 0

    def _find_slice_index(self, fa_vol):
        other_axes = tuple(i for i in range(fa_vol.ndim) if i != self.sagittal_axis)
        mask_count = self._brain_mask(fa_vol).sum(axis=other_axes)
        fa_mean = fa_vol.mean(axis=other_axes)
        fa_mean[mask_count <= 0.90 * mask_count.max()] = 1
        return int(np.argmin(fa_mean))

    def __call__(self, x, y=None):
        fa_vol = x[0]
        slice_idx = self._find_slice_index(fa_vol)  # so depende da entrada
        fa_slice = np.take(fa_vol, slice_idx, axis=self.sagittal_axis)
        x = np.expand_dims(fa_slice, 0).astype(np.float32)
        if y is None:
            return x, None  # inferencia: nao ha mascara a recortar
        cc_slice = np.take(y[0], slice_idx, axis=self.sagittal_axis)
        y = np.expand_dims((cc_slice > 0).astype(np.float32), 0)
        return x, y


def build_preprocess_3d(sagittal_axis=SAGITTAL_AXIS):
    return ComposeTransforms(
        [
            MinMaxNormalize(),
            ExtractMidSagittalSlice(sagittal_axis=sagittal_axis),
        ]
    )


# ============================================================================
# Pre-processamento: fatias 2D em disco
# ============================================================================


def preprocess(preprocess_fn, data_dir, split_dir, out_root, force=False):
    """Extrai uma fatia sagital media por volume e salva em .npz.

    Salvar em disco acelera o treino: o codigo passa a ler fatias 2D ja
    normalizadas em vez do volume NIfTI original.
    """
    for mode in MODES:
        out_dir = Path(out_root) / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset = BrainHack3Data(
            mode, data_dir=data_dir, split_dir=split_dir, transform=preprocess_fn
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
    """Tarefa bidimensional: segmentar o CC na fatia sagital media a partir da FA."""

    def __init__(self, mode, processed_dir, transform=None):
        # Indexa os .npz gerados pelo pre-processamento.
        # CUIDADO: a ordem de "glob" nao e deterministica! Por isso o sorted.
        self.dataset = sorted(glob(os.path.join(str(processed_dir), mode, "*.npz")))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        npz = np.load(self.dataset[i])
        # A FA normalizada vive em [0, 1]: ler como float32 (uint8 zeraria a imagem).
        # A mascara continua binaria, entao uint8 esta correto para ela.
        img = npz["img"].astype(np.float32).squeeze()
        tgt = npz["tgt"].astype(np.uint8).squeeze()

        if self.transform is not None:
            out = self.transform(image=img, mask=tgt)
            img, tgt = out["image"], out["mask"]

        # Formato esperado pela rede: [canal, altura, largura]
        img = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)
        tgt = torch.from_numpy(np.ascontiguousarray(tgt)).float().unsqueeze(0)
        return img, tgt


def get_transform(transform_str: str):
    """Factory de transformacoes (None = sem augmentation).

    A string de entrada controla qual objeto de transformada e instanciado, o
    que facilita a reproducibilidade: `transform_str` e um hiperparametro.
    """
    if transform_str == "rotate_crop":
        return A.Compose(
            [
                A.Rotate(limit=(-10, 10), p=0.5),
                A.RandomCrop(width=64, height=64),
            ]
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
        train_t = get_transform(self.hparams.train_transform_str)
        eval_t = get_transform(self.hparams.eval_transform_str)
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
        self.enc = UNetEncoder(n_channels, init_channel, norm, dim)
        self.dec = UNetDecoder(n_classes, init_channel, norm, dim)
        print(f"UNet: in={n_channels} out={n_classes} dim={dim} init_ch={init_channel}")

    def forward(self, x):
        return self.dec(*self.enc(x))


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

    def forward(self, x):
        return self.model(x).sigmoid()

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


def save_triptych(plt, img, tgt, pred, index, split, out_dir):
    """Salva FA, alvo e predicao contigua lado a lado (equivalente aos plots do notebook)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    for j, (arr, title) in enumerate(
        [
            (img, f"FA, {split} {index}"),
            (tgt, "CC (alvo)"),
            (pred, "Predicao (continua)"),
        ]
    ):
        plt.subplot(1, 3, j + 1)
        plt.imshow(arr, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{split}_{index:03d}.png", dpi=110)
    plt.close()


def predict_split(model, dataset, device, split, figures_dir=None, max_figures=6):
    """Roda a rede em todo o split e devolve (alvos, predicoes) como numpy."""
    tgts_np, preds_np = [], []
    # Resolve o matplotlib uma vez: se faltar, as figuras ficam desligadas.
    plt = load_pyplot() if figures_dir is not None else None

    for i in range(len(dataset)):
        img, tgt = dataset[i]

        # torch.no_grad() desabilita gradientes (so necessarios no treino):
        # economiza memoria e tempo.
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).cpu().squeeze().numpy()

        tgts_np.append(tgt.squeeze().numpy())
        preds_np.append(pred)

        # A FA so e necessaria para as figuras, entao nao acumulamos o split inteiro.
        if plt is not None and i < max_figures:
            save_triptych(
                plt, img.squeeze().numpy(), tgts_np[-1], pred, i, split, figures_dir
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
        "nin": 1,
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
    p.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR, help="Saida das fatias .npz.")
    p.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT, help="Raiz dos logs/checkpoints.")
    p.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    p.add_argument("--force-preprocess", action="store_true", help="Reprocessa mesmo se os .npz existirem.")
    p.add_argument("--sagittal-axis", type=int, default=SAGITTAL_AXIS)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
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
    p.add_argument("--figures-dir", default=None, help="Se informado, salva PNGs das predicoes.")
    p.add_argument("--max-figures", type=int, default=6)
    p.add_argument("--checkpoint", default=None, help="Checkpoint para eval/test (padrao: o mais recente).")

    args = p.parse_args(argv)
    pedidos = set(STAGES) if "all" in args.stages else set(args.stages)
    # Ordem canonica: "--stages test eval" nao pode avaliar antes de escolher o threshold.
    args.stages = [stage for stage in STAGES if stage in pedidos]
    # precision aceita "32", "16-mixed", "bf16-mixed", ...
    if args.precision.isdigit():
        args.precision = int(args.precision)
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
        data_dir = Path(args.data_dir) if args.data_dir else find_data_root(Path(args.search_root))
        print(f"\nDATA_DIR: {data_dir}")
        preprocess(
            build_preprocess_3d(args.sagittal_axis),
            data_dir=data_dir,
            split_dir=args.split_dir,
            out_root=processed_dir,
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
            trained, data_module.val, device, "val", args.figures_dir, args.max_figures
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
            trained, data_module.test, device, "test", args.figures_dir, args.max_figures
        )
        report(compute_metrics(tgts_np, preds_np, threshold), threshold, "test")


if __name__ == "__main__":
    main()
