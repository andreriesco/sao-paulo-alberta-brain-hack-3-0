"""brainhack — biblioteca do hands-on de segmentacao do corpo caloso (BrainHack 3.0).

Todo o codigo reutilizavel do notebook `gabarito-hands-on-brainhack-dti-cc-segmentation`
vive aqui: datasets, transformadas, pre-processamento, UNet, LightningModule e metricas.
Os comentarios didaticos e as marcas `# GABARITO (bug N)` foram preservados.

Importar este modulo NAO toca o disco: a resolucao de caminhos esta em `configure()`.

Uso no Kaggle (anexe como Dataset ou Utility Script):

    from brainhack import configure, BrainHack3Data, preprocess
    DATA_DIR = configure()
"""

import json
import os
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
# GABARITO (bug 6): reorientacao dos volumes com MONAI (ver realign_volume abaixo).
from monai.transforms import LoadImage, Orientation, SaveImage
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================================
# Dados: caminhos e dataset 3D
# ============================================================================

DATA_ROOT = Path("/kaggle/working")
DATA_JSON = Path("/kaggle/input/competitions/sao-paulo-alberta-brain-hack-3-0")
COMP_DIR = Path("/kaggle/input/competitions/sao-paulo-alberta-brain-hack-3-0")
PROCESSED_DATA_FOLDER = "preprocessed_cc"
SAGITTAL_AXIS = 0

def find_data_root(start: Path) -> Path:
    """Retorna o diretorio pai das pastas de sujeito, a qualquer profundidade.

    Identifica sujeito pelo conteudo (presenca dos NIfTI esperados),
    nao por profundidade nem por formato do nome.
    """
    targets = {"FA.nii", "cc_mask_mricloud_1.25.nii"}
    for p in sorted(start.rglob("*")):
        if p.is_dir() and targets <= {f.name for f in p.iterdir() if f.is_file()}:
            return p.parent
    raise FileNotFoundError(
        f"Nenhuma pasta contendo {sorted(targets)} encontrada em {start}"
    )

DATA_DIR = None  # so existe depois de configure(); importar o modulo nao toca o disco.


def configure(comp_dir: Path = COMP_DIR) -> Path:
    """Resolve DATA_DIR — e o antigo trecho de nivel de modulo da celula 6.

    Chame antes de qualquer outra coisa (dataset, preprocess, treino): sem isso
    DATA_DIR fica None e BrainHack3Data nao acha as pastas de sujeito.
    """
    global DATA_DIR

    DATA_DIR = find_data_root(
        comp_dir,
    )

    print(DATA_DIR)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATA_DIR.resolve()}")

    return DATA_DIR


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(path).get_fdata()


class BrainHack3Data(Dataset):
    '''
    Dataset que acessar arquivos 3D dos pacientes, seguindo o split de treino, validação ou teste (mode).

    Separar dados no nível do paciente é essencial para evitar contaminação (dados do mesmo paciente no treino e teste por exemplo).
    '''
    # Uso de propriedades da classe para constantes relacionadas ao dataset.
    FA_FILE = "FA.nii"
    CC_FILE = "cc_mask_mricloud_1.25.nii"


    def __init__(self, mode: str, transform=None):
        
        # Biblioteca PATH permite construção dinâmica de caminhos com operador /
        # GABARITO (bug 1): o caminho era fixo em "train.json", entao mode nao tinha efeito
        # e train/val/test carregavam o mesmo indice -> montamos o nome do JSON a partir de mode.
        split_path = DATA_JSON / f"{mode}.json"

        # Sempre bom verificar se o arquivo existe.
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split não encontrado: {split_path}\n"
                "Execute: python data/split_subjects.py"
            )
        
        # Salvamos o índice de sujeitos para o mode informado.
        with open(split_path) as f:
            self.subject_ids = json.load(f)

        # Salvamos o objeto transformada, opcional.
        self.transform = transform

    def __len__(self):
        # Na abstração de orientação a objetos do Python, o __len__ roda quando a função len() é chamada sobre o objeto.
        return len(self.subject_ids)

    def __getitem__(self, i):
        # Consulta do índice de sujeitos.
        subject_id = self.subject_ids[i]
        subject_dir = DATA_DIR / subject_id

        # Função que extrai os dados do arquivo em NIfTI. Note que evitamos nomes "mágicos" usando constantes da classe.
        fa = load_nifti(subject_dir / BrainHack3Data.FA_FILE)
        cc = load_nifti(subject_dir / BrainHack3Data.CC_FILE)

        # Expandimos as dimensões para que o tensor tenha formato [canal, Z, Y, X], padrão em aprendizado de máquina.
        img = np.expand_dims(fa, 0).astype(np.float32)
        mask = np.expand_dims(cc, 0).astype(np.float32)

        # Caso uma transformação tenha sido passada, aplica-la.
        # Note que a entrada e saída são modeladas como pares de imagem e máscara, que é uma escolha de design.
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # É sempre interessante retornar metadados, facilita análises futuras.
        metadata = {"subject_id": subject_id}
        
        return img, mask, metadata

# ============================================================================
# Transformadas do dataset 3D
# ============================================================================

class ComposeTransforms:
    """Aplica uma lista de transformadas em sequência."""
    def __init__(self, transforms):
        # Note que transforms é uma lista de objetos "chamáveis" (funções que implementam __call__).
        self.transforms = transforms

    def __call__(self, x, y=None):

        # Aplica cada transformada em sequência sobre tuplas x, y.
        # y=None é o caminho de inferência (volume novo, sem máscara).
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
    """
    Extrai a fatia sagital média de volumes 3D.
    Critério: menor FA médio entre as fatias com tecido suficiente segundo a
    máscara de CÉREBRO (arquivo T1_brain_mask_1.25.nii* do sujeito ou, na falta
    dele, fa_vol > 0). A máscara do corpo caloso (o alvo y) NÃO é usada aqui:
    ela é o que a rede deve prever, e num volume novo ela nem existe.
    """
    def __init__(self, sagittal_axis=0):
        self.sagittal_axis = sagittal_axis

    def _brain_mask(self, fa_vol, subject_dir=None):
        # Máscara de cérebro: vem da ENTRADA, nunca do alvo.
        if subject_dir is not None:
            hits = sorted(Path(subject_dir).glob("T1_brain_mask_1.25.nii*"))
            if hits:
                return load_nifti(hits[0]) > 0
        # Sem arquivo: aproximação a partir da própria FA (voxels com sinal).
        return fa_vol > 0


    def _find_slice_index(self, fa_vol):
        other_axes = tuple(i for i in range(fa_vol.ndim) if i != self.sagittal_axis)
        mask_count = self._brain_mask(fa_vol).sum(axis=other_axes)
        fa_mean = fa_vol.mean(axis=other_axes)
        fa_mean[mask_count <= 0.90 * mask_count.max()] = 1
        return int(np.argmin(fa_mean))

    def __call__(self, x, y=None):
        fa_vol = x[0]
        slice_idx = self._find_slice_index(fa_vol)  # só depende da entrada
        fa_slice = np.take(fa_vol, slice_idx, axis=self.sagittal_axis)
        x = np.expand_dims(fa_slice, 0).astype(np.float32)
        if y is None:
            return x, None  # inferência: não há máscara a recortar
        cc_slice = np.take(y[0], slice_idx, axis=self.sagittal_axis)
        y = np.expand_dims((cc_slice > 0).astype(np.float32), 0)
        return x, y


preprocess_3d = ComposeTransforms([
    MinMaxNormalize(),
    ExtractMidSagittalSlice(sagittal_axis=SAGITTAL_AXIS),
])

# ============================================================================
# Pre-processamento: reorientacao e salvamento em 2D
# ============================================================================

# GABARITO (bug 6): erro oculto - cada sujeito do HCP pode chegar em uma orientacao diferente.
# Sem padronizar a orientacao, a "fatia sagital media" cai em planos anatomicos distintos de
# sujeito para sujeito, e a rede aprende com fatias desalinhadas entre si.
# Solucao: reorientar todos os NIfTI para o mesmo codigo de eixos ("LAS") antes do pre-processamento 2D.


def realign_volume(in_path_list, output_dir, axcodes):

    for in_path in in_path_list:
            
        # carrega a imagem e seu affine original (cada volume pode ter uma orientacao diferente nesse ponto)
        data, meta = LoadImage(ensure_channel_first=True, image_only=False)(in_path)
        # reorienta os dados (e atualiza o affine) para o codigo de eixos alvo, ex: "LAS" -> Left, Anterior, Superior
        data = Orientation(axcodes=axcodes)(data)
        # salva o resultado; SaveImaged usa o affine atualizado, entao o arquivo de saida fica fisicamente correto na nova orientacao
        SaveImage(output_dir=str(output_dir), output_postfix="reoriented", separate_folder=False, resample=False, print_log=False)(data, meta)

# --- exemplo de uso
# for p in tqdm(list_paths_kaggle):

#     path_t1 = os.path.join(p, "T1_1.25.nii.gz")
#     path_t1_brain = os.path.join(p, "T1_brain_1.25.nii.gz")
#     path_brain_mask = os.path.join(p, "T1_brain_mask_1.25.nii.gz")
#     path_cc_fs = os.path.join(p, "cc_mask_fs_1.25.nii.gz")
#     path_cc_mricloud = os.path.join(p, "cc_mask_mricloud_1.25.nii.gz")

#     in_path_list = [path_t1, path_t1_brain, path_brain_mask, path_cc_fs, path_cc_mricloud]

#     pipeline = realign_volume(in_path_list, p, "LAS")


# Raiz reorientada (DATA_DIR fica em /kaggle/input, que e somente leitura).
REORIENT_ROOT = DATA_ROOT / "reoriented"
AXCODES = "LAS"

# Nomes gerados pelo SaveImage do MONAI: <stem>_<output_postfix><output_ext>.
REORIENTED_FA_FILE = "FA_reoriented.nii.gz"
REORIENTED_CC_FILE = "cc_mask_mricloud_1.25_reoriented.nii.gz"


def reorient_dataset():
    '''
    Reorienta FA e CC de cada sujeito para AXCODES e devolve a nova raiz de dados.

    E idempotente: sujeitos ja reorientados sao pulados, entao rodar a celula duas vezes nao refaz o trabalho.
    '''
    for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        out_dir = REORIENT_ROOT / subject_dir.name
        if out_dir.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        realign_volume(
            [subject_dir / BrainHack3Data.FA_FILE, subject_dir / BrainHack3Data.CC_FILE],
            out_dir,
            AXCODES,
        )

    return REORIENT_ROOT


def preprocess():
    '''
    Essa função faz o pré-processamento dos dados e salva em disco.
    Uma chamada dessa função deve gerar os dados 2D de treino, validação e teste.

    A modularização anterior de abstração do dataset 3D, e transformada como objeto chamável deixa esse código bem simples.

    GABARITO: os DOIS erros graves eram (bug 2) o dataset instanciado fora do loop, sempre com mode="train",
    e (bug 3) o pré-processamento sem normalização. O bug 6 (orientação) também é corrigido aqui.
    '''
    # GABARITO (bug 6): os volumes vinham em orientações diferentes -> reorientamos tudo para "LAS"
    # e passamos a ler os NIfTI reorientados (DATA_DIR e os nomes de arquivo apontam para a nova raiz).
    global DATA_DIR
    DATA_DIR = reorient_dataset()
    BrainHack3Data.FA_FILE = REORIENTED_FA_FILE
    BrainHack3Data.CC_FILE = REORIENTED_CC_FILE

    # GABARITO (bug 3): faltava normalizar a FA -> reusamos preprocess_3d, que já compõe
    # MinMaxNormalize() antes de ExtractMidSagittalSlice().
    preprocess_function = preprocess_3d

    for mode in ["train", "val", "test"]:
        out_dir = Path(PROCESSED_DATA_FOLDER) / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        # GABARITO (bug 2): o dataset era criado antes do loop, então mode nunca mudava e os três
        # splits recebiam os mesmos sujeitos -> instanciamos dentro do loop, com o mode da iteração.
        dataset = BrainHack3Data(mode, transform=preprocess_function)

        for img, tgt, metadata in tqdm(dataset, desc=f"Preprocess {mode}"):
            subject_id = metadata["subject_id"]
            save_path = out_dir / f"{subject_id}.npz"
            np.savez_compressed(save_path, img=img, tgt=tgt)

# ============================================================================
# Dataset 2D, augmentation e DataModule
# ============================================================================

class BrainHack3Data2D(Dataset):
    '''
    Esse dataset modela nossa tarefa bidimensional: segmentar o corpo caloso de uma fatia sagital média usando uma UNet e o mapa de FA como entrada.

    Ele deve 
    '''
    def __init__(self, mode, transform=None):
        '''
        mode: train, val ou test
        transform: transformação opcional a ser aplicada aos dados
        '''
        
        # Indexa paths dos arquivos que pré-processamos acima, simplesmente guardando todos os arquivos com fim .npz em uma lista. 
        # CUIDADO: a ordem de "glob" não é determinística! Por isso o sorted.
        self.dataset = sorted(glob(os.path.join(PROCESSED_DATA_FOLDER, mode, "*.npz")))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        '''
        Lê o arquivo .npz e prepara os tensores PyTorch para o treino.

        GABARITO: o erro sútil e grave estava na conversão de tipo da imagem (ver comentário abaixo).
        '''
        npz = np.load(self.dataset[i])
        # GABARITO (bug 4): a FA normalizada vive em [0, 1] e o cast para uint8 zerava a imagem inteira
        # (quebrando visualização e escala) -> a imagem passa a ser lida como float32.
        # A máscara continua binária, então uint8 está correto para ela.
        img = npz["img"].astype(np.float32).squeeze()
        tgt = npz["tgt"].astype(np.uint8).squeeze()

        if self.transform is not None:
            out = self.transform(image=img, mask=tgt)
            img, tgt = out["image"], out["mask"]

        # Formato esperado pela rede: [canal, altura, largura]
        img = torch.from_numpy(img).float().unsqueeze(0)
        tgt = torch.from_numpy(tgt).float().unsqueeze(0)
        return img, tgt


def get_transform(transform_str: str):
    '''
    Factory de transformações (None = sem augmentation).

    A ideia de uma "fábrica de objetos" é muito útil para organizar código de aprendizado profundo.

    Simplesmente a string de entrada controla qual o objeto de transformada que será instanciado, facilitando reproducibilidade de experimentos.

    transform_str deve ser guardado como um hiperparâmetro do experimento.

    Note que a biblioteca Albumentations de transformadas 2D segue um esquema de composição semelhante a nossas transformadas 3D!

    Não tem erros nessa função.
    '''
    if transform_str == "rotate_crop":
        return A.Compose([
            A.Rotate(limit=(-10, 10), p=0.5),
            A.RandomCrop(width=64, height=64),
        ])
    if transform_str == "center_crop":
        return A.Compose([
            A.CenterCrop(width=128, height=128),
        ])
    return None


class BrainHack3DataModule(pl.LightningDataModule):
    '''
    O DataModule ajuda a organizar o seu experimento, especialmente com a seção setup, onde 
    você pode centrar tarefas de criação do dataset. 

    Aqui no nosso exemplo simples, o LightningDataModule inicializa os datasets e seus respectivos DataLoaders.

    DataLoaders são responsáveis por iterar sobre os dados de forma eficiente e criar batches (conjuntos de amostras).
    
    Redes convolucionais como a UNet geralmente processam batches inteiros de uma vez, em vez de aprender de amostra em amostra.
    '''
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage=None):
        train_t = get_transform(self.hparams.train_transform_str)
        eval_t = get_transform(self.hparams.eval_transform_str)
        self.train = BrainHack3Data2D("train", transform=train_t)
        self.val = BrainHack3Data2D("val", transform=eval_t)
        self.test = BrainHack3Data2D("test", transform=eval_t)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.nworkers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.nworkers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.nworkers, shuffle=False)

# ============================================================================
# Funcao de perda, arquitetura UNet e LightningModule
# ============================================================================

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    '''
    Calcula o coeficiente de Dice entre a entrada e o alvo.

    Ilustração: 2x overlap / união entre as máscaras.

    Note um detalhe importante: durante o treino, temos um conjunto de imagens e os alvos (máscaras) respectivas.
    Podemos calcular o Dice para cada imagem e depois calcular a média, ou considerar o batch inteiro de uma vez.

    Não há erros nessa função.
    '''
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f"Dice: tensor sem batch (shape {input.shape})")

    # Se a entrada tem duas dimensões ou queremos tratar o batch inteiro de uma vez, aplicamos o dice na linearização dos valores. 
    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)

    # Caso a entrada tenha mais de duas dimensões, calculamos o Dice para cada elemento do batch.
    dice = 0
    for i in range(input.shape[0]):
        dice += dice_coeff(input[i, ...], target[i, ...])

    return dice / input.shape[0]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm, reduce, dim):
        '''
        Módulo dinâmico 3D ou 2D, cria uma camada de duas convoluções com batch normalization e leaky ReLU.

        O Argumento reduce controla o stride da segunda convolução para reduzir a resolução espacial da saída.
        '''
        super().__init__()
        if norm:
            norms = [getattr(nn, f"BatchNorm{dim}")(out_ch) for _ in range(2)]
        else:
            norms = [nn.Identity(), nn.Identity()]

        self.conv = nn.Sequential(
            getattr(nn, f"Conv{dim}")(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
            norms[0],
            nn.LeakyReLU(inplace=True),
            getattr(nn, f"Conv{dim}")(out_ch, out_ch, kernel_size=3, padding=1, stride=2 if reduce else 1, bias=False),
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
        # Ajuste de tamanho quando as dimensões não batem após upsample
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


class CCSegmentation(pl.LightningModule):
    def __init__(self, hparams):
        '''
        Lightning Module é uma abstração da arquitetura por completo, implementando os passos de treino e validação (o que acontece com um batch).

        No construtor, inicializamos a arquitetura, e salvamos os hiperparâmetros, que agora podem ser referenciados como `self.hparams`.
        '''
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = UNet(
            n_channels=self.hparams.nin,
            n_classes=self.hparams.nout,
            norm=True,
            dim="2d",
            init_channel=32,
        )

    def forward(self, x):
        # GABARITO (bug 5): a rede devolvia logits sem faixa definida, mas o Dice e o threshold
        # assumem probabilidades em [0, 1] -> aplicamos sigmoid na saída.
        return self.model(x).sigmoid()

    def step(self, mode, batch):
        '''
        O Lightning Module implementa o passo de treino e validação.

        O método `step` é chamado para cada batch, e implementa o que deve ser feito com ele.

        Basicamente, precisamos passar os dados de entrada pela rede e calcular a perda.

        O Lightning cuida de gerenciamento do otimizador, mover para GPU, etc.
        '''
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
        '''
        Aqui configuramos o otimizador.
        '''
        return Adam(self.model.parameters(), lr=self.hparams.lr)

# ============================================================================
# Metricas de segmentacao
# ============================================================================

def seg_metrics(gts, preds, metrics, struct_names=["cc"]):
    # Loop sobre pares de predição e alvo para compará-los
    for gt, pred, label in zip(gts, preds, struct_names):
        # sitk implementa filtors que fornecem múltiplas métricas.
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        hausdorff = sitk.HausdorffDistanceImageFilter()

        # converte numpy para formato do sitk
        gt_img = sitk.GetImageFromArray(gt)
        pred_img = sitk.GetImageFromArray(pred)

        # executa filtros (métricas)
        overlap.Execute(gt_img, pred_img)

        # salva métricas em dicionário dinâmico
        metrics[label]["dice"].append(overlap.GetDiceCoefficient())
        metrics[label]["jaccard"].append(overlap.GetJaccardCoefficient())
        try:
            hausdorff.Execute(gt_img, pred_img)
            metrics[label]["hd"].append(hausdorff.GetHausdorffDistance())
        except Exception:
            metrics[label]["hd"].append(nan)
