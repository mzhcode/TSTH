import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional as fnn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset, create_loader
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from my_loss_soft import My_Loss, My_Loss_eval
from utils_hash_ncg import mean_average_precision_R, acg_test

from numpy.lib import recfunctions
import scipy.io as scio
from tonic.dataset import Dataset
import os
import tonic
from tonic import DiskCachedDataset
import torch.utils
from typing import Any, Tuple
from cut_mix import CutMix, EventMix, MixUp
from rand_aug import *


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')
# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='cifar10.yml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments') # imagenet.yml  cifar10.yml

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Model detail
parser.add_argument('--model', default='vitsnn', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('-T', '--time-step', type=int, default=8, metavar='time',
                    help='simulation time step of spiking neuron (default: 4)')
parser.add_argument('-L', '--layer', type=int, default=4, metavar='layer',
                    help='model layer (default: 4)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--dim', type=int, default=None, metavar='N',
                    help='embedding dimsension of feature')
parser.add_argument('--num_heads', type=int, default=None, metavar='N',
                    help='attention head number')
parser.add_argument('--patch-size', type=int, default=None, metavar='N',
                    help='Image patch size')
parser.add_argument('--mlp-ratio', type=int, default=None, metavar='N',
                    help='expand ration of embedding dimension in MLP block')


# Dataset / Model parameters
parser.add_argument('-data-dir', metavar='DIR',default="/media/data/spike-transformer-network/torch/cifar10/",
                    help='path to dataset') #./torch/imagenet/
parser.add_argument('--dataset', '-d', metavar='NAME', default='torch/cifar10',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')

parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--val-batch-size', type=int, default=16, metavar='N',
                    help='input val batch size for training (default: 32)')
# parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
#                     help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[1.0, 1.0], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def to_float_tensor(x):
    tensor = torch.tensor(x, dtype=torch.half)
    return tensor


class UCF101DVS(Dataset):
    """ASL-DVS dataset <https://github.com/PIX2NVS/NVS2Graph>. Events have (txyp) ordering.
    ::

        @inproceedings{bi2019graph,
            title={Graph-based Object Classification for Neuromorphic Vision Sensing},
            author={Bi, Y and Chadha, A and Abbas, A and and Bourtsoulatze, E and Andreopoulos, Y},
            booktitle={2019 IEEE International Conference on Computer Vision (ICCV)},
            year={2019},
            organization={IEEE}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    sensor_size = (240, 180, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names
    folder_name = 'UCF101DVS'

    def __init__(self, save_to, transform=None, target_transform=None):
        super(UCF101DVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        # if not self._check_exists():
        #     raise NotImplementedError(
        #         'Please manually download the dataset from'
        #         ' https://www.dropbox.com/sh/ie75dn246cacf6n/AACoU-_zkGOAwj51lSCM0JhGa?dl=0 '
        #         'and extract it to {}'.format(self.location_on_system))
        self.location_on_system = save_to
        classes = os.listdir(self.location_on_system)
        self.int_classes = dict(zip(classes, range(len(classes))))
        # unique_suffixes = set()
        # self.test = []

        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith("mat"):
                    fsize = os.path.getsize(path + '/' + file) / float(1024)
                    if fsize < 1:
                        # print('{} size {} K'.format(file, fsize))
                        continue
                    self.data.append(path + "/" + file)
                    self.targets.append(self.int_classes[path.split('/')[-1]])

        self.length = self.__len__()
        self.cls_count = np.bincount(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        events, target = scio.loadmat(self.data[index]), self.targets[index]
        events = np.column_stack(
            [
                events["ts"],
                events["x"],
                self.sensor_size[1] - 1 - events["y"],
                events["pol"],
            ]
        )
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        print(self.folder_name)
        return self._folder_contains_at_least_n_files_of_type(
            10818, ".mat"
        )


def unpack_mix_param(args):
    mix_up = args['mix_up'] if 'mix_up' in args else False
    cut_mix = args['cut_mix'] if 'cut_mix' in args else False
    event_mix = args['event_mix'] if 'event_mix' in args else False
    beta = args['beta'] if 'beta' in args else 1.
    prob = args['prob'] if 'prob' in args else .5
    num = args['num'] if 'num' in args else 1
    num_classes = args['num_classes'] if 'num_classes' in args else 10
    noise = args['noise'] if 'noise' in args else 0.
    gaussian_n = args['gaussian_n'] if 'gaussian_n' in args else None
    return mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n


def get_UCF101DVS_data(batch_size, step, **kwargs):
    """
    获取DVS CIFAR10数据
    http://journal.frontiersin.org/article/10.3389/fnins.2017.00309/full
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """

    sensor_size = UCF101DVS.sensor_size
    train_transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    test_transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    train_dataset = UCF101DVS('/mnt/mzh/DVS-video/train/', transform=train_transform)
    test_dataset = UCF101DVS('/mnt/mzh/DVS-video/test/', transform=test_transform)

    train_transform = transforms.Compose([
        to_float_tensor,
        transforms.CenterCrop((176, 176)),
        transforms.RandomHorizontalFlip(),
    ])

    test_transform = transforms.Compose([
        to_float_tensor,
        transforms.CenterCrop((176, 176)),
    ])

    if 'rand_aug' in kwargs.keys():
        if kwargs['rand_aug'] is True:
            n = kwargs['randaug_n']
            m = kwargs['randaug_m']
            train_transform.transforms.insert(2, RandAugment(m=m, n=n))

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path='/mnt/mzh/4090_project/data/UCF101_DVS/cache/train_cache/',
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path='/mnt/mzh/4090_project/data/UCF101_DVS/cache/test_cache/',
                                     transform=test_transform)

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        # print('cut_mix', beta, prob, num, num_classes)
        train_dataset = CutMix(train_dataset,
                               beta=beta,
                               prob=prob,
                               num_mix=num,
                               num_class=num_classes,
                               noise=noise)

    if event_mix:
        train_dataset = EventMix(train_dataset,
                                 beta=beta,
                                 prob=prob,
                                 num_mix=num,
                                 num_class=num_classes,
                                 noise=noise,
                                 gaussian_n=gaussian_n)

    if mix_up:
        train_dataset = MixUp(train_dataset,
                              beta=beta,
                              prob=prob,
                              num_mix=num,
                              num_class=num_classes,
                              noise=noise)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,
        pin_memory=True, drop_last=True, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False,
        pin_memory=True, drop_last=False, num_workers=2
    )

    return train_loader, test_loader, mixup_active

def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:1'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    import model

    model = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.2,
        drop_block_rate=None,
        img_size_h=args.img_size, img_size_w=args.img_size,
        patch_size=args.patch_size, embed_dims=args.dim, num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
        in_channels=2, num_classes=args.num_classes, qkv_bias=False,
        depths=args.depths, sr_ratios=1,
    )


    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True, find_unused_parameters=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    loader_train, loader_eval, mixup_active = get_UCF101DVS_data(args.batch_size, 8)

    criterion = My_Loss(args, mixup_active, num_aug_splits)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    sim_matrix_last = torch.eye(101, dtype=torch.float32).cuda()
    sim_matrix_last.requires_grad = False
    sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
    sim_matrix_now.requires_grad = False
    sim_matrix_hard = torch.eye(101, dtype=torch.float32).cuda()
    sim_matrix_hard.requires_grad = False
    count_matrix = torch.zeros_like(sim_matrix_now)
    count_matrix.requires_grad = False

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            if epoch <= 50:
                train_metrics, sim_matrix_now, count_matrix = train_one_epoch(
                    epoch, model, loader_train, optimizer, criterion, args,
                    sim_matrix_hard.detach(), sim_matrix_now.detach(), count_matrix.detach(),
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                if epoch == 50:
                    sim_matrix_last = torch.where(count_matrix > 0, sim_matrix_now / count_matrix, sim_matrix_now)
                    sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
                    sim_matrix_now.requires_grad = False
                    count_matrix = torch.zeros_like(sim_matrix_now)
                    count_matrix.requires_grad = False
                    sim_matrix_now_np = sim_matrix_last.detach().cpu().numpy()
                    # 打开txt文件并以追加模式写入数据
                    with open('sim_matrices.txt', 'a') as f:
                        f.write("sim_matrix_last:\n")
                        f.write(np.array2string(sim_matrix_now_np, separator=', ',
                                                threshold=sim_matrix_now_np.size) + "\n\n")

            elif epoch > 50 and epoch <=70:
                train_metrics, sim_matrix_now, count_matrix = train_one_epoch(
                    epoch, model, loader_train, optimizer, criterion, args,
                    0.88*sim_matrix_hard.detach() + 0.12*sim_matrix_last.detach(), sim_matrix_now.detach(), count_matrix.detach(),
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                if epoch == 70:
                    sim_matrix_last_bool = sim_matrix_last >=0.01
                    sim_matrix_now_bool = sim_matrix_now >=0.01
                    sim_matrix_now = torch.where(sim_matrix_now_bool & sim_matrix_last_bool, sim_matrix_now, torch.tensor(0.0))
                    sim_matrix_last = torch.where(count_matrix > 0, sim_matrix_now / count_matrix, sim_matrix_now)
                    sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
                    sim_matrix_now.requires_grad = False
                    count_matrix = torch.zeros_like(sim_matrix_now)
                    count_matrix.requires_grad = False
                    sim_matrix_now_np = sim_matrix_last.detach().cpu().numpy()
                    # 打开txt文件并以追加模式写入数据
                    with open('sim_matrices.txt', 'a') as f:
                        f.write("sim_matrix_last:\n")
                        f.write(np.array2string(sim_matrix_now_np, separator=', ',
                                                threshold=sim_matrix_now_np.size) + "\n\n")

            elif epoch > 70 and epoch <= 90:
                train_metrics, sim_matrix_now, count_matrix = train_one_epoch(
                    epoch, model, loader_train, optimizer, criterion, args,
                    0.75*sim_matrix_hard.detach() + 0.25*sim_matrix_last.detach(), sim_matrix_now.detach(), count_matrix.detach(),
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                if epoch == 90:
                    sim_matrix_last_bool = sim_matrix_last >=0.01
                    sim_matrix_now_bool = sim_matrix_now >=0.01
                    sim_matrix_now = torch.where(sim_matrix_now_bool & sim_matrix_last_bool, sim_matrix_now, torch.tensor(0.0))
                    sim_matrix_last = torch.where(count_matrix > 0, sim_matrix_now / count_matrix, sim_matrix_now)
                    sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
                    sim_matrix_now.requires_grad = False
                    count_matrix = torch.zeros_like(sim_matrix_now)
                    count_matrix.requires_grad = False
                    sim_matrix_now_np = sim_matrix_last.detach().cpu().numpy()
                    # 打开txt文件并以追加模式写入数据
                    with open('sim_matrices.txt', 'a') as f:
                        f.write("sim_matrix_last:\n")
                        f.write(np.array2string(sim_matrix_now_np, separator=', ',
                                                threshold=sim_matrix_now_np.size) + "\n\n")

            elif epoch > 90 and epoch <= 110:
                train_metrics, sim_matrix_now, count_matrix = train_one_epoch(
                    epoch, model, loader_train, optimizer, criterion, args,
                    0.625*sim_matrix_hard.detach() + 0.375*sim_matrix_last.detach(), sim_matrix_now.detach(), count_matrix.detach(),
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                if epoch == 110:
                    sim_matrix_last_bool = sim_matrix_last >=0.01
                    sim_matrix_now_bool = sim_matrix_now >=0.01
                    sim_matrix_now = torch.where(sim_matrix_now_bool & sim_matrix_last_bool, sim_matrix_now, torch.tensor(0.0))
                    sim_matrix_last = torch.where(count_matrix > 0, sim_matrix_now / count_matrix, sim_matrix_now)
                    sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
                    sim_matrix_now.requires_grad = False
                    count_matrix = torch.zeros_like(sim_matrix_now)
                    count_matrix.requires_grad = False
                    sim_matrix_now_np = sim_matrix_last.detach().cpu().numpy()
                    # 打开txt文件并以追加模式写入数据
                    with open('sim_matrices.txt', 'a') as f:
                        f.write("sim_matrix_last:\n")
                        f.write(np.array2string(sim_matrix_now_np, separator=', ',
                                                threshold=sim_matrix_now_np.size) + "\n\n")

            elif epoch > 110 and epoch <= 130:
                train_metrics, sim_matrix_now, count_matrix = train_one_epoch(
                    epoch, model, loader_train, optimizer, criterion, args,
                    0.5*sim_matrix_hard.detach() + 0.5*sim_matrix_last.detach(), sim_matrix_now.detach(), count_matrix.detach(),
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                if epoch == 130:
                    sim_matrix_last_bool = sim_matrix_last >=0.01
                    sim_matrix_now_bool = sim_matrix_now >=0.01
                    sim_matrix_now = torch.where(sim_matrix_now_bool & sim_matrix_last_bool, sim_matrix_now, torch.tensor(0.0))
                    sim_matrix_last = torch.where(count_matrix > 0, sim_matrix_now / count_matrix, sim_matrix_now)
                    sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
                    sim_matrix_now.requires_grad = False
                    count_matrix = torch.zeros_like(sim_matrix_now)
                    count_matrix.requires_grad = False
                    sim_matrix_now_np = sim_matrix_last.detach().cpu().numpy()
                    # 打开txt文件并以追加模式写入数据
                    with open('sim_matrices.txt', 'a') as f:
                        f.write("sim_matrix_last:\n")
                        f.write(np.array2string(sim_matrix_now_np, separator=', ',
                                                threshold=sim_matrix_now_np.size) + "\n\n")

            elif epoch > 130 and epoch <= 150:
                train_metrics, sim_matrix_now, count_matrix = train_one_epoch(
                    epoch, model, loader_train, optimizer, criterion, args,
                    0.375*sim_matrix_hard.detach() + 0.625*sim_matrix_last.detach(), sim_matrix_now.detach(), count_matrix.detach(),
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                if epoch == 150:
                    sim_matrix_last_bool = sim_matrix_last >=0.01
                    sim_matrix_now_bool = sim_matrix_now >=0.01
                    sim_matrix_now = torch.where(sim_matrix_now_bool & sim_matrix_last_bool, sim_matrix_now, torch.tensor(0.0))
                    sim_matrix_last = torch.where(count_matrix > 0, sim_matrix_now / count_matrix, sim_matrix_now)
                    sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
                    sim_matrix_now.requires_grad = False
                    count_matrix = torch.zeros_like(sim_matrix_now)
                    count_matrix.requires_grad = False
                    sim_matrix_now_np = sim_matrix_last.detach().cpu().numpy()
                    # 打开txt文件并以追加模式写入数据
                    with open('sim_matrices.txt', 'a') as f:
                        f.write("sim_matrix_last:\n")
                        f.write(np.array2string(sim_matrix_now_np, separator=', ',
                                                threshold=sim_matrix_now_np.size) + "\n\n")

            elif epoch > 150 and epoch <= 170:
                train_metrics, sim_matrix_now, count_matrix = train_one_epoch(
                    epoch, model, loader_train, optimizer, criterion, args,
                    0.25*sim_matrix_hard.detach() + 0.75*sim_matrix_last.detach(), sim_matrix_now.detach(), count_matrix.detach(),
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                if epoch == 170:
                    sim_matrix_last_bool = sim_matrix_last >=0.01
                    sim_matrix_now_bool = sim_matrix_now >=0.01
                    sim_matrix_now = torch.where(sim_matrix_now_bool & sim_matrix_last_bool, sim_matrix_now, torch.tensor(0.0))
                    sim_matrix_last = torch.where(count_matrix > 0, sim_matrix_now / count_matrix, sim_matrix_now)
                    sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
                    sim_matrix_now.requires_grad = False
                    count_matrix = torch.zeros_like(sim_matrix_now)
                    count_matrix.requires_grad = False
                    sim_matrix_now_np = sim_matrix_last.detach().cpu().numpy()
                    # 打开txt文件并以追加模式写入数据
                    with open('sim_matrices.txt', 'a') as f:
                        f.write("sim_matrix_last:\n")
                        f.write(np.array2string(sim_matrix_now_np, separator=', ',
                                                threshold=sim_matrix_now_np.size) + "\n\n")

            elif epoch > 170 and epoch <= 190:
                train_metrics, sim_matrix_now, count_matrix = train_one_epoch(
                    epoch, model, loader_train, optimizer, criterion, args,
                    0.12*sim_matrix_hard.detach() + 0.88*sim_matrix_last.detach(), sim_matrix_now.detach(), count_matrix.detach(),
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                if epoch == 190:
                    sim_matrix_last_bool = sim_matrix_last >=0.01
                    sim_matrix_now_bool = sim_matrix_now >=0.01
                    sim_matrix_now = torch.where(sim_matrix_now_bool & sim_matrix_last_bool, sim_matrix_now, torch.tensor(0.0))
                    sim_matrix_last = torch.where(count_matrix > 0, sim_matrix_now / count_matrix, sim_matrix_now)
                    sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
                    sim_matrix_now.requires_grad = False
                    count_matrix = torch.zeros_like(sim_matrix_now)
                    count_matrix.requires_grad = False
                    sim_matrix_now_np = sim_matrix_last.detach().cpu().numpy()
                    # 打开txt文件并以追加模式写入数据
                    with open('sim_matrices.txt', 'a') as f:
                        f.write("sim_matrix_last:\n")
                        f.write(np.array2string(sim_matrix_now_np, separator=', ',
                                                threshold=sim_matrix_now_np.size) + "\n\n")

                    # print(sim_matrix_now)
            else:
                train_metrics, sim_matrix_now, count_matrix = train_one_epoch(
                    epoch, model, loader_train, optimizer, criterion, args, sim_matrix_last.detach(), sim_matrix_now.detach(), count_matrix.detach(),
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                sim_matrix_last_bool = sim_matrix_last >=0.01
                sim_matrix_now_bool = sim_matrix_now >=0.01
                sim_matrix_now = torch.where(sim_matrix_now_bool & sim_matrix_last_bool, sim_matrix_now, torch.tensor(0.0))
                sim_matrix_last = torch.where(count_matrix > 0, sim_matrix_now / count_matrix, sim_matrix_now)
                sim_matrix_now = torch.eye(101, dtype=torch.float32).cuda()
                sim_matrix_now.requires_grad = False
                count_matrix = torch.zeros_like(sim_matrix_now)
                count_matrix.requires_grad = False
                sim_matrix_now_np = sim_matrix_last.detach().cpu().numpy()
                # 打开txt文件并以追加模式写入数据
                with open('sim_matrices.txt', 'a') as f:
                    f.write("sim_matrix_now:\n")
                    f.write(np.array2string(sim_matrix_now_np, separator=', ', threshold=sim_matrix_now_np.size) + "\n\n")

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, loader_train, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, loader_train, epoch, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args, sim_matrix_last, sim_matrix_now, count_matrix,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input, target = input.cuda(), target.cuda()
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            hash_feature, hash_out, cls_out = model(input)
            hash_out = torch.where(hash_out == 0, torch.full_like(hash_out, -1), hash_out)
            sim_matrix_now, count_matrix, hash_loss, cls_loss, loss = loss_fn(hash_feature, hash_out, cls_out, target.to(torch.int64), sim_matrix_last, sim_matrix_now, count_matrix, epoch)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            # loss.backward()
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        fnn.reset_net(model)

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:

                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)]), sim_matrix_now, count_matrix


def validate(model, loader, loader_train, args, amp_autocast=suppress, log_suffix=''):
    # vali_criterion = My_Loss_eval(args)
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    query_label_matrix = np.empty(shape=(0,))
    query_hash_matrix = np.empty(shape=(0, 64))
    database_label_matrix = np.empty(shape=(0,))
    database_hash_matrix = np.empty(shape=(0, 64))

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # print(batch_idx)
            last_batch = batch_idx == last_idx
            input, target = input.cuda(), target.cuda()
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                hash_feature, hash_out, cls_out = model(input)
                hash_out = torch.where(hash_out == 0, torch.full_like(hash_out, -1), hash_out)
            if isinstance(cls_out, (tuple, list)):
                cls_out = cls_out[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                cls_out = cls_out.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = validate_loss_fn(cls_out, target.to(torch.int64))
            fnn.reset_net(model)

            acc1, acc5 = accuracy(cls_out, target.to(torch.int64), topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), cls_out.size(0))
            top5_m.update(acc5.item(), cls_out.size(0))

            hash_code = hash_out.cpu().numpy()
            query_label_matrix = np.concatenate((query_label_matrix, target.cpu().numpy()), axis=0)
            query_hash_matrix = np.concatenate((query_hash_matrix, hash_code), axis=0)

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

        for batch_idx, (input, target) in enumerate(loader_train):
            # print(batch_idx)
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            input, target = input.cuda(), target.cuda()
            with amp_autocast():
                hash_feature, hash_out, cls_out = model(input)
                hash_out = torch.where(hash_out == 0, torch.full_like(hash_out, -1), hash_out)
            if isinstance(cls_out, (tuple, list)):
                cls_out = cls_out[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                cls_out = cls_out.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            fnn.reset_net(model)

            torch.cuda.synchronize()

            hash_code = hash_out.cpu().numpy()
            database_label_matrix = np.concatenate((database_label_matrix, target.cpu().numpy()), axis=0)
            database_hash_matrix = np.concatenate((database_hash_matrix, hash_code), axis=0)

        Map30, Recall30 = mean_average_precision_R(database_hash_matrix, query_hash_matrix, database_label_matrix,
                                               query_label_matrix, 30, 101)
        Map50, Recall50 = mean_average_precision_R(database_hash_matrix, query_hash_matrix, database_label_matrix,
                                               query_label_matrix, 50, 101)
        Map100, Recall100 = mean_average_precision_R(database_hash_matrix, query_hash_matrix, database_label_matrix,
                                               query_label_matrix, 100, 101)
        # ACG, NDCG = acg_test(database_hash_matrix, query_hash_matrix, database_label_matrix, query_label_matrix, 100)
        print("Map30:", Map30, "Recall30:", Recall30, "Map50:", Map50, "Recall50:", Recall50, "Map100:", Map100, "Recall100:", Recall100)

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg), ('Map30', Map30), ('Recall30', Recall30), ('Map50', Map50), ('Recall50', Recall50), ('Map100', Map100), ('Recall100', Recall100) ])

    return metrics


if __name__ == '__main__':
    main()