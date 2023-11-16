"""
Dual-Teacher
Copyright (c) 2023-present NAVER Cloud Corp.
distributed under NVIDIA Source Code License for SegFormer
--------------------------------------------------------
References:
SegFormer: https://github.com/NVlabs/SegFormer
--------------------------------------------------------
"""
import argparse
import copy
import os
import os.path as osp
import time
import logging
import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
import numpy as np
from torchvision.transforms import ToTensor
from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
from seg_core.model import MiT_SegFormer
from seg_core.optimizer import PolyWarmupAdamW
from seg_core.augmentations import ClassMixLoss, compute_classmix, compute_cutmix, compute_ic
import seg_core.eval_seg as eval_seg
import torch.nn.functional as F
import warnings
from torchvision.utils import save_image
from dist_helper import setup_distributed
import torch.distributed as dist
from mmseg.apis import single_gpu_test
from mmcv.image import tensor2imgs
from PIL import Image, ImageOps, ImageFilter
import random
from torchvision import transforms
from copy import deepcopy

warnings.filterwarnings("ignore")
criterion_u = torch.nn.CrossEntropyLoss(reduction='none').cuda()


def train_sup(args, model, optimizer, train_loader, val_loader, criterion, max_iters, print_iters, eval_iters):
    train_iterator = iter(train_loader)
    if args.ddp:
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank = 0
    for epoch in range(200):
        for i in range(len(train_loader)):

            model.train()
            try:
                batch_data = next(train_iterator)
            except:
                train_iterator = iter(train_loader)
                batch_data = next(train_iterator)

            image = batch_data['img'].data[0].cuda(non_blocking=True)
            label = batch_data['gt_semantic_seg'].data[0].squeeze(dim=1).cuda(non_blocking=True)
            outputs = model(image)
            outputs = F.interpolate(outputs, size=label.shape[1:], mode='bilinear', align_corners=False)
            seg_loss = criterion(outputs, label.type(torch.long))

            optimizer.zero_grad()
            seg_loss.backward()
            optimizer.step()

        if rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logging.info("save_path:{}".format(args.save_path))
            logging.info("Iter: %d; LR: %.3e; seg_loss: %f" % (i + 1, lr, seg_loss.item()))
            print("Iter: %d; LR: %.3e; seg_loss: %f" % (i + 1, lr, seg_loss.item()))
            logging.info('[iter:{}] Validation:'.format(i + 1))
            print('[iter:{}] Validation:'.format(i + 1))
            val_score = val(model.module, val_loader)
            logging.info('mIoU:{:.5f}'.format(val_score['Mean IoU'] * 100))
            print('mIoU:{:.5f}'.format(val_score['Mean IoU'] * 100))
            model.train()


def train_dual(args, model, model_teacher, model_teacher2, optimizer, train_loader, train_loader_u, val_loader, criterion, cm_loss_fn, max_iters, print_iters, eval_iters):
    if args.ddp:
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank = 0
    best_miou, best_epoch = 0, 0
    for epoch in range(200):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        train_loader_u.sampler.set_epoch(epoch)
        train_iterator = iter(train_loader)
        train_iterator_u = iter(train_loader_u)

        if epoch % 2 == 0:
            ema_model = model_teacher
            do_cut_mix = True
            do_class_mix = False
        else:
            ema_model = model_teacher2
            do_cut_mix = False
            do_class_mix = True

        ema_model.train()
        for i in range(len(train_loader)):
            try:
                batch_data_u = next(train_iterator_u)
            except:
                train_iterator_u = iter(train_loader_u)
                batch_data_u = next(train_iterator_u)

            try:
                batch_data = next(train_iterator)
            except:
                train_iterator = iter(train_loader)
                batch_data = next(train_iterator)

            image = batch_data['img'].data[0].cuda(non_blocking=True)
            label = batch_data['gt_semantic_seg'].data[0].squeeze(dim=1).cuda(non_blocking=True)
            image_u = batch_data_u['img'].data[0].cuda(non_blocking=True)
            label_u = batch_data['gt_semantic_seg'].data[0].squeeze(dim=1).cuda(non_blocking=True)

            b, _, h, w = image.shape

            image_u_strong = deepcopy(image_u)
            image_u_strong = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_u_strong)
            image_u_strong = transforms.RandomGrayscale(p=0.2)(image_u_strong)

            if do_class_mix:
                loss = compute_classmix(b, h, w, criterion, cm_loss_fn, model, ema_model, image, label, image_u, image_u_strong, threshold=0.95)
            if do_cut_mix:
                loss = compute_cutmix(h, w, image, label, criterion, model, ema_model, image_u, threshold=0.95)

            loss_dc = compute_ic(model, ema_model, image_u, image_u_strong, criterion_u, label_u, h, w, threshold=0.95)
            total_loss = loss + loss_dc * 0.2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if args.ddp:
                reduced_loss = loss.clone().detach()
                dist.all_reduce(reduced_loss)

            update_ema(model_teacher=ema_model, model=model, alpha_teacher=0.99, iteration=i)

            if rank == 0:
                if (i + 1) % print_iters == 0:
                    lr = optimizer.param_groups[0]['lr']
                    logging.info("Epoch: %d; Iter: %d; LR: %.3e; loss: %f" % (epoch, i + 1, lr, loss.item()))
                    print("Epoch: %d; Iter: %d; LR: %.3e; loss: %f" % (epoch, i + 1, lr, loss.item()))

        if rank == 0:
            logging.info('[Epoch {}] [iter:{}] Validation:'.format(epoch, i + 1))
            print('[Epoch {}] [iter:{}] Validation:'.format(epoch, i + 1))

            val_score = val(model.module, val_loader)
            miou = val_score['Mean IoU'] * 100
            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
            logging.info('mIoU:{:.5f} Best mIOU:{:.5f} on epoch {}'.format(miou, best_miou, best_epoch))
            print('mIoU:{:.5f} Best mIOU:{:.5f} on epoch {}'.format(miou, best_miou, best_epoch))
            model.train()


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def val(model, data_loader):
    model.eval()
    preds, gts = [], []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            image = data['img'][0].cuda(non_blocking=True)
            label = data['gt_semantic_seg'][0].cuda(non_blocking=True)
            outputs = model(image)
            resized_outputs = F.interpolate(outputs, size=label.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(label.cpu().numpy().astype(np.int16))

    score = eval_seg.scores(gts, preds, num_classes=150)
    model.train()
    return score


def val_ddp(args, epoch, model, data_loader):
    model.eval()
    preds, gts = [], []
    if args.ddp:
        data_loader.sampler.set_epoch(epoch)
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # print(data)
            image = data['img'][0].cuda(non_blocking=True)
            label = data['gt_semantic_seg'][0].cuda(non_blocking=True)

            outputs = model(image)
            resized_outputs = F.interpolate(outputs, size=label.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(label.cpu().numpy().astype(np.int16))
            if args.ddp:
                preds = torch.from_numpy(np.array(preds)).cuda()
                gts = torch.from_numpy(np.array(gts)).cuda()
                dist.all_reduce(preds)
                dist.all_reduce(gts)
    gts = list(gts)
    preds = list(preds)

    score = eval_seg.scores(gts, preds, num_classes=150)
    return score


def intersectionAndUnion(output, target, K, ignore_index):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def update_ema(model_teacher, model, alpha_teacher, iteration):
    with torch.no_grad():
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
        for ema_param, param in zip(model_teacher.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]


def setup_logger(filename='test.log'):
    ## setup logger
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--ddp', default=False, action='store_true')
    parser.add_argument('--dual_teacher', default=False, action='store_true')
    parser.add_argument('--unimatch_aug', default=False, action='store_true')
    parser.add_argument('--save_path', type=str, help='log moemo')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dc', default=False, action='store_true')

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """
    import tempfile
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def image_saver(input, name):
    """
    :param name: "path/name"
    """
    if input.dim() == 3:
        input = input.unsqueeze(dim=0)
    save_image(input.float(), str(name) + '.jpg')


def main():
    setup_logger()

    args = parse_args()
    mit_type = args.backbone[-1]
    if mit_type == '5':
        args.config = 'local_configs/segformer/B' + mit_type + '/segformer.b' + mit_type + '.640x640.ade.160k.py'
    else:
        args.config = 'local_configs/segformer/B' + mit_type + '/segformer.b' + mit_type + '.512x512.ade.160k.py'

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    torch.backends.cudnn.benchmark = False

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    distributed = False
    if args.ddp:
        rank, word_size = setup_distributed(port=args.port)
        distributed = True
    else:
        rank = 0
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    print('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # log some basic info
    print(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        print(f'Set random seed to {args.seed}, deterministic: '
              f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = MiT_SegFormer(backbone=args.backbone,
                          num_classes=150,
                          embedding_dim=256,
                          pretrained=True)

    if args.ddp: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model_teacher = MiT_SegFormer(backbone=args.backbone + '_ema',
                                  num_classes=150,
                                  embedding_dim=256,
                                  pretrained=True).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False

    model_teacher2 = MiT_SegFormer(backbone=args.backbone + '_ema',
                                   num_classes=150,
                                   embedding_dim=256,
                                   pretrained=True).cuda()
    for p in model_teacher2.parameters():
        p.requires_grad = False

    param_groups = model.get_param_groups()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print('trainable_params:', trainable_params)

    shuffle = True
    if args.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        shuffle = False

    max_iters = 50000
    print_iters = 100
    eval_iters = 5000

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": cfg.optimizer.lr,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.lr * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iters,
        max_iter=max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )
    supervised_full = False

    if supervised_full:
        datasets = [build_dataset(cfg.data.train)]
    else:
        datasets = [build_dataset(cfg.data.train_semi_l)]

    datasets_u = [build_dataset(cfg.data.train_semi_u)]
    datasets_val = [build_dataset(cfg.data.val)]

    batch_size = 4
    train_loader = [
        build_dataloader(
            ds,
            samples_per_gpu=batch_size,
            workers_per_gpu=0,
            num_gpus=1,
            dist=distributed,
            shuffle=shuffle,
            seed=cfg.seed,
            drop_last=True,
            pin_memory=True) for ds in datasets
    ]
    train_loader_u = [
        build_dataloader(
            ds,
            samples_per_gpu=batch_size,
            workers_per_gpu=0,
            num_gpus=1,
            dist=distributed,
            shuffle=shuffle,
            seed=cfg.seed,
            drop_last=True,
            pin_memory=True) for ds in datasets_u
    ]
    val_loader = [
        build_dataloader(
            ds,
            samples_per_gpu=1,
            workers_per_gpu=0,
            num_gpus=1,
            dist=distributed,
            shuffle=False,
            seed=cfg.seed,
            drop_last=False,
            pin_memory=True) for ds in datasets_val
    ]

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
    cm_loss_fn = ClassMixLoss(weight=None, reduction='none', ignore_index=255)
    train_dual(args, model, model_teacher, model_teacher2, optimizer, train_loader[0], train_loader_u[0], val_loader[0], criterion, cm_loss_fn, max_iters, print_iters, eval_iters)


if __name__ == '__main__':
    main()
