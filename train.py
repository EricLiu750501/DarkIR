import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms

from options.options import parse
from archs import create_model, create_optim_scheduler, resume_model, save_checkpoint
from data import create_test_data
from data.dataset_reader.datapipeline import MyDataset_Crop, TemporalWindowDataset
from data.dataset_reader.dataset_bvi_lowlight import _split_scenes, _collect_windows
from losses import L1Loss, MSELoss, CharbonnierLoss, VGGLoss, EdgeLoss, FrequencyLoss
from losses.loss import SSIMloss
from utils.test_utils import setup, cleanup, eval_model, shuffle_sampler
from utils.utils import init_wandb, logging_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_bvi_single_frame(root_path, scenes):
    low_folders = ['low_light_10', 'low_light_20']
    high_folders = ['normal_light_10', 'normal_light_20']
    low_paths = []
    high_paths = []
    for scene in scenes:
        scene_path = os.path.join(root_path, scene)
        for low_folder, high_folder in zip(low_folders, high_folders):
            low_dir = os.path.join(scene_path, low_folder)
            high_dir = os.path.join(scene_path, high_folder)
            if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
                continue
            low_frames = sorted([f for f in os.listdir(low_dir) if f.lower().endswith('.png')])
            high_frames = sorted([f for f in os.listdir(high_dir) if f.lower().endswith('.png')])
            frame_set = sorted(list(set(low_frames) & set(high_frames)))
            low_paths.extend([os.path.join(low_dir, f) for f in frame_set])
            high_paths.extend([os.path.join(high_dir, f) for f in frame_set])
    return low_paths, high_paths


def build_train_loader(opt, rank, world_size):
    data_opt = opt['datasets']
    name = data_opt['name']
    batch_size = data_opt['train']['batch_size']
    num_workers = data_opt['train']['n_workers']
    cropsize = data_opt['train'].get('cropsize', None)
    crop_type = data_opt['train'].get('crop_type', 'Random')
    root_path = data_opt['train']['train_path']
    seed = data_opt['train'].get('split_seed', 2025)
    train_ratio = data_opt['train'].get('train_ratio', 0.8)

    tensor_transform = transforms.ToTensor()
    flips = transforms.RandomHorizontalFlip(p=0.5)

    if name == 'BVI_Lowlight':
        temporal_window = data_opt['train'].get('temporal_window', 1)
        frame_stride = data_opt['train'].get('frame_stride', 1)
        train_scenes, _ = _split_scenes(root_path, seed=seed, train_ratio=train_ratio)

        if temporal_window > 1:
            windows_low, centers_high = _collect_windows(root_path, train_scenes, temporal_window=temporal_window, frame_stride=frame_stride)
            if len(windows_low) == 0:
                raise ValueError(f'No temporal windows found under {root_path}. Check dataset structure and extensions.')
            train_dataset = TemporalWindowDataset(windows_low, centers_high, cropsize=cropsize,
                                                  tensor_transform=tensor_transform, flips=flips, test=False, crop_type=crop_type)
        else:
            low_paths, high_paths = build_bvi_single_frame(root_path, train_scenes)
            if len(low_paths) == 0:
                raise ValueError(f'No frames found under {root_path}. Check dataset structure and extensions.')
            train_dataset = MyDataset_Crop(low_paths, high_paths, cropsize=cropsize,
                                           tensor_transform=tensor_transform, flips=flips, test=False, crop_type=crop_type)
    else:
        raise NotImplementedError(f'{name} is not implemented for training')

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle=True, rank=rank)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        samplers = [train_sampler]
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        samplers = None

    return train_loader, samplers


def build_loss(opt, rank):
    loss_opt = opt['loss']
    loss_name = loss_opt.get('name', 'l1')
    if loss_name == 'l1':
        return L1Loss(loss_weight=loss_opt.get('weight', 1.0))
    if loss_name == 'mse':
        return MSELoss(loss_weight=loss_opt.get('weight', 1.0))
    if loss_name == 'charbonnier':
        return CharbonnierLoss(loss_weight=loss_opt.get('weight', 1.0))
    if loss_name == 'vgg':
        return VGGLoss(loss_weight=loss_opt.get('weight', 1.0))
    if loss_name == 'edge':
        return EdgeLoss(rank=rank, loss_weight=loss_opt.get('weight', 1.0))
    if loss_name == 'freq':
        return FrequencyLoss(loss_weight=loss_opt.get('weight', 1.0))
    if loss_name == 'ssim':
        return SSIMloss(loss_weight=loss_opt.get('weight', 1.0))
    raise NotImplementedError(f'Unsupported loss: {loss_name}')


def train_one_epoch(model, train_loader, criterion, optim, rank):
    model.train()
    total_loss = 0.0
    for high_batch, low_batch in train_loader:
        high_batch = high_batch.to(rank)
        low_batch = low_batch.to(rank)
        optim.zero_grad()
        outputs = model(low_batch)
        loss = criterion(outputs, high_batch)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / max(len(train_loader), 1)


def run_training(rank, world_size, opt):
    setup(rank, world_size=world_size)
    set_seed(opt.get('seed', 2025))

    init_wandb(rank, opt)

    try:
        train_loader, samplers = build_train_loader(opt, rank, world_size)
        test_loader, _ = create_test_data(rank, world_size, opt['datasets'])
    except Exception:
        cleanup()
        raise

    model, _, _ = create_model(opt['network'], rank, use_ddp=world_size > 1)
    optim, scheduler = create_optim_scheduler(opt['train'], model)
    model, optim, scheduler, start_epoch = resume_model(model, optim, scheduler, opt['save']['path'], rank, opt['network'].get('resume_training', False))

    criterion = build_loss(opt, rank)

    metrics_train = {'best_psnr': 0, 'epoch': 0, 'train_loss': 0}

    epochs = opt['train']['epochs']
    for epoch in range(start_epoch, epochs):
        shuffle_sampler(samplers, epoch)
        train_loss = train_one_epoch(model, train_loader, criterion, optim, rank)
        scheduler.step()

        metrics_train['epoch'] = epoch
        metrics_train['train_loss'] = train_loss

        metrics_eval = {}
        metrics_eval, imgs_dict = eval_model(model, test_loader, metrics_eval, rank=rank, world_size=world_size, eta=(rank == 0))
        metrics_train['best_psnr'] = save_checkpoint(model, optim, scheduler, metrics_eval, metrics_train, opt['save'], rank=rank)

        if opt['wandb']['init'] and rank == 0:
            logger = logging_dict(metrics_train.copy(), metrics_eval, imgs_dict)
            import wandb
            wandb.log(logger)

    cleanup()


def main():
    parser = argparse.ArgumentParser(description='DarkIR training')
    parser.add_argument('-p', '--config', type=str, default='./options/train/Baseline.yml')
    args = parser.parse_args()

    opt = parse(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.get('gpu', '0')

    world_size = int(opt.get('world_size', 1))
    if world_size > 1:
        mp.spawn(run_training, args=(world_size, opt), nprocs=world_size, join=True)
    else:
        run_training(0, world_size, opt)


if __name__ == '__main__':
    main()
