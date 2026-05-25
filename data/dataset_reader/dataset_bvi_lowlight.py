import os
import random

from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms

try:
    from .datapipeline import TemporalWindowDataset
    from .utils import check_paths
except:
    from datapipeline import TemporalWindowDataset
    from utils import check_paths


def _list_frames(folder):
    frames = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return sorted(frames)


def _build_windows(low_dir, high_dir, temporal_window=3, frame_stride=1):
    low_frames = _list_frames(low_dir)
    high_frames = _list_frames(high_dir)
    frame_set = sorted(list(set(low_frames) & set(high_frames)))

    if not frame_set:
        return [], []

    index_map = {name: idx for idx, name in enumerate(frame_set)}
    windows = []
    centers = []
    half = temporal_window // 2
    for name in frame_set:
        center_idx = index_map[name]
        indices = []
        for offset in range(-half, half + 1):
            idx = center_idx + offset * frame_stride
            idx = max(0, min(idx, len(frame_set) - 1))
            indices.append(idx)
        window = [os.path.join(low_dir, frame_set[i]) for i in indices]
        center = os.path.join(high_dir, name)
        windows.append(window)
        centers.append(center)

    return windows, centers


def _split_scenes(root_path, seed=2025, train_ratio=0.8):
    scenes = [d for d in os.listdir(root_path) if d.startswith('S') and os.path.isdir(os.path.join(root_path, d))]
    scenes = sorted(scenes)
    if not scenes:
        raise ValueError(f'No BVI-Lowlight scenes found under: {root_path}')
    random.seed(seed)
    random.shuffle(scenes)
    split_index = int(len(scenes) * train_ratio)
    train_scenes = scenes[:split_index]
    test_scenes = scenes[split_index:]
    return train_scenes, test_scenes


def _collect_windows(root_path, scenes, temporal_window=3, frame_stride=1):
    windows_low = []
    centers_high = []
    low_folders = ['low_light_10', 'low_light_20']
    high_folders = ['normal_light_10', 'normal_light_20']

    for scene in scenes:
        scene_path = os.path.join(root_path, scene)
        for low_folder, high_folder in zip(low_folders, high_folders):
            low_dir = os.path.join(scene_path, low_folder)
            high_dir = os.path.join(scene_path, high_folder)
            if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
                continue
            windows, centers = _build_windows(low_dir, high_dir, temporal_window=temporal_window, frame_stride=frame_stride)
            windows_low.extend(windows)
            centers_high.extend(centers)

    if windows_low and centers_high:
        flat_low = [path for window in windows_low for path in window]
        check_paths([flat_low, centers_high])

    return windows_low, centers_high


def main_dataset_bvi_lowlight(rank=1,
                              root_path='../../data/datasets/BVI-Lowlight',
                              batch_size_test=1,
                              verbose=False,
                              num_workers=1,
                              world_size=1,
                              temporal_window=3,
                              frame_stride=1,
                              seed=2025,
                              train_ratio=0.8,
                              split='test'):
    train_scenes, test_scenes = _split_scenes(root_path, seed=seed, train_ratio=train_ratio)
    scenes = train_scenes if split == 'train' else test_scenes

    windows_low, centers_high = _collect_windows(root_path, scenes, temporal_window=temporal_window, frame_stride=frame_stride)

    if verbose:
        print(f'BVI-Lowlight scenes: {len(scenes)} | windows: {len(windows_low)}')

    tensor_transform = transforms.ToTensor()

    test_dataset = TemporalWindowDataset(windows_low, centers_high, cropsize=None,
                                         tensor_transform=tensor_transform, test=True)

    if world_size > 1:
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle=True, rank=rank)
        samplers = [test_sampler]
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)
    else:
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=False)
        samplers = None

    return test_loader, samplers
