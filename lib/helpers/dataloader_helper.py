# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
# from lib.datasets.kitti.kitti_dataset_monodetr import KITTI_Dataset as KITTI_Dataset_default


import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_dataset_monodetr import KITTI_Dataset as KITTI_Dataset_default


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_dataloader(cfg, workers=8):
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set  = KITTI_Dataset(split=cfg['test_split'],  cfg=cfg)
    else:
        raise NotImplementedError

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=cfg['batch_size'],
        num_workers=workers,
        worker_init_fn=my_worker_init_fn,
        shuffle=True,
        pin_memory=pin,
        persistent_workers=(workers > 0),
        prefetch_factor=2 if workers > 0 else None,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg['batch_size'],
        num_workers=workers,
        worker_init_fn=my_worker_init_fn,
        shuffle=False,
        pin_memory=pin,
        persistent_workers=(workers > 0),
        prefetch_factor=2 if workers > 0 else None,
        drop_last=False,
    )
    return train_loader, test_loader
