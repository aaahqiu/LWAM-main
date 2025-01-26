import argparse
import torch
import torch.nn as nn
import random
import numpy as np
from datetime import datetime
from logger.logger import *
import torch.distributed as dist
import os
from datasets import XKDataset
from trainers import Trainer
from utils.init_model import init_model


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def parse_args():
    paser = argparse.ArgumentParser()
    paser.add_argument("--seed", type=int, default=114, help="random seed")
    paser.add_argument("--model_name", type=str, default="LWRNetF", help="model name")
    paser.add_argument("--save_dir", type=str, default="./save_dir/", help="save dir")
    paser.add_argument("--data_dir", type=str, default="./data_dir/", help="dataset dir")
    paser.add_argument("--checkpoint", type=str, default=None, help="model checkpoint")
    paser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    paser.add_argument("--lr_scheduler", type=bool, default=True, help="learning rate scheduler")
    paser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
    paser.add_argument("--batch_size", type=int, default=4, help="batch size")
    paser.add_argument("--image_size", type=int, default=1024, help="image size")
    paser.add_argument("--class_num", type=int, default=6, help="class_num")
    paser.add_argument("--run_id", type=str, default=None, help="run id")
    paser.add_argument("--save_period", type=int, default=50, help="save model period")
    paser.add_argument("--early_stop", type=int, default=400, help="early stop")
    paser.add_argument("--max_num_save", type=int, default=20, help="max num save")

    args = paser.parse_args()
    return args


def main(args):
    random_seed(args.seed)
    save_dir = Path(args.save_dir)
    if args.run_id is None:
        args.run_id = datetime.now().strftime(r'%m%d_%H%M')
    args.log_save_dir = save_dir / args.run_id / 'logs'
    args.model_save_dir = save_dir / args.run_id / 'models'
    os.makedirs(args.log_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    with open(args.log_save_dir / 'config.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args.__dict__.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    logger = get_logger('train', args.log_save_dir)

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")

    device = torch.device('cuda:{}'.format(os.getenv('LOCAL_RANK', None)))
    model=init_model(args,args.model_name)
    model=model.to(device)
    if args.checkpoint:
        model.load_state_dict({k.replace('module.', ''): v for k, v in
                                 torch.load(args.checkpoint, map_location=device).items()})
    logger.info(model)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device,
                                                find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    else:
        scheduler = None

    batch_size = args.batch_size // dist.get_world_size()
    train_dataset = XKDataset(args.data_dir, image_size=args.image_size, domain='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=0, pin_memory=True)
    val_dataset = XKDataset(args.data_dir, image_size=args.image_size, domain='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                             pin_memory=True)

    trainer = Trainer(args, logger, model, optimizer, criterion, device, train_loader, val_loader, scheduler)
    trainer.train()

    dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    main(args)
