import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils import Save_Tool
from utils.metrics import Eval_Metrics


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label':
                device_input[key] = value.to(device)
            elif type(value) is list or type(value) is torch.Size:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


class Trainer():
    def __init__(self, args, logger, model, optimizer, criterion, device, train_loader, valid_loader=None,
                 scheduler=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.do_validation = self.valid_loader is not None
        self.logger = logger
        self.scheduler = scheduler
        self.log_step = 10
        self.best_valid_mIoU = -1
        self.train_metrics = Eval_Metrics(num_classes=self.args.class_num,
                                          metrics=['mIoU', 'mDice'],
                                          classes=['Background', 'Super', 'Incomplete'
                                              , 'Hopping', 'Streaking'
                                              , 'Lattice'
                                                   ],
                                          )
        self.valid_metrics = Eval_Metrics(num_classes=self.args.class_num,
                                          metrics=['mIoU', 'mDice'],
                                          classes=['Background', 'Super', 'Incomplete'
                                              , 'Hopping', 'Streaking'
                                              , 'Lattice'
                                                   ],
                                          logger=self.logger)
        self.save_list = Save_Tool(max_num=self.args.max_num_save)

    def train(self):
        self.model.train()
        not_improved_count = 0
        train_loss = []
        valid_loss = []
        miou = []
        for epoch in range(self.args.epochs):
            self.logger.info(f'---------------- Epoch: {epoch + 1} ----------------')
            start_time = time.time()
            result = self.train_one_epoch(epoch + 1)

            if os.getenv('LOCAL_RANK', None) == '0':
                train_loss.append(result['train_loss'])
                valid_loss.append(result['val_valid_loss'])
                miou.append(result['val_all_classes']['IoU'])

                self._save_loss_curve(train_loss, valid_loss)
                self._save_mIoU_curve(miou)

                best = False
                improved = result['val_all_classes']['IoU'] >= self.best_valid_mIoU
                if improved:
                    self.best_valid_mIoU = result['val_all_classes']['IoU']
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.args.early_stop:
                    self.logger.info('Validation performance did not improve for {} epochs. Training stops.'.format(
                        self.args.early_stop))
                    break

                if best:
                    save_path = os.path.join(self.args.model_save_dir,
                                             f'best_model_{epoch + 1}_{int(self.best_valid_mIoU * 10000)}.pth')
                    self.save_list.update(save_path)
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f'Save current best model to {save_path}')

                if (epoch + 1) % self.args.save_period == 0:
                    save_path = os.path.join(self.args.model_save_dir, f'model_{epoch + 1}.pth')
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f'Save model to {save_path}')

                end_time = time.time()
                self.logger.info(
                    f'Epoch: {epoch + 1}, Time: {end_time - start_time}')

    def train_one_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        train_loss = []

        for batch_index, batch_input in enumerate(tqdm(self.train_loader)):
            batch_input = to_device(batch_input, self.device)
            labels = batch_input['label']
            masks = self.model(batch_input['image'])
            # masks = torch.nn.functional.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=True)
            loss = self.criterion(masks, labels)
            preds = torch.nn.functional.softmax(masks, dim=1)
            preds = torch.argmax(preds, dim=1)
            self.train_metrics.update(preds, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss.append(loss.item())

            if batch_index % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_index),
                    loss.item()))

        log = self.train_metrics.compute()
        train_loss = np.nanmean(train_loss)
        self.logger.info(f'Local Rank: {os.getenv("LOCAL_RANK", None)},Epoch: {epoch},Train Loss: {train_loss}')
        log['train_loss'] = train_loss

        if os.getenv("LOCAL_RANK", None) == '0':
            if self.do_validation:
                val_log = self.valid(epoch)
                log.update(**{'val_' + k: v for k, v in val_log.items()})
                self.logger.info(
                    f'Epoch: {epoch},Val Loss: {val_log["valid_loss"]}')

        if self.scheduler is not None:
            self.scheduler.step()

        return log

    def valid(self, epoch=None):
        self.model.eval()
        self.valid_metrics.reset()
        valid_loss = []

        for batch_index, batch_input in enumerate(tqdm(self.valid_loader)):
            batch_input = to_device(batch_input, self.device)
            labels = batch_input['label']
            with torch.no_grad():
                masks = self.model(batch_input['image'])
                # masks = torch.nn.functional.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=True)
            loss = self.criterion(masks, labels)
            preds = torch.nn.functional.softmax(masks, dim=1)
            preds = torch.argmax(preds, dim=1)
            self.valid_metrics.update(preds, labels)

            valid_loss.append(loss.item())

        val_log = self.valid_metrics.compute()
        valid_loss = np.nanmean(valid_loss)
        val_log['valid_loss'] = valid_loss

        return val_log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx + 1
        total = self.train_loader.__len__()

        return base.format(current, total, 100.0 * current / total)

    def _save_loss_curve(self, train_loss, valid_loss):
        iterations = list(range(1, len(train_loss) + 1))
        fig, ax = plt.subplots()
        ax.plot(iterations, train_loss, label='Train_Loss')
        ax.plot(iterations, valid_loss, label='Vaild_Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curve')
        ax.grid(True)
        ax.legend()
        plt.savefig(os.path.join(self.args.log_save_dir, 'loss_curve.png'))
        plt.close()

    def _save_mIoU_curve(self, mIoU):
        x_values = range(1, len(mIoU) + 1)

        # 绘制损失图
        plt.plot(x_values, mIoU, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('mIoU Curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.args.log_save_dir, 'miou_curve.png'))
        plt.close()
