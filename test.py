import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import cv2
import numpy as np
import time
from datasets import XKDataset
from models import UNetNested,  LWRNetF, repvgg_model_convert
from train import parse_args, random_seed
from trainers.trainer import to_device
from utils import Eval_Metrics

class Test:
    def __init__(self, device, model, data_loader, metric):
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.metric = metric
        self.save_dir = './save_dir/'
    def test(self):
        self.model.eval()
        self.metric.reset()
        with torch.no_grad():
            for batch_input in tqdm(self.data_loader):
                batch_input = to_device(batch_input, self.device)
                masks = self.model(batch_input['image'])
                masks = F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=True)
                masks = torch.argmax(masks, dim=1)
                save_name = batch_input['name'][0].split('.')[0] + '.png'
                # self.visualize(masks, save_name, 'predict')
                # self.visualize(batch_input['label'], save_name, 'label')

                self.metric.update(batch_input['label'], masks)

        log = self.metric.compute()
        return log

    def visualize(self, mask, name, categroy='predict'):
        mask = mask.squeeze()
        mask = mask.cpu()
        mask = mask.numpy()

        cmap = np.array([[0, 0, 0],  # 黑
                         [255, 0, 255],  # 灰
                         [255, 0, 0],  # 红
                         [0, 255, 0],  # 绿
                         [0, 255, 255],  # 青
                         [0, 0, 255]])  # 蓝
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()
        for l in range(0, len(cmap)):
            r[mask == l] = cmap[l, 0]
            g[mask == l] = cmap[l, 1]
            b[mask == l] = cmap[l, 2]
        label = np.concatenate((np.expand_dims(b, axis=-1), np.expand_dims(g, axis=-1),
                                np.expand_dims(r, axis=-1)), axis=-1)
        if not os.path.exists(os.path.join(self.save_dir, f'{categroy}')):
            os.makedirs(os.path.join(self.save_dir, f'{categroy}'))
        image = cv2.imwrite(os.path.join(self.save_dir, f'{categroy}/{name}'), label)


if __name__ == '__main__':
    random_seed(114)
    model_path = '/home/data/best_model.pth'
    device = torch.device('cuda:0')
    model = LWRNetF(n_classes=6)
    model.eval()
    # model = repvgg_model_convert(model)
    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load(model_path, map_location=device).items()})
    model.eval()

    test_metric = Eval_Metrics(num_classes=6,
                               metrics=['mIoU', 'mDice'],
                               classes=['Background', 'Super', 'Incomplete', 'Hopping', 'Streaking', 'Lattice'],
                               )
    data_dir = '/home/data/'
    test_dataset = XKDataset(data_dir, 'test', 1024)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=0, pin_memory=True)
    test = Test(device, model, test_loader, test_metric)
    start = time.time()
    test.test()
    end = time.time()
    fps = len(test_loader) / (end - start)
    print(f'fps: {fps}')
