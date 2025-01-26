from prettytable import PrettyTable
from models import UNetNested, LWRNetF, repvgg_model_convert
import torch
from thop import profile
import numpy as np


def model_para_sum(model, logger=None):
    total_para = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_para = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    non_trainable_para = total_para - trainable_para
    total_para = round(total_para, 2)
    trainable_para = round(trainable_para, 2)
    non_trainable_para = round(non_trainable_para, 2)
    table = PrettyTable(['Statistic', 'Trainable', 'Non-trainable', 'Total'])
    table.add_row(['Parameter(M)', f'{trainable_para}M', f'{non_trainable_para}M', f'{total_para}M'])
    logger.info('\n' + f'{table}') if logger else print(table)


def model_flops_sum(model, test_input, logger=None):
    flops, params = profile(model, inputs=(test_input,))
    logger.info(f'FLOPs: {flops / 1e9:.2f}G') if logger else print(f'FLOPs: {flops / 1e9:.2f}G')
    logger.info(f'Params: {params / 1e6:.2f}M') if logger else print(f'Params: {params / 1e6:.2f}M')


def fps_count(model, device):
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float).to(device)
    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))


if __name__ == '__main__':
    device = torch.device('cuda:0')
    test_input = torch.randn(1, 3, 1024, 1024).to(device)
    # model= UNetNested(in_channels=3, n_classes=6)
    model = LWRNetF(n_classes=6)
    # model = repvgg_model_convert(model)
    model.to(device)
    model_para_sum(model)
    model_flops_sum(model, test_input)
    fps_count(model, device)
