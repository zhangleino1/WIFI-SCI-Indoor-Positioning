# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import os

import torch

from cnn_net_model import CNN_Net


ckpt_path = './logs/cnn/version_0/checkpoints/last.ckpt'

print('=== 回归模型检查点测试 ===')
print(f'检查点路径: {ckpt_path}')
print(f'检查点存在: {os.path.exists(ckpt_path)}')

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    print(f"时间步长: {hparams.get('time_step')}")
    print(f"子载波数: {hparams.get('num_subcarriers')}")
    print(f"回归损失: {hparams.get('reg_loss')}")

    model = CNN_Net.load_from_checkpoint(ckpt_path, map_location='cpu')
    model.eval()
    print('模型加载成功')

    time_step = hparams.get('time_step', 15)
    num_subcarriers = hparams.get('num_subcarriers', 30)
    test_data = torch.randn(16, 6, time_step, num_subcarriers)
    print(f'测试数据形状: {test_data.shape}')

    with torch.no_grad():
        preds = model(test_data)

    print(f'模型输出形状: {preds.shape}')
    print('前 5 个预测坐标:')
    print(preds[:5])

    if preds.shape == (16, 2) and torch.isfinite(preds).all():
        print('SUCCESS: 模型输出为有效的二维坐标回归结果')
    else:
        print('WARNING: 模型输出形状或数值异常')
else:
    print('ERROR: 检查点文件不存在')
