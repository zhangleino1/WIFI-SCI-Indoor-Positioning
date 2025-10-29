# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import torch
from cnn_net_model import CNN_Net
import os

# 检查点路径
ckpt_path = './logs/cnn/version_0/checkpoints/cnn-best-epoch=00-val_loss=4.558.ckpt'

print("=== 测试模型检查点 ===")
print(f"检查点路径: {ckpt_path}")
print(f"检查点存在: {os.path.exists(ckpt_path)}")

if os.path.exists(ckpt_path):
    # 加载检查点信息
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    print(f"模型类别数: {hparams['num_classes']}")
    print(f"时间步长: {hparams['time_step']}")
    
    # 加载模型
    model = CNN_Net.load_from_checkpoint(ckpt_path, map_location='cpu')
    model.eval()
    print("模型加载成功")
    
    # 创建测试数据
    test_data = torch.randn(16, 6, 15, 30)
    print(f"测试数据形状: {test_data.shape}")
    
    # 前向传播
    with torch.no_grad():
        logits = model(test_data)
        predictions = torch.argmax(logits, dim=1)
    
    print(f"模型输出形状: {logits.shape}")
    print(f"预测类别: {predictions}")
    
    # 统计预测分布
    unique_preds = torch.unique(predictions)
    print(f"预测的唯一类别数: {len(unique_preds)}")
    print(f"预测类别范围: [{predictions.min()}, {predictions.max()}]")
    
    if len(unique_preds) > 5:
        print("SUCCESS: 模型能预测多个不同类别!")
    else:
        print("WARNING: 模型预测仍集中在少数类别")
        
else:
    print("ERROR: 检查点文件不存在")