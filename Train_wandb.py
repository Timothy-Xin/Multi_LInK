import os
import uuid
import pickle
import wandb
import numpy as np
import torch
from pathlib import Path

# 替换为你的 API key（从 wandb.ai/settings 拷贝）
os.environ["WANDB_API_KEY"] = "db1f65fd8fc7f3a57ac6faef9325f367caca7333"

from LInK.nn import ContrastiveTrainLoop
from LInK.DataUtils import prep_curves, uniformize

# ✅ 直接定义参数
import argparse
args = argparse.Namespace(
    cuda_device='0',
    checkpoint_folder='./Checkpoints_wandb/',
    checkpoint_name='checkpoint.LInK',
    checkpoint_interval=1,
    data_folder='./Data/',
    baseline='',
    epochs=50,
    batch_size=256,
    checkpoint_continue='',
    wandb_project='LInK-Training',
    wandb_entity='897315610-southeast-university'
)

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 启动 wandb
wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config=vars(args),
)

# 创建 checkpoint 文件夹
os.makedirs(args.checkpoint_folder, exist_ok=True)

# 加载或初始化训练器
if args.checkpoint_continue:
    ckpt_path = os.path.join(args.checkpoint_folder, args.checkpoint_continue)
    if not os.path.exists(ckpt_path):
        raise ValueError(f'Checkpoint {ckpt_path} not found. Please provide a valid checkpoint.')
    with open(ckpt_path, 'rb') as f:
        Trainer = pickle.load(f)
else:
    Trainer = ContrastiveTrainLoop(baseline=args.baseline, device=device, schedule_max_steps=args.epochs)

# 加载数据
data_path = os.path.join(args.data_folder, 'target_curves.npy')
mech_path = os.path.join(args.data_folder, 'graphs.npy')

if not os.path.exists(data_path) or not os.path.exists(mech_path):
    raise ValueError('Required data files not found. Run Download.py or provide the correct path.')

data = np.load(data_path)
mechanisms = np.load(mech_path, allow_pickle=True)

# 训练主循环
epochs_remaining = args.epochs - Trainer.current_epoch

for i in range(epochs_remaining):
    epoch = Trainer.current_epoch + 1
    hist = Trainer.train(data, mechanisms, args.batch_size, 1)

    # wandb 日志记录
    if isinstance(hist, list):
        hist = np.array(hist)  # shape = (steps_per_epoch, 3)
        wandb.log({
            "epoch": epoch,
            "train_loss": float(np.mean(hist[:, 0])),
            "train_loss_clip": float(np.mean(hist[:, 1])),
            "train_loss_mech": float(np.mean(hist[:, 2])),
            "train_loss_hist": wandb.Histogram(hist[:, 0])
        })
    elif isinstance(hist, dict):
        wandb.log({"epoch": epoch, **hist})
    else:
        wandb.log({"epoch": epoch, "train_loss": hist})

    # 保存 checkpoint
    if epoch % args.checkpoint_interval == 0:
        with open(os.path.join(args.checkpoint_folder, f"{args.checkpoint_name}_epoch{epoch}"), 'wb') as f:
            pickle.dump(Trainer, f)
        with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'wb') as f:
            pickle.dump(Trainer, f)

# 最终保存一次
with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'wb') as f:
    pickle.dump(Trainer, f)

wandb.finish()
