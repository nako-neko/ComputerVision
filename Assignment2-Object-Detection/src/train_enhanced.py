import os
import tqdm
import numpy as np
import argparse
import torch
import torchvision
from torchvision import transforms
from data.dataset import Dataset
from model.ADdetector import resnet50
from utils.loss import yololoss

# --------------------------------------------------------
# 全局函数定义
# --------------------------------------------------------
def load_pretrained(net):
    # 使用 weights 参数替代被弃用的 pretrained=True
    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_state_dict = resnet.state_dict()
    net_dict = net.state_dict()
    for k in resnet_state_dict.keys():
        if k in net_dict.keys() and not k.startswith('fc'):
            net_dict[k] = resnet_state_dict[k]
    net.load_state_dict(net_dict)

# --------------------------------------------------------
# 主程序入口 (Windows多进程必须保护)
# --------------------------------------------------------
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_S', default=14, type=int, help='YOLO grid num')
    parser.add_argument('--yolo_B', default=2, type=int, help='YOLO box num')
    parser.add_argument('--yolo_C', default=4, type=int, help='detection class num')
    
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_workers', default=16, type=int, help='dataloader workers')
    
    # 新增 Warmup 参数
    parser.add_argument('--warmup_epochs', default=2, type=int, help='number of warmup epochs')

    parser.add_argument('--seed', default=666, type=int, help='random seed')
    parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
    parser.add_argument('--output_dir', default='checkpoints', type=str, help='output directory')

    parser.add_argument('--l_coord', default=3., type=float, help='hyper parameter for localization loss')
    parser.add_argument('--l_noobj', default=0.1, type=float, help='hyper parameter for no object loss')
    parser.add_argument('--image_size', default=448, type=int, help='image size')
    
    args = parser.parse_args()

    # Environment Setting
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f'DEVICE: {device}')
    print(f'CUDA DEVICE COUNT: {torch.cuda.device_count()}')

    # Other settings
    args.load_pretrain = True
    print(args)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize model & loss
    criterion = yololoss(args, l_coord=args.l_coord, l_noobj=args.l_noobj)
    ad_detector = resnet50(args=args)
    
    if args.load_pretrain:
        load_pretrained(ad_detector)
    ad_detector = ad_detector.to(device)

    ad_detector.train()

    # initialize optimizer
    optimizer = torch.optim.AdamW(ad_detector.parameters(), betas=(0.9, 0.999), lr=args.learning_rate)

    # ========================================================
    # 新增: 学习率调度器 (Cosine Annealing)
    # T_max 设置为 num_epochs，eta_min 设置为 1e-6 防止 LR 降为 0
    # ========================================================
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # initialize dataset
    # 仅使用最基本的 ToTensor，数据增强已在 Dataset 类内部实现
    train_dataset = Dataset(args, split='train', transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_dataset = Dataset(args, split='val', transform=[transforms.ToTensor()])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {args.batch_size}')

    best_val_loss = np.inf

    for epoch in range(args.num_epochs):
        ad_detector.train()
        total_loss = 0.
        
        # ========================================================
        # 新增: Warmup 策略
        # 在前几个 epoch，线性增加 LR，避免大 LR 导致模型震荡
        # ========================================================
        if epoch < args.warmup_epochs:
            # 线性 Warmup: lr = target_lr * (current_step / warmup_steps)
            # 这里简化为按 epoch 级别 warmup
            warmup_lr = args.learning_rate * ((epoch + 1) / args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # 打印当前 LR
        current_lr = optimizer.param_groups[0]['lr']
        print(('\n' + '%10s' * 4) % ('epoch', 'loss', 'gpu', 'lr'))
        
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            pred = ad_detector(images)
            loss = criterion(pred, target)

            total_loss += loss.data

            # Gradient backward
            optimizer.zero_grad()
            loss.backward()
            
            # ========================================================
            # 新增: 梯度裁剪 (Standard Practice)
            # 防止梯度爆炸，增强稳定性
            # ========================================================
            torch.nn.utils.clip_grad_norm_(ad_detector.parameters(), max_norm=10.0)
            
            optimizer.step()

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s' + '%10.4g') % (
                '%g/%g' % (epoch + 1, args.num_epochs), 
                total_loss / (i + 1), 
                mem, 
                current_lr
            )
            progress_bar.set_description(s)

        # ========================================================
        # 新增: Warmup 结束后更新 Scheduler
        # ========================================================
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # validation
        validation_loss = 0.0
        ad_detector.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.to(device)
                target = target.to(device)

                prediction = ad_detector(images)
                loss = criterion(prediction, target)
                validation_loss += loss.data
        
        validation_loss /= len(val_loader)
        print(f"validation loss: {validation_loss.item():.4f}")

        if best_val_loss > validation_loss:
            best_val_loss = validation_loss
            save = {'state_dict': ad_detector.state_dict()}
            torch.save(save, os.path.join(output_dir, 'ad_detector_best.pth'))

        # 每 10 个 epoch 保存一次，节省空间，或者只保存 best
        if (epoch + 1) % 10 == 0:
            save = {'state_dict': ad_detector.state_dict()}
            torch.save(save, os.path.join(output_dir, f'ad_detector_epoch_{epoch+1}.pth'))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # 新增: 训练结束后将结果写入 CSV，方便 Grid Search 统计
    # ========================================================
    csv_file = "grid_search_summary.csv"
    # 如果文件不存在，添加表头 (虽然 grid_search.py 会做，但为了独立运行也能写)
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("OutputDir, BestValLoss, LR, BS, L_Coord, L_NoObj\n")
            
    with open(csv_file, "a") as f:
        # 使用 .item() 转换 tensor，防止格式错误
        val_loss_val = best_val_loss.item() if torch.is_tensor(best_val_loss) else best_val_loss
        f.write(f"{args.output_dir}, {val_loss_val:.4f}, {args.learning_rate}, {args.batch_size}, {args.l_coord}, {args.l_noobj}\n")

    print("Training finished.")
