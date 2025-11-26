import subprocess
import itertools
import os
import sys
import time

# 定义 Python 解释器
PYTHON_EXEC = sys.executable 

# === 搜索空间 (基于参考作业优化) ===
search_space = {
    'learning_rate': [1e-4, 5e-5],  # 大 Batch Size 通常需要大一点的 LR
    'batch_size': [32, 64], 
    'l_coord': [3.0, 5.0, 8.0],     # 坐标损失权重
    'l_noobj': [0.1, 0.3, 0.5]      # 无物体置信度损失权重
}

# 固定参数
FIXED_EPOCHS = 10
NUM_WORKERS = 16

# 基础命令
BASE_CMD = [
    PYTHON_EXEC, "src/train_enhanced.py",
    "--dataset_root", "./dataset",
    "--num_epochs", str(FIXED_EPOCHS),
    "--num_workers", str(NUM_WORKERS)
]

SEARCH_DIR = "checkpoints_grid_search"

def run_grid_search():
    if not os.path.exists(SEARCH_DIR):
        os.makedirs(SEARCH_DIR)
    
    # 初始化 CSV 头（如果文件不存在）
    if not os.path.exists("grid_search_summary.csv"):
        with open("grid_search_summary.csv", "w") as f:
            f.write("OutputDir, BestValLoss, LR, BS, L_Coord, L_NoObj\n")

    keys = search_space.keys()
    values = search_space.values()
    combinations = list(itertools.product(*values))
    
    print(f"🚀 开始 Grid Search，计划进行 {len(combinations)} 组实验")
    print(f"💾 结果将汇总在: grid_search_summary.csv")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # 构造实验名，方便辨认
        # 例: run_lr5e-05_coord8.0_noobj0.35
        exp_name = f"run_lr{params['learning_rate']}_coord{params['l_coord']}_noobj{params['l_noobj']}"
        output_dir = os.path.join(SEARCH_DIR, exp_name)
        
        print(f"\n[{i+1}/{len(combinations)}] Running: {exp_name}")
        print(f"   Params: {params}")

        # 组装命令
        cmd = BASE_CMD + ["--output_dir", output_dir]
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])
            
        try:
            start = time.time()
            # 运行训练
            subprocess.run(cmd, check=True)
            cost = (time.time() - start) / 60
            print(f"✅ 完成 (耗时: {cost:.1f} min)")
            
        except subprocess.CalledProcessError:
            print(f"❌ 实验失败")
            continue
        except KeyboardInterrupt:
            print("\n🛑 手动停止")
            break

if __name__ == "__main__":
    run_grid_search()
