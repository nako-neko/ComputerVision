import subprocess
import itertools
import os
import sys
import time

PYTHON_EXEC = sys.executable 

# === 【重跑配置】只包含失败的那部分 ===
# 根据你的 log，失败的是 run_lr5e-05 开头的实验
rerun_space = {
    'learning_rate': [5e-5],      # 只跑这一个 LR
    'batch_size': [64],           # 保持 BS=64 (前提是你清理了显存)
    'l_coord': [3.0, 5.0, 8.0],   # 所有 Coord
    'l_noobj': [0.1, 0.3, 0.5]    # 所有 NoObj
}

FIXED_EPOCHS = 10
NUM_WORKERS = 16

BASE_CMD = [
    PYTHON_EXEC, "src/train_enhanced.py",
    "--dataset_root", "./dataset",
    "--num_epochs", str(FIXED_EPOCHS),
    "--num_workers", str(NUM_WORKERS)
]

SEARCH_DIR = "checkpoints_grid_search"

def run_rerun():
    # 确保 CSV 存在
    if not os.path.exists("grid_search_summary.csv"):
        print("警告：找不到之前的 summary 文件，将创建新文件。")
        with open("grid_search_summary.csv", "w") as f:
            f.write("OutputDir, BestValLoss, LR, BS, L_Coord, L_NoObj\n")

    keys = rerun_space.keys()
    values = rerun_space.values()
    combinations = list(itertools.product(*values))
    
    print(f"🔄 开始重跑失败的实验，共 {len(combinations)} 组")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # 保持和之前一样的命名规则
        exp_name = f"run_lr{params['learning_rate']}_coord{params['l_coord']}_noobj{params['l_noobj']}"
        output_dir = os.path.join(SEARCH_DIR, exp_name)
        
        # 检查是否已经跑成功过（防止重复跑）
        if os.path.exists(os.path.join(output_dir, "ad_detector_best.pth")):
            print(f"⏩ [跳过] {exp_name} 似乎已经存在结果。")
            continue
            
        print(f"\n[{i+1}/{len(combinations)}] Rerunning: {exp_name}")
        print(f"   Params: {params}")
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
            print(f"❌ 依然失败: {exp_name}")
            # 如果是显存不够，这里会再次报错，建议直接由用户 Ctrl+C 终止
            continue
        except KeyboardInterrupt:
            print("\n🛑 手动停止")
            break

if __name__ == "__main__":
    run_rerun()
