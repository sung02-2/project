import subprocess
import argparse
import glob
import os
import re
from datetime import datetime

# ========== Natural Sort Function ==========
def natural_sort_key(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]

# ========== 準備 ==========
trained_log = "trainlog/trained_files.txt"
os.makedirs("trainlog", exist_ok=True)
if not os.path.exists(trained_log):
    open(trained_log, "w").close()

# 讀取已訓練過的 transfer 檔案
with open(trained_log, "r") as f:
    trained_files = set(line.strip().replace("\\", "/") for line in f.readlines())

# 收集並自然排序所有 transfer 檔案
all_transfer_files = sorted(glob.glob("output/transfer/transfer*.csv"), key=natural_sort_key)

# 過濾出還沒訓練過的檔案
untrained_files = [f for f in all_transfer_files if f.replace("\\", "/") not in trained_files]

# ========== 參數解析 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--max-files", type=int, default=300, help="最多訓練幾個 transfer 檔")
parser.add_argument("--dry-run", action="store_true", help="只列出將訓練的檔案，不執行 transform.py")
args = parser.parse_args()

selected_files = untrained_files[:args.max_files]

if not selected_files:
    print("✅ 沒有新的 transfer 檔案可供訓練。")
    exit()

print(f"\n🎯 將訓練 {len(selected_files)} 個檔案：")
for f in selected_files:
    print("  •", f)

if args.dry_run:
    print("🧪 Dry-run 模式：未執行 transform.py")
    exit()

# ========== 儲存中繼清單供 transform.py 使用 ==========
selected_file_list = "trainlog/selected_for_training.txt"
with open(selected_file_list, "w") as f:
    for fpath in selected_files:
        f.write(fpath.replace("\\", "/") + "\n")

# 額外備份一份帶時間戳記的記錄（可選）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"trainlog/training_batch_{timestamp}.txt", "w") as f:
    for fpath in selected_files:
        f.write(fpath.replace("\\", "/") + "\n")

# ========== 執行 transform.py ==========
transform_path = os.path.join("pytorch", "transform.py")
if not os.path.exists(transform_path):
    print(f"❌ 找不到 transform.py：{transform_path}")
    exit(1)

print("\n🚀 執行 transform.py 中...")
subprocess.run(["python", transform_path])
