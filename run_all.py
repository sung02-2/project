#!/usr/bin/env python3
import os
import sys
import subprocess

# ====== 配置區域 ======
# 目標玩家 (逗號分隔)，作為環境變數提供給 Go 程式
TARGET_PLAYERS = os.getenv("TARGET_PLAYER", "device,dev1ce")
# DEM 存放資料夾 (相對於本腳本所在目錄)
DEM_FOLDER = os.getenv("DEM_FOLDER", "../DEM")
# 腳本根目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 將必要環境變數匯出
os.environ["TARGET_PLAYER"] = TARGET_PLAYERS
os.environ["DEM_FOLDER"] = DEM_FOLDER

# 確保輸出資料夾存在
for folder in ["output/kills", "output/puredata", "logs"]:
    full_path = os.path.join(BASE_DIR, folder)
    os.makedirs(full_path, exist_ok=True)

# ====== 第一步：執行 filter.py 解壓並重命名 DEM ======
def run_filter():
    print("▶ 開始執行 filter.py：解壓與重命名 DEM")
    cwd_dir = os.path.join(BASE_DIR, "demfilter")
    script = os.path.join(cwd_dir, "filter.py")
    subprocess.run([sys.executable, script], check=True, cwd=cwd_dir)
    print("✅ filter 完成\n")

# ====== 第二步：執行 catchkillevent.go 擷取擊殺事件 ======
def run_kill():
    print("▶ 開始執行 catchkillevent.go：擷取擊殺事件")
    cwd_dir = os.path.join(BASE_DIR, "catch_kill")
    script = os.path.join(cwd_dir, "catchkillevent.go")
    subprocess.run([
        "go", "run", script,
        "-target", TARGET_PLAYERS,
        "-folder", DEM_FOLDER
    ], check=True, cwd=cwd_dir, env=os.environ)
    print("✅ 擷取殺人事件完成\n")

# ====== 第三步：執行 catchtickevent.go 擷取 Tick 資料 ======
def run_tick():
    print("▶ 開始執行 catchtickevent.go：擷取 Tick 資料")
    cwd_dir = os.path.join(BASE_DIR, "catch_tick")
    script = os.path.join(cwd_dir, "catchtickevent.go")
    subprocess.run(["go", "run", script], check=True, cwd=cwd_dir, env=os.environ)
    print("✅ 擷取 Tick 資料完成\n")

# ====== 主流程 ======
def main():
    try:
        run_filter()
    except subprocess.CalledProcessError as e:
        print(f"❌ filter.py 執行失敗: {e}")
        sys.exit(e.returncode)
    try:
        run_kill()
    except subprocess.CalledProcessError as e:
        print(f"❌ catchkillevent.go 執行失敗: {e}")
        sys.exit(e.returncode)
    try:
        run_tick()
    except subprocess.CalledProcessError as e:
        print(f"❌ catchtickevent.go 執行失敗: {e}")
        sys.exit(e.returncode)
    print("🎉 所有流程執行完畢 🎉")

if __name__ == '__main__':
    main()
