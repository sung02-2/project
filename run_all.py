#!/usr/bin/env python3
import os
import sys
import subprocess

# ====== 配置區域 ======
# 腳本根目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 目標玩家 (逗號分隔)
TARGET_PLAYERS = os.getenv("TARGET_PLAYER", "device,dev1ce,ZfG1v3N,devve-,ASTdevice,Astdevice,ast. device,ast. dev1ce,device-,DeViCe,sundeV1CE,dev1cE,device ^_^,SUNDEv1ce,device'14 -.-,devzera;q")
# DEM 存放資料夾：預設為專案根目錄的 DEM
# run_all.py 位於 analyzeCS2，專案根目錄位於上層的 analyze
DEFAULT_DEM = os.path.abspath(os.path.join(BASE_DIR, "..", "DEM"))
# 絕對指定 DEM 路徑為上層 analyze 的 DEM
DEFAULT_DEM = os.path.abspath(os.path.join(BASE_DIR, "..", "DEM"))
# 忽略外部已設定的 DEM_FOLDER，直接使用 DEFAULT_DEM
DEM_FOLDER = DEFAULT_DEM
# 清理可能含相对路径的环境变量引入
DEM_FOLDER = os.path.abspath(DEM_FOLDER)


# 將必要環境變數匯出給 Go 程式
os.environ["TARGET_PLAYER"] = TARGET_PLAYERS
os.environ["DEM_FOLDER"] = DEM_FOLDER

# 確保輸出資料夾存在
for folder in ("output/kills", "output/puredata", "logs"):  
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# ====== 第一步：DEM 解壓與重命名 ======
def run_filter():
    cwd = os.path.join(BASE_DIR, "demfilter")
    print(f"▶ 執行 DEM 解壓腳本 (cwd: {cwd})")
    subprocess.run([sys.executable, os.path.join(cwd, "filter.py")], check=True, cwd=cwd)
    print("✅ filter 完成\n")

# ====== 第二步：擷取擊殺事件 ======
def run_kill():
    cwd = os.path.join(BASE_DIR, "catch_kill")
    print(f"▶ 執行 catchkillevent.go (cwd: {cwd})，DEM_FOLDER={DEM_FOLDER}")
    subprocess.run([
        "go", "run", os.path.join(cwd, "catchkillevent.go"),
        "-target", TARGET_PLAYERS,
        "-folder", DEM_FOLDER
    ], check=True, cwd=cwd, env=os.environ)
    print("✅ 擷取殺人事件完成\n")

# ====== 第三步：擷取 Tick 資料 ======
def run_tick():
    cwd = os.path.join(BASE_DIR, "catch_tick")
    print(f"▶ 執行 catchtickevent.go (cwd: {cwd})")
    subprocess.run([
        "go", "run", os.path.join(cwd, "catchtickevent.go"),
        "-folder", DEM_FOLDER
    ], check=True, cwd=cwd, env=os.environ)
    print("✅ 擷取 Tick 資料完成\n")

# ====== 主流程 ======
def main():
    # 打印最終確認
    print(f"▶ 最終 DEM_FOLDER:{DEM_FOLDER}")
    for func in (run_filter, run_kill, run_tick):
        try:
            func()
        except subprocess.CalledProcessError as e:
            print(f"❌ 執行失敗: {e}")
            sys.exit(e.returncode)
    print("🎉 所有流程執行完畢 🎉")

if __name__ == '__main__':
    main()