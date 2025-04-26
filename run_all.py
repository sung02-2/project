import os
import subprocess
import glob
import json
from progress_handler_kill import load_progress, save_progress

# === 玩家設定與資料夾 ===
os.environ["TARGET_PLAYER"] = "device,dev1ce"
REPLY_FOLDER = "REPLY"
TARGET_PLAYERS = [name.strip() for name in os.getenv("TARGET_PLAYER", "device").split(",")]

for folder in ["output/kills", "output/labels", "output/puredata", "output/transfer", "logs"]:
    os.makedirs(folder, exist_ok=True)

# === Filter 搬運 DEM ===
def run_filter():
    print("\n📁 執行 filter.py 攝取未處理 DEM...")
    subprocess.run(["python", "demfilter/filter.py"], check=True)

# === 擷取 Kills（Go 程式） ===
def run_go_kill():
    print("\n📌 擷取所有 DEM 擊殺事件...")
    env = os.environ.copy()
    env['DEM_FOLDER'] = REPLY_FOLDER
    env['TARGET_PLAYER'] = ",".join(TARGET_PLAYERS)
    subprocess.run(["go", "run", "catch_kill/catchkillevent.go"], env=env, check=True)

# === 擷取 Tick（Go 程式） ===
def run_go_tick(index, dem_path):
    print(f"\n📌 擷取 Tick 資料：puredata{index}.csv")
    env = os.environ.copy()
    env['DEM_PATH'] = dem_path
    env['OUTPUT_INDEX'] = str(index)
    subprocess.run(["go", "run", "catch_tick/catchtickevent.go"], env=env, check=True)

# === 執行 traindatatransform.py ===
def run_transform():
    print("\n📌 資料轉換 (puredata → transfer)...")
    subprocess.run(["python", "traindatatransform.py"], check=True)

# === 主流程 ===
def main():
    run_filter()

    print("🔍 收集 DEM 檔案...")
    dem_files = sorted(glob.glob(os.path.join(REPLY_FOLDER, "game*.dem")))
    print("📂 找到 DEM：", dem_files)

    run_go_kill()

    kills_files = sorted(glob.glob("output/kills/kills*.json"))

    for kill_file in kills_files:
        basename = os.path.basename(kill_file)
        index = int(basename.replace("kills", "").replace(".json", ""))
        puredata_path = f"output/puredata/puredata{index}.csv"

        if os.path.exists(puredata_path):
            print(f"⚠️ {puredata_path} 已存在，略過 tick 擷取")
            continue

        with open(kill_file, "r") as f:
            data = json.load(f)
            if not data:
                continue
            demo_id = data[0]['demo_id']

        dem_path = os.path.join(REPLY_FOLDER, f"game{demo_id}.dem")
        if not os.path.exists(dem_path):
            print(f"❌ 找不到 DEM：{dem_path}，略過")
            continue

        run_go_tick(index, dem_path)

    run_transform()

    print("\n✅ 資料擷取與轉換完成（未進行模型訓練）")

if __name__ == "__main__":
    main()