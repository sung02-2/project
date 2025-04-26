import os
import shutil
import glob
import json
import re

demstorage_folder = "demfilter/demstorage"
reply_folder = "REPLY"
log_file = "logs/dem_log.json"

# 建立資料夾
os.makedirs(reply_folder, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 讀取已處理過的 dem 檔案名稱
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        processed_dems = set(json.load(f))
else:
    processed_dems = set()

# 抓出 REPLY 裡目前最大的 gameN.dem 編號
existing_files = glob.glob(os.path.join(reply_folder, "game*.dem"))
existing_indexes = []

for filename in existing_files:
    match = re.search(r"game(\d+)\.dem", filename)
    if match:
        existing_indexes.append(int(match.group(1)))

start_index = max(existing_indexes, default=0) + 1

# 所有未處理的 dem 檔案
all_dem_files = sorted(glob.glob(os.path.join(demstorage_folder, "*.dem")))

copied_count = 0

for dem_file in all_dem_files:
    dem_filename = os.path.basename(dem_file)

    if dem_filename in processed_dems:
        print(f"⚠️ {dem_filename} 已處理過，跳過")
        continue

    new_filename = f"game{start_index}.dem"
    new_path = os.path.join(reply_folder, new_filename)

    try:
        shutil.copy(dem_file, new_path)
        print(f"✅ 已將 {dem_filename} 複製為 {new_filename}")
        processed_dems.add(dem_filename)
        start_index += 1
        copied_count += 1

        # 寫入 log
        with open(log_file, "w") as f:
            json.dump(sorted(processed_dems), f, indent=4)

    except Exception as e:
        print(f"❌ 複製 {dem_filename} 失敗: {e}")

if copied_count == 0:
    print("⚠️ 沒有新的 DEM 被複製")
else:
    print(f"\n🎉 完成！共複製 {copied_count} 個 DEM 檔案")
