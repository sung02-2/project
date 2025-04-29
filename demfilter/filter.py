import os
import glob
import json
import re
import subprocess
from datetime import datetime

# 資料夾設定
demstorage_folder = "../../DEM"
log_dir = "derarlog"
log_file = os.path.join(log_dir, "demRAR_log.json")
os.makedirs(log_dir, exist_ok=True)

# 載入已處理的 RAR 紀錄
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        processed_rars = set(json.load(f))
else:
    processed_rars = set()

# 持續處理直到沒有未處理的 RAR 為止
while True:
    rar_files = sorted(glob.glob(os.path.join(demstorage_folder, "*.rar")))
    unprocessed_rars = [r for r in rar_files if os.path.basename(r) not in processed_rars]
    if not unprocessed_rars:
        break

    rar = unprocessed_rars[0]
    rar_filename = os.path.basename(rar)
    print(f"\n📦 解壓縮中：{rar}")

    try:
        seven_zip_path = r"C:\\Program Files\\7-Zip\\7z.exe"
        result = subprocess.run(
            [seven_zip_path, "e", "-aoa", rar, f"-o{demstorage_folder}"],
            check=True,
            capture_output=True,
            text=True
        )
        processed_rars.add(rar_filename)

        # 從 stdout 抓出剛解壓的 .dem 檔案名稱
        dem_names = re.findall(r'Extracting\\s+(.+?\\.dem)', result.stdout, re.IGNORECASE)
        for name in dem_names:
            old_path = os.path.join(demstorage_folder, name)

            if not os.path.exists(old_path):
                print(f"⚠️ 找不到檔案：{name}，可能尚未寫入完成")
                continue

            mtime = os.path.getmtime(old_path)
            timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            new_name = f"{os.path.splitext(name)[0]}_{timestamp}.dem"
            new_path = os.path.join(demstorage_folder, new_name)

            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"📝 檔名已更改為：{new_name}")
            else:
                print(f"⚠️ 檔案已存在：{new_name}，略過改名")

        # 🧼 對資料夾中所有未加 timestamp 的 .dem 重新命名
        for path in glob.glob(os.path.join(demstorage_folder, "*.dem")):
            filename = os.path.basename(path)
            if re.search(r'_\d{4}-\d{2}-\d{2}\.dem$', filename):
                continue  # 已有 timestamp

            mtime = os.path.getmtime(path)
            timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            name_no_ext = os.path.splitext(filename)[0]
            new_name = f"{name_no_ext}_{timestamp}.dem"
            new_path = os.path.join(demstorage_folder, new_name)

            if not os.path.exists(new_path):
                os.rename(path, new_path)
                print(f"📝 後補改名為：{new_name}")
            else:
                print(f"⚠️ 後補檔案已存在：{new_name}，略過")

        # 每處理一個 RAR，立即更新紀錄
        with open(log_file, "w") as f:
            json.dump(sorted(processed_rars), f, indent=4)

    except Exception as e:
        print(f"❌ 無法解壓 {rar}：{e}")

print("\n🎉 解壓與紀錄完成！")