# progress_handler_kill.py

import json
import os

PROGRESS_FILE = "logs/progress_kill.json"
os.makedirs("logs", exist_ok=True)

def load_progress():
    try:
        if not os.path.exists(PROGRESS_FILE):
            return {"last_demo_index": -1, "last_group_index": -1}
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 讀取進度檔案時發生錯誤：{e}")
        return {"last_demo_index": -1, "last_group_index": -1}

def save_progress(last_demo_index, last_group_index):
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({
                "last_demo_index": last_demo_index,
                "last_group_index": last_group_index
            }, f, indent=4)
    except Exception as e:
        print(f"❌ 儲存進度時失敗：{e}")

def reset_progress():
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print("🔄 已重設 progress_kill.json")
    except Exception as e:
        print(f"❌ 重設進度時發生錯誤：{e}")

def print_progress():
    progress = load_progress()
    print(f"📌 目前進度（KILL）：demo index = {progress['last_demo_index']}, group index = {progress['last_group_index']}")
