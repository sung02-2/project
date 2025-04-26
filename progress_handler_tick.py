import os
import json

TICK_PROGRESS_FILE = "logs/progress_tick.json"
os.makedirs("logs", exist_ok=True)

def load_tick_progress():
    try:
        if not os.path.exists(TICK_PROGRESS_FILE):
            return {"last_output_index": -1}
        with open(TICK_PROGRESS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 讀取 Tick 進度時發生錯誤：{e}")
        return {"last_output_index": -1}

def save_tick_progress(index):
    try:
        with open(TICK_PROGRESS_FILE, "w") as f:
            json.dump({"last_output_index": index}, f, indent=4)
    except Exception as e:
        print(f"❌ 儲存 Tick 進度失敗：{e}")
