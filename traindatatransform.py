import pandas as pd
import numpy as np
import glob
import os

puredata_folder = "output/puredata"
transfer_folder = "output/transfer"
os.makedirs(transfer_folder, exist_ok=True)

group_size = 64

def process_file(puredata_path, transfer_path):
    if os.path.exists(transfer_path):
        print(f"✅ 已存在 {transfer_path}，跳過處理")
        return

    df = pd.read_csv(puredata_path)
    df.fillna(0, inplace=True)

    if "GroupID" not in df.columns or df.empty:
        print(f"⚠️ {puredata_path} 缺少 GroupID 欄位或為空，跳過")
        return

    valid_groups = []
    original_group_ids = []
    demo_ids = []

    grouped = df.groupby("GroupID")
    for group_id, group in grouped:
        if len(group) == group_size:
            valid_groups.append(group.copy())
            original_group_ids.extend(group["GroupID"].tolist())
            demo_ids.extend(group["DemoID"].tolist())

    if not valid_groups:
        print(f"⚠️ {puredata_path} 中無有效 group，跳過")
        return

    final_df = pd.concat(valid_groups, ignore_index=True)
    final_df["GroupID"] = original_group_ids
    final_df["DemoID"] = demo_ids  # 加入 DemoID 資訊

    # 正規化 Tick 和 ViewX
    final_df["Normalized_Tick"] = final_df.groupby("GroupID")["Tick"].transform(lambda x: x - x.min())
    final_df["ViewX"] = final_df.groupby("GroupID")["ViewX"].transform(lambda x: x - x.iloc[0])

    final_df["Delta_ViewX"] = final_df["ViewX"].shift(-1) - final_df["ViewX"]
    final_df["Delta_ViewY"] = final_df["ViewY"].shift(-1) - final_df["ViewY"]
    final_df["Delta_Tick"] = final_df["Tick"].shift(-1) - final_df["Tick"]

    final_df["Total_Angular_Velocity"] = np.sqrt(final_df["Delta_ViewX"]**2 + final_df["Delta_ViewY"]**2) / final_df["Delta_Tick"]
    final_df["Total_Angular_Velocity"] = final_df["Total_Angular_Velocity"].replace([np.inf, -np.inf], 0).fillna(0)

    final_df["Delta_Angular_Velocity"] = final_df["Total_Angular_Velocity"].shift(-1) - final_df["Total_Angular_Velocity"]
    final_df["Angular_Acceleration"] = final_df["Delta_Angular_Velocity"] / final_df["Delta_Tick"]
    final_df["Angular_Acceleration"] = final_df["Angular_Acceleration"].replace([np.inf, -np.inf], 0).fillna(0)

    final_df.drop(columns=["Delta_ViewX", "Delta_ViewY", "Delta_Tick", "Delta_Angular_Velocity"], inplace=True)
    final_df = final_df.round(4)

    final_df.to_csv(transfer_path, index=False)
    print(f"✅ 已儲存 {transfer_path}")

all_files = sorted(glob.glob(os.path.join(puredata_folder, "puredata*.csv")))

for file in all_files:
    basename = os.path.basename(file).replace("puredata", "").replace(".csv", "")
    transfer_file = os.path.join(transfer_folder, f"transfer{basename}.csv")
    process_file(file, transfer_file)
