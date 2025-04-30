import pandas as pd
import numpy as np
import os

# ===== 主程式區塊：直接執行，不需外部呼叫 =====
if __name__ == "__main__":
    # 設定檔案路徑（可自行修改）
    input_path = "../output/puredata/puredata.csv"
    output_path = "../output/transfer/transferdata.csv"
    weapon_label_path = "weaponlabel.txt"

    # 讀入資料
    df = pd.read_csv(input_path)

    # 特徵擷取
    def extract_features(df, weapon_label_path="weaponlabel.txt"):
        df = df.copy()
        df.sort_values(by=["GroupID", "Tick"], inplace=True)

        # ===== 原始與差分視角特徵 =====
        df["ViewY_raw"] = df["ViewY"]
        df["DeltaViewX"] = df.groupby("GroupID")["ViewX"].diff().fillna(0)
        df["DeltaViewY"] = df.groupby("GroupID")["ViewY"].diff().fillna(0)
        df["YawVelocity"] = df["DeltaViewX"]
        df["PitchVelocity"] = df["DeltaViewY"]
        df["YawAccel"] = df.groupby("GroupID")["YawVelocity"].diff().fillna(0)
        df["PitchAccel"] = df.groupby("GroupID")["PitchVelocity"].diff().fillna(0)

        # ===== 移動特徵 =====
        df["Speed"] = np.sqrt(df["VelX"]**2 + df["VelY"]**2 + df["VelZ"]**2)
        df["Acceleration"] = df.groupby("GroupID")["Speed"].diff().fillna(0)
        df["MoveDir"] = np.degrees(np.arctan2(df["VelY"], df["VelX"]))

        # ===== 相對敵人資訊 =====
        dx = df["VictimPosX"] - df["PosX"]
        dy = df["VictimPosY"] - df["PosY"]
        dz = df["VictimPosZ"] - df["PosZ"]
        df["RelativeDistance"] = np.sqrt(dx**2 + dy**2 + dz**2)
        dvx = df["VictimVelX"] - df["VelX"]
        dvy = df["VictimVelY"] - df["VelY"]
        dvz = df["VictimVelZ"] - df["VelZ"]
        df["RelativeVelocity"] = np.sqrt(dvx**2 + dvy**2 + dvz**2)

        # ===== 瞄準角度差 =====
        enemy_angle = np.degrees(np.arctan2(dy, dx))
        df["AimOffset"] = (enemy_angle - df["ViewX"] + 180) % 360 - 180

        # ===== 視角與移動方向夾角 =====
        df["ViewMoveAngleDiff"] = (df["MoveDir"] - df["ViewX"] + 180) % 360 - 180

        # ===== Group 統計特徵 =====
        summary_list = []
        for group_id, group_df in df.groupby("GroupID"):
            delta_viewx = group_df["DeltaViewX"]
            movedir = group_df["MoveDir"]
            speed = group_df["Speed"]
            view_move_diff = group_df["ViewMoveAngleDiff"]

            if len(group_df) >= 2 and delta_viewx.std() != 0 and movedir.std() != 0:
                corr_viewX_moveAngle = np.corrcoef(delta_viewx, movedir)[0, 1]
                if np.isnan(corr_viewX_moveAngle): corr_viewX_moveAngle = 0
            else:
                corr_viewX_moveAngle = 0

            if len(group_df) >= 2 and delta_viewx.std() != 0 and speed.std() != 0:
                corr_viewX_speed = np.corrcoef(delta_viewx, speed)[0, 1]
                if np.isnan(corr_viewX_speed): corr_viewX_speed = 0
            else:
                corr_viewX_speed = 0

            turning_speed = speed[delta_viewx.abs() > 2]
            std_speed_while_turning = turning_speed.std() if len(turning_speed) > 0 else 0
            avg_angle_diff = view_move_diff.abs().mean()

            summary_list.append({
                "GroupID": group_id,
                "corr_viewX_moveAngle": corr_viewX_moveAngle,
                "corr_viewX_speed": corr_viewX_speed,
                "std_speed_while_turning": std_speed_while_turning,
                "avg_angle_diff": avg_angle_diff
            })

        df_summary = pd.DataFrame(summary_list)
        df = df.merge(df_summary, on="GroupID", how="left")

        # ===== Demo 日期處理 =====
        group_dates = df.groupby("GroupID")["DemoTime"].first().reset_index().rename(columns={"DemoTime": "StartDemoDate"})
        df = df.merge(group_dates, on="GroupID", how="left")
        df["DateOrdinal"] = pd.to_datetime(df["StartDemoDate"]).map(lambda d: d.toordinal())

        # ===== Tick 正規化處理 =====
        df["NormTick"] = df.groupby("GroupID").cumcount() / 63

        # ===== 武器數值化處理 =====
        label_map = {}
        if os.path.exists(weapon_label_path):
            with open(weapon_label_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name != "":
                        label_map[name.lower()] = i
        for w in df["WeaponName"].dropna().unique():
            lw = w.lower()
            if lw not in label_map:
                label_map[lw] = len(label_map)
        reverse_map = {v: k for k, v in label_map.items()}
        with open(weapon_label_path, "w", encoding="utf-8") as f:
            for i in range(len(reverse_map)):
                f.write(reverse_map[i] + "\n")
        df["WeaponLabel"] = df["WeaponName"].str.lower().map(label_map)

        return df

    df_features = extract_features(df, weapon_label_path=weapon_label_path)
    df_features.to_csv(output_path, index=False)
