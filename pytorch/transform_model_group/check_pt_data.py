import torch

# ===== 設定檔案路徑 =====
pt_path = "../../output/pt/train.pt"  # 你可以改為 val.pt 或 test.pt
data = torch.load(pt_path)

x_seq = data["x_seq"]
x_group = data["x_group"]
y = data["y"]
group_ids = data["group_ids"]

print(f"📦 檔案路徑：{pt_path}")
print(f"✔️ x_seq shape：{x_seq.shape}")        # [N, 64, 18]
print(f"✔️ x_group shape：{x_group.shape}")    # [N, 6]
print(f"✔️ y shape：{y.shape}")                # [N, 1]
print(f"✔️ group_ids 數量：{len(group_ids)}")

# ===== 資料值檢查 =====
has_nan_seq = torch.isnan(x_seq).any().item()
has_nan_group = torch.isnan(x_group).any().item()
has_nan_y = torch.isnan(y).any().item()
print("\n🔍 標籤 y 的唯一值：", torch.unique(y))
print("❓ 是否有 NaN：", has_nan_seq, has_nan_group, has_nan_y)
print("❓ 是否有 inf：", torch.isinf(x_seq).any().item(), torch.isinf(x_group).any().item(), torch.isinf(y).any().item())

# ===== 最大/最小值概覽 =====
print("\n📊 x_seq 最大值：", torch.max(x_seq).item(), "最小值：", torch.min(x_seq).item())
print("📊 x_group 最大值：", torch.max(x_group).item(), "最小值：", torch.min(x_group).item())

# ===== 如果有 NaN，列出是哪些 group 出問題 =====
if has_nan_group:
    print("\n❗ 以下 group 的 x_group 含 NaN：")
    nan_mask = torch.isnan(x_group).any(dim=1)  # 每一列是否含 NaN
    nan_indices = torch.where(nan_mask)[0]
    for idx in nan_indices[:10]:  # 最多列出前 10 個
        print(f"  ➤ GroupID: {group_ids[idx]} | x_group: {x_group[idx].tolist()}")
    print(f"🔢 共 {len(nan_indices)} 筆 x_group 含 NaN")

else:
    print("\n✅ 所有 x_group 資料都沒有 NaN")
