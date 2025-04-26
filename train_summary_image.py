import os
import json
import matplotlib.pyplot as plt

# ========== 輸入與輸出設定 ==========
log_path = "trainlog/report_summary.json"
plot_dir = "trainlog/plots"
os.makedirs(plot_dir, exist_ok=True)

# ========== 讀取 JSON 資料 ==========
with open(log_path, "r") as f:
    data = json.load(f)

# ========== 收集各種 early metrics ==========
labels = []
early_loss = []
early_acc = []
early_precision = []
early_recall = []
early_f1 = []

for entry in data:
    labels.append(entry["file"])
    early = entry.get("early_metrics", {})
    early_loss.append(early.get("loss", 0))
    early_acc.append(early.get("accuracy", 0))
    early_precision.append(early.get("precision", 0))
    early_recall.append(early.get("recall", 0))
    early_f1.append(early.get("f1_score", 0))

# ========== 繪製折線圖的工具函數 ==========
def plot_metric(values, ylabel, filename):
    plt.figure()
    plt.plot(labels, values, marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Transfer Files")
    plt.ylabel(ylabel)
    plt.title(f"Early {ylabel} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

# ========== 畫圖 ==========
plot_metric(early_loss, "Loss", "early_loss.png")
plot_metric(early_acc, "Accuracy", "early_accuracy.png")
plot_metric(early_precision, "Precision", "early_precision.png")
plot_metric(early_recall, "Recall", "early_recall.png")
plot_metric(early_f1, "F1 Score", "early_f1.png")

print("✅ 圖表已儲存在 trainlog/plots/")