# ========== 程式碼 3：分析誤差分布與分類效果 ==========

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# ========== 參數 ==========
CSV_PATH = "tickwise_prediction_error.csv"

# ========== 讀取資料 ==========
df = pd.read_csv(CSV_PATH)
X = df[["Loss"]].values
y = df["Label"].values

# ========== 繪製誤差分布圖 ==========
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Loss", hue="Label", bins=40, kde=True, palette={0: "red", 1: "blue"})
plt.title("誤差分布圖：主角 vs 非主角")
plt.xlabel("MSE Loss")
plt.ylabel("樣本數")
plt.legend(labels=["非主角", "主角"])
plt.tight_layout()
plt.savefig("error_distribution.png")
plt.show()

# ========== 訓練分類器並顯示報告 ==========
clf = LogisticRegression()
clf.fit(X, y)
y_pred = clf.predict(X)
y_prob = clf.predict_proba(X)[:, 1]

print("\n📋 分類報告:")
print(classification_report(y, y_pred, digits=4))

# ========== 繪製 ROC 曲線 ==========
fpr, tpr, thresholds = roc_curve(y, y_prob)
auc = roc_auc_score(y, y_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC 曲線")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()