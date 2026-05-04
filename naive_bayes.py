import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# =========================
# CLEAN DATA
# =========================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# =========================
# Xóa các dữ liệu bị trùng lặp trong dataset
# =========================
df.drop_duplicates(inplace=True)

print("\nClass Distribution:")
print(df["Label"].value_counts())

# =========================
# FEATURE SELECTION
# 18 đặc trưng quan trọng nhất
selected_features = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Packet Length Mean",
    "Packet Length Std",
    "SYN Flag Count",
    "ACK Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "URG Flag Count",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
]

selected_features = [f for f in selected_features if f in df.columns]

X = df[selected_features]
y = df["Label"]

# =========================
# TRAIN TEST SPLIT (CỐ ĐỊNH RANDOM)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True
)

# =========================
# SCALING (CHUẨN - KHÔNG LEAK)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODELS (2.5)
#  tạo 5 cách học khác nhau để phát hiện model tốt nhất
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# lưu kết quả của tất cả model
results = []
best_model = None
best_f1 = 0

# =========================
# TRAIN + EVALUATE
# =========================
for name, model in models.items():

    print("\n==============================")
    print("Model:", name)

    # model nào cần scale thì dùng dữ liệu đã scale, còn không thì dùng dữ liệu gốc.
    if name in ["SVM", "KNN", "Naive Bayes", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        X_eval = X_train_scaled
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        X_eval = X_train
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    # kết quả đánh giá model sau khi dự đoán
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Analyze:
    # vẽ bảng hiển thị model đoán đúng / sai bao nhiêu
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ✔ CROSS VALIDATION (CHỐNG ẢO)
    cv_score = cross_val_score(model, X_eval, y_train, cv=5).mean()
    print("Cross-validation score:", round(cv_score, 4))

    # kết quả để so sánh
    results.append(
        {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "CV": cv_score,
        }
    )

    # theo dõi và chọn ra model tốt nhất
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# =======tổng hợp kết quả cuối cùng của tất cả model==================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1", ascending=False)

print("\n==============================")
print("FINAL MODEL COMPARISON")
print(results_df)

# =========================
# SAVE MODEL
# =========================
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nBest Model:", best_model_name)
print("Model saved as best_model.pkl")
