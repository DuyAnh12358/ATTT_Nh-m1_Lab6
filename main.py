import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


# ===== HÀM CHẠY KNN =====
def run_knn():
    print("\n=== RUNNING KNN ===")

    # 1. LOAD DATA
    df = pd.read_csv("data.csv")

    # 2. CLEAN DATA
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    # 3. CHỌN FEATURE
    selected_features = [
        'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
        'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s',
        'Pkt Len Mean', 'Pkt Len Std', 'SYN Flag Cnt',
        'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
        'PSH Flag Cnt', 'URG Flag Cnt'
    ]

    X = df[selected_features]
    y = df['Label']

    # 4. ENCODE
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 5. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. SCALE
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 7. KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # 8. PREDICT
    y_pred = knn.predict(X_test)

    # 9. RESULT
    print("\n=== KNN RESULT ===")
    print(classification_report(y_test, y_pred))

    # 10. CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix - KNN")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ===== RESET LOOP =====
while True:
    user_input = input("\nNhấn Enter để chạy | gõ 'r' để reset | 'q' để thoát: ")

    if user_input.lower() == 'q':
        print("Thoát chương trình.")
        break

    elif user_input.lower() == 'r':
        print("Đang reset...\n")
        continue  # chạy lại vòng lặp

    else:
        run_knn()