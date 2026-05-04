import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import classification_report, confusion_matrix

print("1. Đang tải dữ liệu từ các file CSV...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

y_train = pd.read_csv('y_train.csv').values.ravel() 
y_test = pd.read_csv('y_test.csv').values.ravel()

print("2. Khởi tạo các mô hình...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000), 
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("3. Bắt đầu huấn luyện và đánh giá từng mô hình...\n")
for name, model in models.items():
    print(f"==================================================")
    print(f"--- Đang chạy mô hình: {name} ---")
    
   
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    
  
    print("\n[Báo cáo phân loại]")
    print(classification_report(y_test, y_pred, zero_division=0))
    
   
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Thực tế (Actual)')
    plt.xlabel('Dự đoán (Predicted)')
    
    print("Đang hiển thị biểu đồ Ma trận nhầm lẫn. Hãy tắt cửa sổ biểu đồ để chạy tiếp mô hình sau")
    plt.show() 
    print("\n")

