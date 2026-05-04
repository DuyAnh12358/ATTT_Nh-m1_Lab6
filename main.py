from src.preprocess import load_data, preprocess
from src.imbalance import balance_data
from src.svm_model import train_svm, evaluate, save_model
from sklearn.model_selection import train_test_split

# 1. Load data
df = load_data()

# ✅ FIX lỗi cột Label (có khoảng trắng)
df.columns = df.columns.str.strip()

# ✅ GIẢM DATA (tránh nặng máy)
df = df.sample(frac=0.05, random_state=42)

# 2. Preprocess
X, y, scaler, le = preprocess(df)

# 3. BỎ imbalance (SMOTE)
X_res, y_res = balance_data(X, y)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# 5. Train SVM
model = train_svm(X_train, y_train)

# 6. Evaluate
evaluate(model, X_test, y_test)

# 7. Save model
save_model(model, scaler, le)