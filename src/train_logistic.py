from preprocess import preprocess_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train():
    df = preprocess_pipeline()

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])

    X = df.drop('Label', axis=1)
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "models/logistic_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

if __name__ == "__main__":
    train()