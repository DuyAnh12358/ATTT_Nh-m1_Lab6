import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("Đang tạo bộ dữ liệu giả")


X_fake, y_fake = make_classification(
    n_samples=5000,      
    n_features=18,       
    n_informative=10,    
    n_redundant=4,       
    n_classes=4,         
    weights=[0.7, 0.1, 0.1, 0.1], 
    random_state=42
)


feature_names = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s',
    'Pkt Len Mean', 'Pkt Len Std', 'SYN Flag Cnt',
    'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'URG Flag Cnt'
]


label_map = {0: 'BENIGN', 1: 'DDoS', 2: 'PortScan', 3: 'WebAttack'}
y_fake_text = [label_map[label] for label in y_fake]


X_df = pd.DataFrame(X_fake, columns=feature_names)
y_df = pd.DataFrame(y_fake_text, columns=['Label'])


X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)


X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)


print(f"Kích thước X_train: {X_train.shape}")
print(f"Kích thước X_test:  {X_test.shape}")