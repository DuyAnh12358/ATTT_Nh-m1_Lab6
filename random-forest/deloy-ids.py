import pandas as pd
import random
import time
from sklearn.ensemble import RandomForestClassifier

print("1. Đang khởi động Hệ thống IDS...")

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test.csv') 
y_test = pd.read_csv('y_test.csv').values.ravel() 


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("[OK] Mô hình Random Forest đã sẵn sàng chiến đấu!\n")


def process_incoming_flow(model, flow_data, feature_names):
    """
    Hàm nhận 1 luồng mạng (1 dòng dữ liệu) và phân loại nó.
    """

    flow_df = pd.DataFrame([flow_data], columns=feature_names)
    
   
    prediction = model.predict(flow_df)[0]
    
 
    if prediction != 'BENIGN':
      
        dest_port = random.choice([80, 443, 21, 22, 8080, 3389])
        
      
        print(f"\033[91m[ALERT] Suspicious traffic detected: {prediction}. Destination Port: {dest_port}\033[0m")
    else:
  
        print(f"\033[92m[INFO] Normal traffic flow processed safely.\033[0m")

print("=== BẮT ĐẦU GIẢ LẬP BẮT GÓI TIN THỜI GIAN THỰC ===")
print("Nhấn Ctrl+C để dừng hệ thống.\n")


feature_columns = X_test.columns

try:
 
    for i in range(20): 
 
        random_index = random.randint(0, len(X_test) - 1)
        incoming_flow = X_test.iloc[random_index]
        actual_label = y_test[random_index] 
        
        print(f"Bắt được gói tin #{i+1} (Thực tế là: {actual_label}) -> Đang phân tích...")
        
        process_incoming_flow(rf_model, incoming_flow, feature_columns)
        print("-" * 60)
        
      
        time.sleep(1.0)
        
    print("Đã hoàn thành phiên quét demo 20 gói tin")
    
except KeyboardInterrupt:
    print("\n[!] Hệ thống IDS đã bị tắt bởi người dùng")