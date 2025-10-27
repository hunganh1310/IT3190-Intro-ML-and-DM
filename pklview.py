import pickle
import pandas as pd
import matplotlib.pyplot as plt

# 🔁 Đọc file .pkl (thay đường dẫn nếu cần)
with open('data/diabetes_train.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# 🔑 Kiểm tra các phần trong file
print("Các thành phần trong file:")
print(data_dict.keys())  # thường là ['data', 'target']

# 📦 Tách dữ liệu và nhãn
X = data_dict['data']     # Features (dữ liệu đầu vào)
y = data_dict['target']   # Target (nhãn cần dự đoán)

# 📊 Chuyển sang DataFrame để dễ xem
df = pd.DataFrame(X)
df['target'] = y  # thêm cột mục tiêu

# 🖨️ In 5 dòng đầu tiên
print("\n📄 5 dòng đầu tiên của dữ liệu:")
print(df.head())

# 📏 Kích thước dữ liệu
print(f"\n📐 Dữ liệu có {df.shape[0]} dòng và {df.shape[1]} cột.")

# 📈 Hiển thị thống kê mô tả
print("\n📊 Thống kê mô tả dữ liệu:")
print(df.describe())

# 📉 Biểu đồ phân phối của biến mục tiêu
plt.figure(figsize=(6,4))
plt.hist(df['target'], bins=30, color='skyblue', edgecolor='black')
plt.title('Biểu đồ phân phối biến mục tiêu (target)')
plt.xlabel('Giá trị target')
plt.ylabel('Số lượng')
plt.grid(True)
plt.tight_layout()
plt.show()
