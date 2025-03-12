import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib 

# 读取Excel数据
df = pd.read_excel('data.xlsx')

# 初始化StandardScaler
scaler = StandardScaler()

# 对前10列进行标准化
columns_to_scale = df.columns[:11]  
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

#保存StandardScaler
joblib.dump(scaler, 'scaler.pkl')  # 保存StandardScaler

# 将标准化后的数据保存到新的Excel文件
df.to_excel('original_data.xlsx', index=False)

print("标准化处理完成，已生成新的Excel文件.xlsx")
