import pandas as pd

# 读取 modified_data.csv 文件，获取 index 列
modified_df = pd.read_csv('modified_data.csv')

# 获取 'index' 列
indices_to_select = modified_df['index']

# 读取 train_data.csv 文件
train_data_df = pd.read_csv('train_data.csv')

# 根据 'index' 列筛选出 train_data.csv 中的行
filtered_train_data = train_data_df[train_data_df['index'].isin(indices_to_select)]

# 保存筛选后的数据到 train_data_modified.csv 文件
filtered_train_data.to_csv('train_data_modified.csv', index=False)

# 查看筛选后的数据
print(filtered_train_data.head())
