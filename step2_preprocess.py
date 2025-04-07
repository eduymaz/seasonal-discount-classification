import pandas as pd
import numpy as np 
from scipy.stats.mstats import winsorize
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("./data/data.csv")

df['order_date'] = pd.to_datetime(df['order_date'])
df['total_price'] = ((1-df['discount'])*(df['unit_price']*df['quantity']))
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['day'] = df['order_date'].dt.day
df['yearquarter'] = df['year'].astype(str) + ' Q' + df['order_date'].dt.quarter.astype(str)


def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers


df['unit_price_winsorized_0.05'] = winsorize(df['unit_price'], limits=[0.05, 0.05])

# Bar Graphs 

plt.figure(figsize=(14, 6))

# 1. Sütun: Unit Price Dağılımı
plt.subplot(2, 2, 1)
sns.histplot(df['unit_price'], kde=True, color='blue')
plt.title('Original Unit Price Distribution')

plt.subplot(2, 2, 2)
sns.boxplot(y=df['unit_price'], color='blue')
plt.title('Original Unit Price Box Plot')

# 2. Sütun: Winsorized Unit Price Dağılımı
plt.subplot(2, 2, 3)
sns.histplot(df['unit_price_winsorized_0.05'], kde=True, color='red')
plt.title('Winsorized Unit Price Distribution')

plt.subplot(2, 2, 4)
sns.boxplot(y=df['unit_price_winsorized_0.05'], color='red')
plt.title('Winsorized Unit Price Box Plot')

plt.tight_layout()
plt.savefig("./results/bar_graphs_comparision.png")


# Correlation matrix

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
plt.title("Korelasyon Matrisi", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.savefig("./results/correlation_matrix.png")

# Label ekleme (discount_effective)
df['discount_effective'] = ((df['discount'] > 0) & 
                            (df['quantity'] > df.groupby(['product_id', 'yearquarter'])['quantity'].shift(1))
                           ).astype(int)


df.to_csv('./data/processed_data.csv', index=False)







