import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.api as sm

# 讀取數據
target1 = 'SI'
target2 = 'GC'
range_start = '2020-10-10'
range_end = '2020-11-10'

asset1 = pd.read_csv(
    r"C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\SI_FROM_20201001_010000_TO_20201130_170000.txt",
    parse_dates=[['Date', 'Time']]
)
asset2 = pd.read_csv(
    r"C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\GC_FROM_20201001_010000_TO_20201130_170000.txt",
    parse_dates=[['Date', 'Time']]
)

# 將日期和時間合併為Datetime欄位
asset1.rename(columns={'Date_Time': 'Datetime'}, inplace=True)
asset2.rename(columns={'Date_Time': 'Datetime'}, inplace=True)

# 設置Datetime為索引
asset1.set_index('Datetime', inplace=True)
asset2.set_index('Datetime', inplace=True)

# 篩選日期範圍內的數據
asset1 = asset1[(asset1.index >= range_start) & (asset1.index <= range_end)]
asset2 = asset2[(asset2.index >= range_start) & (asset2.index <= range_end)]


# 计算协整关系的 w1 和 w2
def calculate_cointegration_w1_w2(price1, price2):
    log_price1 = np.log(price1)
    log_price2 = np.log(price2)

    # OLS 拟合 log(price2) = w1 * log(price1) + w2
    X = sm.add_constant(log_price1)
    model = sm.OLS(log_price2, X).fit()
    w1, w2 = model.params[1], model.params[0]
    return w1, w2


# 计算价差
def calculate_spread(price1, price2, w1, w2):
    log_price1 = np.log(price1)
    log_price2 = np.log(price2)
    spread = w1 * log_price1 - w2 * log_price2
    return spread


# 获取价格和成交量
price1 = asset1['Close']
price2 = asset2['Close']
volume1 = asset1['TotalVolume']
volume2 = asset2['TotalVolume']
price1, price2 = asset1['Close'].align(asset2['Close'], join='inner')

assert price1.index.equals(price2.index), "Indices of price1 and price2 are not aligned!"

# 计算 w1 和 w2
w1, w2 = calculate_cointegration_w1_w2(price1, price2)

# 计算价差
spread = calculate_spread(price1, price2, w1, w2)

# 合并数据
merged_data = pd.DataFrame(index=asset1.index)
merged_data['Spread'] = spread
merged_data['TotalVolume_Asset1'] = volume1
merged_data['TotalVolume_Asset2'] = volume2
merged_data['TotalVolume'] = volume1 + volume2

# 繪製圖表
fig, ax1 = plt.subplots(figsize=(14, 8))

# 價差圖
ax1.plot(merged_data.index, merged_data['Spread'], label='Spread (w1*log(price1) - w2*log(price2))', color='blue')
ax1.axhline(merged_data['Spread'].mean(), color='green', linestyle='--', label='Mean Spread', linewidth=1.5)
ax1.axhline(merged_data['Spread'].mean() + merged_data['Spread'].std(), color='purple', linestyle='--',
            label='Mean + 1 Std', linewidth=1.2)
ax1.axhline(merged_data['Spread'].mean() - merged_data['Spread'].std(), color='purple', linestyle='--',
            label='Mean - 1 Std', linewidth=1.2)
ax1.set_xlabel('Datetime')
ax1.set_ylabel('Spread', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 成交量圖
ax2 = ax1.twinx()
ax2.plot(merged_data.index, merged_data['TotalVolume'], label='Total Volume', color='orange', alpha=0.7)
ax2.set_ylabel('Total Volume', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 標題和佈局
fig.suptitle(f'Price Spread and Total Volume from {range_start} to {range_end}')
fig.tight_layout()

# 保存圖表
output_path = os.path.join(f'20241116', f'{target1} and {target2} from {range_start} to {range_end}')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fig.savefig(output_path + '.png')
plt.close(fig)

print(f"Visualization saved to {output_path}.png")
