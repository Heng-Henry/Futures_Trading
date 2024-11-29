import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

# 读取数据
target1 = 'SI'
target2 = 'GC'
start_datetime = '2020-10-01 09:00:00'  # 起始日期和时间
end_datetime = '2024-11-29 16:00:00'    # 结束日期和时间

asset1 = pd.read_csv(
    f"C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\SI_FROM_20201001_010000_TO_20201130_170000.txt",
    parse_dates=[['Date', 'Time']]
)
asset2 = pd.read_csv(
    f"C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\GC_FROM_20201001_010000_TO_20201130_170000.txt",
    parse_dates=[['Date', 'Time']]
)

# 将日期和时间合并为 Datetime 列
asset1.rename(columns={'Date_Time': 'Datetime'}, inplace=True)
asset2.rename(columns={'Date_Time': 'Datetime'}, inplace=True)

# 设置 Datetime 为索引
asset1.set_index('Datetime', inplace=True)
asset2.set_index('Datetime', inplace=True)

# 过滤数据范围（包括日期和时间）
asset1 = asset1[(asset1.index >= start_datetime) & (asset1.index <= end_datetime)]
asset2 = asset2[(asset2.index >= start_datetime) & (asset2.index <= end_datetime)]

# 提取每个交易的日期
asset1['Date'] = asset1.index.date
asset2['Date'] = asset2.index.date

# 找出两个标的共同交易的日期
common_dates = set(asset1['Date']).intersection(set(asset2['Date']))

# 定义计算协整系数的函数
def calculate_cointegration_w1_w2(price1, price2):
    log_price1 = np.log(price1)
    log_price2 = np.log(price2)

    # OLS 回归 log(price2) = w1 * log(price1) + w2
    X = sm.add_constant(log_price1)
    model = sm.OLS(log_price2, X).fit()
    w1, w2 = model.params[1], model.params[0]
    return w1, w2

# 按每日生成图表
if common_dates:
    common_dates = sorted(common_dates)
    for selected_date in common_dates:
        selected_date_str = selected_date.strftime('%Y-%m-%d')
        print(f"Processing date: {selected_date_str}")

        # 筛选当天数据
        asset1_selected = asset1[asset1['Date'] == selected_date]
        asset2_selected = asset2[asset2['Date'] == selected_date]

        # 筛选开盘前 150 分钟的数据
        open_time = pd.Timestamp(f"{selected_date} 09:00:00")
        end_open_time = open_time + pd.Timedelta(minutes=150)

        asset1_selected = asset1_selected[(asset1_selected.index >= open_time) & (asset1_selected.index < end_open_time)]
        asset2_selected = asset2_selected[(asset2_selected.index >= open_time) & (asset2_selected.index < end_open_time)]

        # 检查是否有足够数据
        if asset1_selected.empty or asset2_selected.empty:
            print(f"No data available for open period on {selected_date_str}. Skipping.")
            continue

        # 对齐数据，确保时间一致
        price1, price2 = asset1_selected['Close'].align(asset2_selected['Close'], join='inner')
        volume1, volume2 = asset1_selected['TotalVolume'].align(asset2_selected['TotalVolume'], join='inner')

        # 检查对齐后数据是否为空
        if price1.empty or price2.empty:
            print(f"No overlapping data for open period on {selected_date_str}. Skipping.")
            continue

        # 计算协整系数 w1 和 w2
        try:
            w1, w2 = calculate_cointegration_w1_w2(price1, price2)
        except Exception as e:
            print(f"Error in calculating cointegration for {selected_date_str}: {e}")
            continue

        # 计算价差
        spread = w1 * np.log(price1) + w2 * np.log(price2)

        # 合并数据
        merged_data = pd.DataFrame(index=price1.index)
        merged_data['Spread'] = spread
        merged_data['TotalVolume_Asset1'] = volume1
        merged_data['TotalVolume_Asset2'] = volume2
        merged_data['TotalVolume'] = volume1 + volume2

        # 前向填充缺失值
        merged_data = merged_data.asfreq('T')
        merged_data.fillna(method='ffill', inplace=True)

        # 计算统计量
        mean_spread = merged_data['Spread'].mean()
        std_spread = merged_data['Spread'].std()
        merged_data['Upper_1Std'] = mean_spread + std_spread
        merged_data['Lower_1Std'] = mean_spread - std_spread
        merged_data['Upper_2Std'] = mean_spread + 2 * std_spread
        merged_data['Lower_2Std'] = mean_spread - 2 * std_spread

        # 绘图
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # 绘制价差
        ax1.plot(merged_data.index, merged_data['Spread'], label='Spread (w1*log(price1) - w2*log(price2))', color='blue')
        ax1.axhline(mean_spread, color='green', linestyle='--', label='Mean Spread', linewidth=1.5)
        ax1.axhline(mean_spread + std_spread, color='purple', linestyle='--', label='Mean + 1 Std', linewidth=1.2)
        ax1.axhline(mean_spread - std_spread, color='purple', linestyle='--', label='Mean - 1 Std', linewidth=1.2)
        ax1.axhline(mean_spread + 2 * std_spread, color='red', linestyle='--', label='Mean + 2 Std', linewidth=1.2)
        ax1.axhline(mean_spread - 2 * std_spread, color='red', linestyle='--', label='Mean - 2 Std', linewidth=1.2)

        ax1.set_xlabel('Datetime')
        ax1.set_ylabel('Spread', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # 绘制总成交量
        ax2 = ax1.twinx()
        ax2.bar(merged_data.index, merged_data['TotalVolume'], label='Total Volume (Asset1 + Asset2)', color='orange', alpha=0.3, width=0.0004)
        ax2.set_ylabel('Total Volume', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # 标题和布局
        fig.suptitle(f'Price Spread and Total Volume (Pre-open 150 Minutes) on {selected_date_str}')
        fig.tight_layout()

        # 保存图表
        output_path = os.path.join(f'20241116_150min', f'{target1} and {target2} {selected_date_str} preopen spread and volume')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path + '.png')
        plt.close(fig)

else:
    print("No common trading dates found.")
