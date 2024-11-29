import sys
import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from typing import List, Tuple
import module.Johanson_class as Jo_class
from module.spreader import Spreader
from config import Pair_Trading_Config
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import math
# 設定當前工作目錄，放在import其他路徑模組之前
sys.path.append("./module")
os.chdir(sys.path[0])

record_time = []

def visualize_p_values_with_metrics(timestamps, p_values):
    print(f'record time{record_time}')
    # Combine date and time, then convert to datetime format
    datetime_series = pd.to_datetime([f"{date} {time}" for date, time in timestamps])

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({'Timestamp': datetime_series, 'P-Value': p_values})

    avg_p_value = np.mean(p_values)
    median_p_value = np.median(p_values)
    std_p_value = np.std(p_values)
    if len(p_values) != 0:
        coint_ratio = sum(p < 0.05 for p in p_values) / len(p_values)
    else:
        coint_ratio = 0


    print("Overall Cointegration Metrics:")
    print(f"Average : {avg_p_value:.4f}")
    print(f"Median : {median_p_value:.4f}")
    print(f'Std {std_p_value}')
    print(f"Cointegration Ratio (P-Value < 0.05): {coint_ratio:.2%}")


    plt.figure(figsize=(14, 7))
    plt.plot(df['Timestamp'], df['P-Value'], color='b', marker='o', label='P-Value')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')

    # Add horizontal lines for mean, median, and standard deviation
    plt.axhline(y=avg_p_value, color='g', linestyle='-', label=f'Mean: {avg_p_value:.4f}')
    # plt.axhline(y=median_p_value, color='orange', linestyle='-', label=f'Median: {median_p_value:.4f}')
    plt.axhline(y=avg_p_value + std_p_value, color='purple', linestyle='--',
                label=f'Mean + 1 Std: {avg_p_value + std_p_value:.4f}')
    plt.axhline(y=avg_p_value - std_p_value, color='purple', linestyle='--',
                label=f'Mean - 1 Std: {avg_p_value - std_p_value:.4f}')

    # Setting date format on x-axis
   # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))  # 每隔一天顯示一次

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))  # Adjust as necessary for readability

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Labels and title
    plt.xlabel('Timestamp')
    plt.ylabel('P-Value')
    plt.title('P-Value Over Time for Cointegration Test')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # dates = [ts[0] for ts in timestamps]
    # times = [ts[1] for ts in timestamps]
    #
    # df = pd.DataFrame({'Date': dates, 'Timestamp': times, 'P-Value': p_values})
    #
    #
    # avg_p_value = np.mean(p_values)
    # median_p_value = np.median(p_values)
    # if len(p_values) != 0:
    #     coint_ratio = sum(p < 0.05 for p in p_values) / len(p_values)
    # else:
    #     coint_ratio = 0
    #
    #
    # print("Overall Cointegration Metrics:")
    # print(f"Average P-Value: {avg_p_value:.4f}")
    # print(f"Median P-Value: {median_p_value:.4f}")
    # print(f"Cointegration Ratio (P-Value < 0.05): {coint_ratio:.2%}")
    #
    #
    # plt.figure(figsize=(14, 7))
    # plt.scatter(df['Timestamp'], df['P-Value'], color='b', label='P-Value')
    # plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
    #
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    #
    # plt.xticks(ticks=df['Timestamp'][::len(df) // 10], rotation=45, ha='right')
    #
    # plt.xlabel('Timestamp')
    # plt.ylabel('P-Value')
    # plt.title('P-Value Over Time for Cointegration Test')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

def load_and_preprocess_data(file_path: str, start_date: str) -> pd.DataFrame:
    """
    載入並預處理交易數據
    :param file_path: CSV文件路徑
    :param start_date: 回測起始日期 (YYYY-MM-DD)
    :return: 預處理後的DataFrame
    """
    df = pd.read_csv(file_path, sep=",")
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time']) - timedelta(hours=8)
    df = df[df['datetime'] >= start_date].reset_index(drop=True)
    df['Date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['Time'] = df['datetime'].dt.strftime('%H:%M:%S')
    df.drop('datetime', axis=1, inplace=True)
    return df

def adjust_prices(df: pd.DataFrame, multiplier: float, additional_multipliers: List[float] = None) -> pd.DataFrame:
    """
    調整價格數據
    :param df: 原始DataFrame
    :param multiplier: 主要價格調整乘數
    :param additional_multipliers: 額外的價格調整乘數列表
    :return: 調整後的DataFrame
    """
    price_columns = ['Open', 'High', 'Low', 'Close']
    df[price_columns] = df[price_columns].astype(float) * multiplier
    if additional_multipliers:
        for add_mult in additional_multipliers:
            df[price_columns] *= add_mult
    df['AVG'] = df[price_columns].mean(axis=1)
    return df

def get_end_datetime(ref_file, target_file):
    return 0
def update_current_end_datetime():
    pass

def observation():
    pass

def get_start_trading_time():
    return 0

def main(config: dict, period_choice: int):
    # 設定交易對和配置
    REF, TARGET = config['REF'], config['TARGET']
    trading_config = Pair_Trading_Config(REF, TARGET, config['open_threshold'], 
                                         config['stop_loss_threshold'], config['test_second'],
                                         config['window_size'])
    current_end_datetime = None
    # 設定日誌路徑
    log_path = f"./Trading_Log_NEW/{REF}{TARGET}/_{REF}{TARGET}_{config['window_size']}length_Trading_log/"
    os.makedirs(log_path, exist_ok=True)
    print(f"日誌目錄已創建: {log_path}")
    
    # 初始化交易機器人
    period = 'morning' if period_choice == 1 else 'night'
    spreader = Spreader(trading_config, REF, TARGET, log_path, period)
    record_time = spreader.record_time
    # 載入並預處理數據
    start_date = config['backtest_start_date']
    ref_df = load_and_preprocess_data(config['ref_file'], start_date)
    target_df = load_and_preprocess_data(config['target_file'], start_date)

    # 調整價格
    ref_df = adjust_prices(ref_df, config['ref_multiplier'])
    target_df = adjust_prices(target_df, config['target_multiplier'], config['target_additional_multipliers'])
    print(ref_df)
    print(target_df)


    # 設定交易時間
    time_open, time_end = config['trading_hours'][period]




    # 先去看資料最後到甚麼時候
    end_datetime = get_end_datetime(config['ref_file'], config['target_file'])
    # 設定觀察期
    while current_end_datetime < end_datetime:
        if observation(ref_df, target_df, spreader, REF, TARGET, interval=config['observation_period'],start_datetime, current_end_datetime):
            ### 觀察期通過就將接下來設成交易期，直到部位出掉
            start_time = get_start_trading_time()
            ### 更改simulate trading function 成出掉部位就會return
            simulate_trading(ref_df, target_df, spreader, REF, TARGET, time_open, time_end,start_time)
        # 更新目前資料的時間到哪了
        update_current_end_datetime()

    # 紀錄交易結果
    #spreader.predictor.existing_df.to_csv(f"./TABLE/{REF}_{TARGET}_formation_table.csv", index=False)

    # 模擬交易
   # simulate_trading(ref_df, target_df, spreader, REF, TARGET, time_open, time_end)
    
    # 保存預測結果
    #spreader.predictor.existing_df.to_csv(f"./TABLE/{REF}_{TARGET}_formation_table.csv", index=False)

def simulate_trading(ref_df: pd.DataFrame, target_df: pd.DataFrame, spreader: Spreader, 
                     REF: str, TARGET: str, time_open: str, time_end: str, ):
    """
    模擬交易過程
    :param ref_df: 參考資產DataFrame
    :param target_df: 目標資產DataFrame
    :param spreader: Spreader實例
    :param REF: 參考資產代碼
    :param TARGET: 目標資產代碼
    :param time_open: 交易開始時間
    :param time_end: 交易結束時間
    """
    i, j = 0, 0
    while i < len(ref_df) and j < len(target_df):
        ref_date, ref_time, ref_price = ref_df.iloc[i][['Date', 'Time', 'Close']]
        target_date, target_time, target_price = target_df.iloc[j][['Date', 'Time', 'Close']]
        
        if ref_date < target_date or (ref_date == target_date and ref_time < target_time):
            i += 1
        elif ref_date > target_date or (ref_date == target_date and ref_time > target_time):
            j += 1
        else:
            if time_open < ref_time <= time_end:
                print(f"模擬交易: {ref_date} {ref_time}")
                print(f"參考價格: {ref_price}, 目標價格: {target_price}")
                spreader.local_simulate(ref_date, ref_time, REF, ref_price, ref_price, time_end)
                spreader.local_simulate(target_date, target_time, TARGET, target_price, target_price, time_end)
            i += 1
            j += 1

if __name__ == "__main__":
    config = {
        'REF': 'CME_SI',
        'TARGET': 'CME_GC',
        'open_threshold': 1.5, #開倉門檻
        'stop_loss_threshold': 10, #平倉門檻
        'test_second': 60, #測試秒數 (收集幾分k)
        'window_size': 90, #kbar窗口大小
        'trading_hours': {
            'morning': ('07:01:00', '13:30:00'),
            'night': ('07:01:00', '21:00:00') # 新增：晚間交易時間
        },
       # 'ref_file': "C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CME\\CME.SI HOT-Minute-Trade.txt", # 配對1的交易數據
        #'target_file': "C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CME\\CME.GC HOT-Minute-Trade.txt", # 配對2的交易數據
        'ref_file':"C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\SI_FROM_20201020_010000_TO_20201030_170000.txt",
        'target_file':"C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\GC_FROM_20201020_010000_TO_20201030_170000.txt",
        #'ref_file': "C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\SI_FROM_20140131_010000_TO_20140131_170000.txt",
        #'target_file': "C:\\Users\\Henry\\Desktop\\NYCU Course\\113上\\實驗室\\PAIRTRADING-SIMULATION-main\\test_trade\\GC_FROM_20140131_010000_TO_20140131_170000.txt",
        'ref_multiplier': 5000, # 配對1價格乘數
        'target_multiplier': 100, # 配對2價格乘數
        'target_additional_multipliers': [30, 2],  # 新增：配對2額外價格乘數 (美金匯率 跟 張數)
        'backtest_start_date': '2020-01-01' , # 新增：回測起始日期
        'observation_period': 150
    }
    
    period_choice = 1  # 0 表示晚間交易, 1 表示早間交易
    main(config, period_choice)

    #
    # visualize_p_values_with_metrics(Jo_class.timestep, Jo_class.p_value_list)
    # print(Jo_class.timestep)
    # print(Jo_class.p_value_list)



    # 如果要測試不同的開倉閾值，可以使用以下代碼
    # open_threshold_list = [1.5, 2, 2.5, 3, 3.5]
    # for open_threshold in open_threshold_list:
    #     config['open_threshold'] = open_threshold
    #     main(config, period_choice)