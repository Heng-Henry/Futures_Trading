# import pandas as pd
#
#
# def extract_time_range(input_file, target_name, start_datetime, end_datetime):
#     data = pd.read_csv(input_file, parse_dates=[['Date', 'Time']])
#
#     start_dt = pd.to_datetime(start_datetime)
#     end_dt = pd.to_datetime(end_datetime)
#
#     data = data[(data['Date_Time'] >= start_dt) & (data['Date_Time'] <= end_dt)]
#
#     data['Date'] = data['Date_Time'].dt.date
#     data['Time'] = data['Date_Time'].dt.time
#     data = data[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TotalVolume']]
#
#     output_filename = f"{target_name}_FROM_{start_dt.strftime('%Y%m%d_%H%M%S')}_TO_{end_dt.strftime('%Y%m%d_%H%M%S')}.txt"
#     data.to_csv(output_filename, index=False, quoting=1)
#
#
# target_name = "GC"
# extract_time_range(
#     input_file=f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CME\\CME.{target_name} HOT-Minute-Trade.txt",
#     target_name=target_name,
#     start_datetime="2014-01-31 09:00:00",
#     end_datetime="2014-01-31 12:00:00"
# )


import os
import pandas as pd


def extract_time_range(input_file, target_name, start_datetime, end_datetime):
    # 讀取資料
    data = pd.read_csv(input_file, parse_dates=[['Date', 'Time']])

    # 將 start_datetime 和 end_datetime 轉換為 datetime 格式
    start_dt = pd.to_datetime(start_datetime)
    end_dt = pd.to_datetime(end_datetime)

    # 過濾符合時間範圍的資料
    data = data[(data['Date_Time'] >= start_dt) & (data['Date_Time'] <= end_dt)]

    # 分離出 Date 和 Time 欄位
    data['Date'] = data['Date_Time'].dt.date
    data['Time'] = data['Date_Time'].dt.time
    data = data[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TotalVolume']]

    # 定義輸出檔案名和目錄
    output_filename = f"{target_name}_FROM_{start_dt.strftime('%Y%m%d_%H%M%S')}_TO_{end_dt.strftime('%Y%m%d_%H%M%S')}.txt"
    output_directory = "./test_trade"  # 指定目錄
    os.makedirs(output_directory, exist_ok=True)  # 創建目錄（如果不存在）

    # 儲存到指定目錄
    output_filepath = os.path.join(output_directory, output_filename)
    data.to_csv(output_filepath, index=False, quoting=1)



target_name = "SI"
extract_time_range(
    input_file=f"C:\\Users\\Henry\\Downloads\\Touchance\\Touchance\\CME\\CME.{target_name} HOT-Minute-Trade.txt",
    target_name=target_name,
    start_datetime="2020-10-01 01:00:00",
    end_datetime="2020-11-30 17:00:00"
)
