from datetime import datetime, timedelta
import pandas as pd
import os
import glob
import numpy as np
import json
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.pyplot as plt
from typing import List, Tuple
from decimal import Decimal
import seaborn as sns
import matplotlib as mpl

mpl.style.use('classic')
print(plt.style.available)

draw_pic = True
record_return = True


class Strategy:
    def __init__(self, id, min, open_sigma, stop_loss_sigma, REF, TARGET):
        self.id = id
        self.min = min
        self.open_sigma = open_sigma
        self.stop_loss_sigma = stop_loss_sigma
        self.ref_symbol = REF
        self.target_symbol = TARGET
        
        self.day = {}
        self.day_count = {}
        self.array = []
        self.dif_time = []
        self.lose_out_time = []
        
        self.ref = self._init_symbol_dict()
        self.target = self._init_symbol_dict()
        
        self.count = 0
        self.pos_count = 0
        self.pos_list = [0] * 5
        self.pnl_list = [0] * 5
        self.winloss_list = [[0, 0] for _ in range(5)]
        self.unit_profit = 0
        self.winner = 0
        self.loser = 0
        self.opentime = None
        self.closetime = None

    def _init_symbol_dict(self):
        return {
            "buy_ps": 0, "sell_ps": 0, "buy_avg": 0, "sell_avg": 0,
            "buy_size": 0, "sell_size": 0, "realize_pnl": 0, "unrealize_pnl": 0
        }

    def calculate_PnL1(self, symbol, side, price, size):
        data = self.ref if symbol == self.ref_symbol else self.target
        is_buy = side == "BUY"
        
        if is_buy:
            self._process_buy(data, price, size)
        else:
            self._process_sell(data, price, size)

    def _process_buy(self, data, price, size):
        if data["sell_size"] == 0:
            self._update_buy_data(data, price, size)
        else:
            self._process_buy_with_existing_sell(data, price, size)

    def _process_sell(self, data, price, size):
        if data["buy_size"] == 0:
            self._update_sell_data(data, price, size)
        else:
            self._process_sell_with_existing_buy(data, price, size)

    def _update_buy_data(self, data, price, size):
        data["buy_ps"] += price * size
        data["buy_size"] += size
        data["buy_avg"] = data["buy_ps"] / data["buy_size"]
        data["unrealize_pnl"] = (price - data["buy_avg"]) * data["buy_size"]

    def _update_sell_data(self, data, price, size):
        data["sell_ps"] += price * size
        data["sell_size"] += size
        data["sell_avg"] = data["sell_ps"] / data["sell_size"]
        data["unrealize_pnl"] = (data["sell_avg"] - price) * data["sell_size"]

    def _process_buy_with_existing_sell(self, data, price, size):
        if size < data["sell_size"]:
            data["realize_pnl"] += size * (data["sell_avg"] - price)
            data["sell_size"] -= size
            data["sell_ps"] -= data["sell_avg"] * size
            data["unrealize_pnl"] = (data["sell_avg"] - price) * data["sell_size"]
        elif size == data["sell_size"]:
            data["realize_pnl"] += data["sell_size"] * (data["sell_avg"] - price)
            data["sell_size"] = data["sell_ps"] = data["sell_avg"] = 0
            data["unrealize_pnl"] = 0
        else:
            data["realize_pnl"] += data["sell_size"] * (data["sell_avg"] - price)
            data["buy_size"] = size - data["sell_size"]
            data["buy_ps"] += price * data["buy_size"]
            data["buy_avg"] = data["buy_ps"] / data["buy_size"]
            data["sell_size"] = data["sell_ps"] = data["sell_avg"] = 0
            data["unrealize_pnl"] = (price - data["buy_avg"]) * data["buy_size"]

    def _process_sell_with_existing_buy(self, data, price, size):
        if size < data["buy_size"]:
            data["realize_pnl"] += size * (price - data["buy_avg"])
            data["buy_size"] -= size
            data["buy_ps"] -= data["buy_avg"] * size
            data["unrealize_pnl"] = (price - data["buy_avg"]) * data["buy_size"]
        elif size == data["buy_size"]:
            data["realize_pnl"] += data["buy_size"] * (price - data["buy_avg"])
            data["buy_size"] = data["buy_ps"] = data["buy_avg"] = 0
            data["unrealize_pnl"] = 0
        else:
            data["realize_pnl"] += data["buy_size"] * (price - data["buy_avg"])
            data["sell_size"] = size - data["buy_size"]
            data["sell_ps"] += price * data["sell_size"]
            data["sell_avg"] = data["sell_ps"] / data["sell_size"]
            data["buy_size"] = data["buy_ps"] = data["buy_avg"] = 0
            data["unrealize_pnl"] = (data["sell_avg"] - price) * data["sell_size"]

    def read_new_log(self, file_str):
        with open(file_str) as f:
            for i, line in enumerate(f):
                dic = json.loads(line)
                self.calculate_PnL1(
                    dic["msg"]["symbol"],
                    str(dic["msg"]["side"]),
                    Decimal(dic["msg"]["price"]).quantize(Decimal("0.1")),
                    Decimal(dic["msg"]["size"]).quantize(Decimal("0.01")),
                )
                
                if all(data["buy_size"] == 0 and data["sell_size"] == 0 for data in [self.ref, self.target]):
                    self._process_trade_completion(dic, i)

    def _process_trade_completion(self, dic, i):
        self.count += 1
        self.array.append(float((self.ref["realize_pnl"] + self.target["realize_pnl"]) / 2000))
        
        date_time_obj = datetime.strptime(f"{dic['msg']['date']} {dic['msg']['time']}", "%Y-%m-%d %H:%M:%S")
        self.closetime = date_time_obj
        
        print(dic['msg']['date'])
        print(self.ref["realize_pnl"] + self.target["realize_pnl"])
        print('============================')
        
        self._update_daily_stats(date_time_obj)
        self._update_trade_stats()
        
        self.ref["realize_pnl"] = self.target["realize_pnl"] = 0
        self.pos_count = 0
        
        if (i + 1) % 2 == 0:
            self.pos_count += 1

    def _update_daily_stats(self, date_time_obj):
        date_key = date_time_obj.strftime("%Y%m%d")
        if date_key not in self.day:
            self.day[date_key] = self.ref["realize_pnl"] + self.target["realize_pnl"] - self.pos_count * 200
            self.day_count[date_key] = 1
        else:
            self.day[date_key] += self.ref["realize_pnl"] + self.target["realize_pnl"] - self.pos_count * 200
            self.day_count[date_key] += 1

    def _update_trade_stats(self):
        total_pnl = self.ref["realize_pnl"] + self.target["realize_pnl"]
        if total_pnl > 0:
            self.winner += 1
        elif total_pnl < 0:
            self.loser += 1
            self.lose_out_time.append(self.closetime)
        self.unit_profit += total_pnl

    def connect_to_local(self, fileExt):
        self.read_new_log(fileExt)

    def return_dataframe(self):
        return [
            self.min, self.open_sigma, self.stop_loss_sigma, sum(self.pos_list),
            sum(self.pnl_list), 0, sum(self.pnl_list) / 60, "",
            self.winner, self.loser,
            np.mean(self.array) / np.std(self.array) * np.sqrt(365),
        ]

    def time_hold(self):
        for t in self.dif_time:
            totsec = t.total_seconds()
            h, m = divmod(totsec, 3600)
            m, _ = divmod(m, 60)
            print(f"{int(h)}:{int(m)}")

    def return_daily_return(self):
        return [[k, v] for k, v in self.day.items()]

    def return_dataframe_addpos(self):
        print(self.pos_list)
        print(self.pnl_list)
        print(sum(self.pnl_list))
        print(self.winner, self.loser)
        print(self.winloss_list)
        print("profit per day", sum(self.pnl_list) / 60)
        print("profit per trade:", sum(self.pnl_list) / sum(self.pos_list))
        
        result = [
            self.min, self.open_sigma, sum(self.pos_list), sum(self.pnl_list),
            sum(self.pnl_list) / sum(self.pos_list), sum(self.pnl_list) / 60,
            "", self.winner, self.loser
        ]
        
        for i in range(5):
            result.extend([
                self.pos_list[i] * (i + 1),
                self.pnl_list[i],
                f"{self.winloss_list[i][0]}/{self.winloss_list[i][1]}"
            ])
        
        return result

    def plot_lose_money_time_distribution(self):
        profit_file = "./LOSE_DISB/"
        os.makedirs(profit_file, exist_ok=True)
        picture_title = 'lose_out_time_distribution.png'
        
        timestamps = [(t + timedelta(hours=8)).time() for t in self.lose_out_time]
        
        plt.figure(figsize=(20, 10))
        plt.hist([t.hour * 3600 + t.minute * 60 + t.second for t in timestamps], bins=30, edgecolor='black')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.title('Time Distribution (Every 4th Log)')
        plt.xticks(range(0, 24*3600, 3600), [f"{h:02d}:00" for h in range(24)], rotation=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(profit_file, picture_title))
        plt.close()

    def plot_performance_with_dd(self, time, open_threshold, stoploss_threshold, window_size, period):
        capital = 184000 * 2 # 保證金
        profit_file = f"./PIC_BT_CN/{self.ref_symbol}_{self.target_symbol}/"
        os.makedirs(profit_file, exist_ok=True)
        picture_title = f"Pairs trading with {self.ref_symbol}_{self.target_symbol}_{window_size}_{time}min_open_{open_threshold}_stop_{stoploss_threshold}_{period}.png"
        
        dates = list(self.day.keys())
        total = np.cumsum([v for v in self.day.values()])
        total_with_capital = [float(v) / capital for v in total]
        
        mdd = calculate_mdd(total[::-1])
        win_rate = self.winner / (self.winner + self.loser) if (self.winner + self.loser) > 0 else 0
        
        dd = [total[i] - max(total[:i+1]) for i in range(len(total))]
        
        # r = pd.DataFrame(total_with_capital)
        # r_neg = pd.DataFrame([i for i in total_with_capital if i < 0])
        # 修改 Sharpe ratio 和 Sortino ratio 的計算
        returns = pd.Series(total_with_capital)
        sharp_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() != 0 else 0
        negative_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 and negative_returns.std() != 0 else 0
        
        highest_x = [total[i] for i in range(len(total)) if total[i] == max(total[:i+1]) and total[i] > 0]
        highest_dt = [i for i in range(len(total)) if total[i] == max(total[:i+1]) and total[i] > 0]
        
        #mpl.style.use("seaborn-v0_8-whitegrid")
        mpl.style.use('seaborn-darkgrid')
        #mpl.style.use('classic')
        color_list = ['g' if r > 0 else 'r' for r in total_with_capital]
        
        f, axarr = plt.subplots(3, sharex=True, figsize=(20, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
        
        axarr[0].plot(np.arange(len(dates)), total, color="b", zorder=1)
        axarr[0].scatter(highest_dt, highest_x, color="lime", marker="o", s=40, zorder=2)
        axarr[0].set_title(picture_title, fontsize=20)
        
        axarr[1].bar(np.arange(len(dates)), dd, color="red")
        axarr[2].bar(np.arange(len(dates)), total_with_capital, color=color_list)
        
        axarr[0].xaxis.set_major_locator(MultipleLocator(80))
        axarr[0].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: dates[int(x)] if 0 <= int(x) < len(dates) else ''))
        axarr[0].grid(True)
        
        shift = (max(total) - min(total)) / 20
        text_loc = max(total) - shift
        
        text_params = [
            (f"Total open number : {self.winner + self.loser}", 0),
            (f"Total profit : {total[-1]:.2f}", 1),
            (f"Total return(%) : {int((total[-1] / capital) * 100)}", 2),
            (f"Win rate : {win_rate:.2f}", 3),
            (f"Sharpe ratio : {sharp_ratio:.4f}", 4),
            (f"Sortino ratio : {sortino_ratio:.4f}", 5),
            (f"Max drawdown (%) : {round(mdd / capital, 4) * 100}", 6)
        ]
        
        for text, i in text_params:
            axarr[0].text(5, text_loc - shift * i, text, fontsize=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(profit_file, picture_title))
        plt.close()

def color_change(n):
    if n:
        return "g"
    else:
        return "r"


def calculate_mdd(cum_reward):
    mdd = 0
    low = -cum_reward[0]
    for r in cum_reward[1:]:
        mdd = max(mdd, low + r)
        low = max(low, -r)
    return mdd



def color_change(n):
    if n:
        return "g"
    else:
        return "r"


def calculate_mdd(cum_reward):
    mdd = 0
    low = -cum_reward[0]
    for r in cum_reward[1:]:
        mdd = max(mdd, low + r)
        low = max(low, -r)
    return mdd


def get_log_files(filename: str, ref: str, target: str, time: int, open_threshold: float, stoploss_threshold: float, window_size: int, period: str) -> List[str]:
    pattern = f"./Trading_Log_NEW/{filename}/_{filename}_{window_size}length_Trading_log/*{ref}{target}_{time}min_{open_threshold}_{stoploss_threshold}_{period}*.log"
    return sorted(glob.glob(pattern), key=os.path.getctime)

def process_strategy(data_path: str, time: int, open_threshold: float, stoploss_threshold: float, ref: str, target: str) -> Tuple[Strategy, List[List]]:
    strategy = Strategy("Allen", time, open_threshold, stoploss_threshold, ref, target)
    strategy.connect_to_local(data_path)
    return strategy, strategy.return_daily_return()

def apply_to_excel(ref: str, target: str, filename: str, test_second: int, open_threshold: float, stoploss_threshold: float, window_size: int, period: str, draw_pic: bool = True, record_return: bool = True):
    time = test_second // 60
    log_files = get_log_files(filename, ref, target, time, open_threshold, stoploss_threshold, window_size, period)
    print(f"Processing {len(log_files)} log files")

    for data_path in log_files:
        print(f"Processing file: {os.path.basename(data_path)}")
        strategy, daily_returns = process_strategy(data_path, time, open_threshold, stoploss_threshold, ref, target)

        if draw_pic:
            strategy.plot_performance_with_dd(time, open_threshold, stoploss_threshold, window_size, period)
            strategy.plot_lose_money_time_distribution()

        if record_return:
            for day_return in daily_returns:
                day_return[1] /= 20
                day_return.append(2000)
                #print(f"Date: {day_return[0]}, Return: {day_return[1]}%, Capital: {day_return[2]}")

        strategy.time_hold()

if __name__ == "__main__":
    REF, TARGET = 'CME_SI', "CME_GC"
    PERIOD = 'morning'  # or 'morning'
    TEST_SECOND = 60
    OPEN_THRESHOLD = 1.5
    STOPLOSS_THRESHOLD = 10.0
    WINDOW_SIZE = 90
    FILENAME = f"{REF}{TARGET}"

    print(f"Running strategy test for {FILENAME}")
    apply_to_excel(REF, TARGET, FILENAME, TEST_SECOND, OPEN_THRESHOLD, STOPLOSS_THRESHOLD, WINDOW_SIZE, PERIOD)

