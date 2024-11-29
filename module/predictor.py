from asyncio.log import logger
#from pty import slave_open
#from turtle import position
#from attr import s
import numpy as np
import collections
import time
import module.PTwithTimeTrend_AllStock as ptm
import pandas as pd
from scipy.ndimage.interpolation import shift
from datetime import timedelta, datetime
import os
from decimal import Decimal
import decimal
import math
import sys

dtype = {
    'S1': str,
    'S2': str,
    'VECMQ': float,
    'mu': float,
    'Johansen_slope': float,
    'stdev': float,
    'model': int,
    'w1': float,
    'w2': float
}
CLOSE_POSITION = {
    "BUY": "SELL",
    "SELL": "BUY"
}
def find_decimals(value):
    return (abs(decimal.Decimal(str(value)).as_tuple().exponent))
def round_price(x, precision_price):
    return float(Decimal(math.ceil(x * 10 **(find_decimals(precision_price)))/10 **(find_decimals(precision_price))).quantize(precision_price))


def trunc_amount(x, precision_amount):
    return float(Decimal(math.ceil(x * 10 **(find_decimals(precision_amount)))/10 **(find_decimals(precision_amount))).quantize(precision_amount))


def makehash():
    return collections.defaultdict(makehash)


class SpreadQuotes:
    spread_price = makehash()
    spread_size = makehash()
    spread_symbol = makehash()
    def __init__(self,ref_symbol,target_symbol):
        self.ref = ref_symbol
        self.target = target_symbol
        
    def set_size(self, symbol, size):
        assert symbol in [self.ref, self.target]
        self.spread_size[symbol] = size

    def get_size(self, symbol):
        assert symbol in [self.ref, self.target]
        return self.spread_size[symbol]

    def set_price(self, symbol, price):
        self.spread_price[symbol] = price

    def get_price(self, symbol):
        assert symbol in [self.ref, self.target]
        return self.spread_price[symbol]

    def set_side(self, symbol, side):
        self.spread_symbol[symbol] = side

    def get_side(self, symbol):
        assert symbol in [self.ref, self.target]
        return self.spread_symbol[symbol]


class Spreads:

    # index = 0
    # is_warmed_up = False

    def __init__(self, window_size):
        self.xs = np.zeros(window_size)
        self.window_size = window_size
        self.index = 0
        self.is_warmed_up = False

    def update(self, x):

        if self.index == self.window_size:
            self.xs = shift(self.xs, -1, cval=0)
            self.index = self.window_size-1
        self.xs[self.index % self.window_size] = x
        print(self.xs)
        if self.index == self.window_size - 1:
            self.is_warmed_up = True
        self.index += 1


class Predictor:
    c = 0 
    five_min_timestamp_1 = 0
    five_min_timestamp_2 = 0
    sec_timestamp_1 = 0
    sec_timestamp_2 = 0

    def __init__(self, window_size, ref_symbol, target_symbol, slippage,log,config):
        self.window_size = window_size
        self.ref_symbol = ref_symbol
        self.target_symbol = target_symbol
        self.ref_spreads = Spreads(self.window_size)
        self.target_spreads = Spreads(self.window_size)
        self.ref_timestamp = 0
        self.target_timestamp = 0
        self.slippage = slippage
        self.spread_quotes = SpreadQuotes(self.ref_symbol,self.target_symbol)
        self.logger = log
        self.position = 0
        self.table = {
            "w1": 0,
            "w2": 0,
            "mu": 0,
            "stdev": 0,
            "model": 0,
            "capital": 2000
        }
        self.ref_size = 0
        self.target_size = 0
        self.cointegration_check = False
        self.timestamp_check = False
        self.count = 0
        self.config = config
    def get_asks(self, orderbook):
        ref_ask = None
        target_ask = None
        if orderbook[self.ref_symbol]['sellQuote'] and orderbook[self.target_symbol]['sellQuote']:
            ref_ask = float(orderbook[self.ref_symbol]
                            ['sellQuote'][0]['price'][0])
            target_ask = float(
                orderbook[self.target_symbol]['sellQuote'][0]['price'][0])
        return ref_ask, target_ask

    def get_bids(self, orderbook):
        ref_bid = None
        target_bid = None
        if orderbook[self.ref_symbol]['buyQuote'] and orderbook[self.target_symbol]['buyQuote']:
            ref_bid = float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][0])
            target_bid = float(
                orderbook[self.target_symbol]['buyQuote'][0]['price'][0])
        return ref_bid, target_bid
    def get_level_asks(self, orderbook):
        ref_ask = None
        target_ask = None
        if orderbook[self.ref_symbol]['sellQuote'] and orderbook[self.target_symbol]['sellQuote']:
            ref_ask = (float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][0]) + float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][1]) + float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][2])) / 3
            target_ask = (float(orderbook[self.target_symbol]['sellQuote'][0]['price'][0]) + float(orderbook[self.target_symbol]['sellQuote'][0]['price'][1]) + float(orderbook[self.target_symbol]['sellQuote'][0]['price'][2])) / 3
        return ref_ask, target_ask

    def get_level_bids(self, orderbook):
        ref_bid = None
        target_bid = None
        if orderbook[self.ref_symbol]['buyQuote'] and orderbook[self.target_symbol]['buyQuote']:
            ref_bid = (float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][0]) + float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][1]) + float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][2])) / 3

            target_bid = (float(orderbook[self.target_symbol]['buyQuote'][0]['price'][0]) + float(orderbook[self.target_symbol]['buyQuote'][0]['price'][1]) + float(orderbook[self.target_symbol]['buyQuote'][0]['price'][2])) / 3
        return ref_bid, target_bid

    def update_spreads(self, orderbook):
        if self.ref_symbol in orderbook and self.target_symbol in orderbook and orderbook[self.ref_symbol]['timestamp'] != self.ref_timestamp and orderbook[self.target_symbol]['timestamp'] != self.target_timestamp :
            # if True :
            self.target_timestamp = orderbook[self.target_symbol]['timestamp']
            self.ref_timestamp = orderbook[self.ref_symbol]['timestamp']
            print(self.ref_timestamp,self.target_timestamp)
            ref_ask, target_ask = self.get_asks(orderbook)
            ref_bid, target_bid = self.get_bids(orderbook)
            ref_mid_price = (ref_ask + ref_bid) / 2  # ref mid price
            target_mid_price = (target_ask + target_bid) / \
                2  # target mid price
            #print(self.ref_timestamp - self.target_timestamp)
            print(f"ref :{ref_mid_price} , target : {target_mid_price}")
            if ref_ask and target_ask and ref_bid and target_bid:
                self.ref_spreads.update(ref_mid_price)
                self.target_spreads.update(target_mid_price)


    def cointegration_test(self,):
        #print("in cointegration")
        tmp = {self.ref_symbol: self.ref_spreads.xs,
               self.target_symbol: self.target_spreads.xs}
        price_series = [[r, t] for r, t in zip(
            self.ref_spreads.xs, self.target_spreads.xs)]
        price_series = np.array(price_series)
        #print("prices series",price_series)
        price_data = pd.DataFrame(tmp)
        #dailytable = ptm.formation_table(price_data,self.window_size)
        dailytable = ptm.refactor_formation_table(
            price_series, self.window_size)
        if len(dailytable) != 0:
            return dailytable[0], dailytable[1], dailytable[2], dailytable[3], dailytable[4]
        else:
            return 0, 0, 0, 0, 0

    def slippage_number(self, x, size):
        neg = x * (-1)
        if self.position == -1:
            return neg if size > 0 else x
        elif self.position == 1:
            return neg if size < 0 else x

    def side_determination(self, size):
        if self.position == -1:
            return "SELL" if size > 0 else "BUY"
        elif self.position == 1:
            return "SELL" if size < 0 else "BUY"

    def open_Quotes_setting(self, ref_trade_price, target_trade_price,timestamp):
        slippage = self.slippage
        self.ref_size, self.target_size = self.table["w1"] * self.table["capital"] / \
            ref_trade_price, self.table["w2"] * \
            self.table["capital"] / target_trade_price
        self.spread_quotes.set_price(
            self.ref_symbol, ref_trade_price * (1 + self.slippage_number(slippage, self.ref_size)))
        self.spread_quotes.set_price(
            self.target_symbol, target_trade_price * (1 + self.slippage_number(slippage, self.target_size)))
        self.spread_quotes.set_size(
            self.ref_symbol, abs(self.ref_size))
        self.spread_quotes.set_size(
            self.target_symbol, abs(self.target_size))
        self.spread_quotes.set_side(
            self.ref_symbol, self.side_determination(self.ref_size)
        )
        self.spread_quotes.set_side(
            self.target_symbol, self.side_determination(self.target_size)
        )
        print(f'reference weight : {self.table["w1"]} , target weight : {self.table["w2"]}')
        print(f'reference_price = {ref_trade_price * (1 + self.slippage_number(slippage,self.ref_size))} . size = {abs(self.ref_size)} , side = {self.side_determination(self.ref_size)}')
        print(f'target_price = {target_trade_price *(1 + self.slippage_number(slippage,self.target_size))} . size = {abs(self.target_size)} , side = {self.side_determination(self.target_size)}')
        self.logger.fills('Binance', str(timestamp), self.ref_symbol,
                           'LIMIT', self.side_determination(self.ref_size), round_price(ref_trade_price,self.config.REF_PRICE_PRECISION), trunc_amount(abs(self.ref_size),self.config.REF_AMOUNT_PRECISION))
        self.logger.fills('Binance', str(timestamp), self.target_symbol,
                           'LIMIT', self.side_determination(self.target_size), round_price(target_trade_price,self.config.TARGET_PRICE_PRECISION), trunc_amount(abs(self.target_size),self.config.TARGET_AMOUNT_PRECISION))
    def close_Quotes_setting(self, ref_trade_price, target_trade_price,timestamp):
        slippage = self.slippage
        # up -> size < 0 -> buy -> ask
        self.spread_quotes.set_price(
            self.ref_symbol, ref_trade_price * (1 - self.slippage_number(slippage, self.ref_size)))
        self.spread_quotes.set_price(
            self.target_symbol, target_trade_price * (1 - self.slippage_number(slippage, self.target_size)))
        self.spread_quotes.set_size(
            self.ref_symbol, abs(self.ref_size))
        self.spread_quotes.set_size(
            self.target_symbol, abs(self.target_size))
        self.spread_quotes.set_side(
            self.ref_symbol, CLOSE_POSITION[self.side_determination(
                self.ref_size)]
        )
        self.spread_quotes.set_side(
            self.target_symbol, CLOSE_POSITION[self.side_determination(
                self.target_size)]
        )
        print(f'reference_price = {ref_trade_price * (1 - self.slippage_number(slippage,self.ref_size))} . size = {abs(self.ref_size)} , side = {CLOSE_POSITION[self.side_determination(self.ref_size)]}')
        print(f'target_price = {target_trade_price *(1 - self.slippage_number(slippage,self.target_size))} . size = {abs(self.target_size)} , side = {CLOSE_POSITION[self.side_determination(self.target_size)]}')
        self.logger.fills('Binance', str(timestamp), self.ref_symbol,
                           'LIMIT', CLOSE_POSITION[self.side_determination(self.ref_size)], round_price(ref_trade_price,self.config.REF_PRICE_PRECISION), trunc_amount(abs(self.ref_size),self.config.REF_AMOUNT_PRECISION))
        self.logger.fills('Binance', str(timestamp), self.target_symbol,
                           'LIMIT', CLOSE_POSITION[self.side_determination(self.target_size)], round_price(target_trade_price,self.config.TARGET_PRICE_PRECISION), trunc_amount(abs(self.target_size),self.config.TARGET_AMOUNT_PRECISION))
        self.position = 0

    def get_target_spread_price(self, orderbook, orderbook_5min, open_threshold, stop_loss_threshold):
        
        if self.ref_spreads.is_warmed_up and self.target_spreads.is_warmed_up and orderbook[self.ref_symbol]['timestamp'] != self.sec_timestamp_1 and orderbook[self.target_symbol]['timestamp'] != self.sec_timestamp_2:
            t = orderbook[self.ref_symbol]['timestamp']
            ref_ask, target_ask = self.get_asks(orderbook)
            ref_bid, target_bid = self.get_bids(orderbook)
            ref_mid_price = (ref_ask + ref_bid) / 2  # ref mid price
            target_mid_price = (target_ask + target_bid) / 2

            self.sec_timestamp_1 = orderbook[self.ref_symbol]['timestamp']
            self.sec_timestamp_2 = orderbook[self.target_symbol]['timestamp']
            # ref_ask, target_ask = self.get_level_asks(orderbook)
            # ref_bid, target_bid = self.get_level_bids(orderbook)
            '''
            if self.position == 0 :
                self.position = -1
                self.ref_size = -0.01 
                self.target_size = 0.05
                if self.ref_size < 0 and self.target_size > 0:
                            self.open_Quotes_setting(ref_ask,target_bid)
                            print(ref_ask,target_bid)
                            return self.spread_quotes
            elif self.position == -1 and self.count == 30 :
                #self.position = 888 
                self.count = 0 
                if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid,target_ask)
                            print(ref_bid,target_ask)
                            return self.spread_quotes
            elif self.position == -1 :
                self.count += 1
            '''
            if self.five_min_timestamp_1 != orderbook_5min[self.ref_symbol]['timestamp'] and self.five_min_timestamp_2 != orderbook_5min[self.target_symbol]['timestamp']:
                self.five_min_timestamp_1 = orderbook_5min[self.ref_symbol]['timestamp']
                self.five_min_timestamp_2 = orderbook_5min[self.target_symbol]['timestamp']
                self.cointegration_check = False
                self.timestamp_check = True
            else:
                self.timestamp_check = False

            if self.position == 0 and self.cointegration_check is False and self.timestamp_check is True:
                print("in test cointegration")
                #raw_input()
                mu, stdev, model, w1, w2 = self.cointegration_test()
                print(mu, stdev, model, w1, w2)
                if model > 0 and model < 4 and w1 * w2 < 0  and (abs(w1) * 2000 < self.config.pos_ratio and abs(w2) * 2000 < self.config.pos_ratio):
                    self.cointegration_check = True
                    self.table = {
                        "w1": float(w1),
                        "w2": float(w2),
                        "mu": float(mu),
                        "stdev": float(stdev),
                        "model": model,
                        "capital": 2000
                    }
                    # path1= f"/home/allen.kuo/check_data/"
                    # path2 = "/home/allen.kuo/output_data/"
                    # isExist1 = os.path.exists(path1)
                    # if not isExist1:    
                    #     # Create a new directory because it does not exist 
                    #     os.makedirs(path1)
                    #     print("The new directory is created!")
                    # isExist2 = os.path.exists(path2)
                    # if not isExist2:    
                    #     # Create a new directory because it does not exist 
                    #     os.makedirs(path2)
                    #     print("The new directory is created!")
                    # check_data = {self.ref_symbol : list(self.ref_spreads.xs),
                    #             self.target_symbol : list(self.target_spreads.xs),
                    #                 }
                    # print(w1,w2,mu,stdev)
                    # output_data = {"w1": [float(w1)],
                    #             "w2": [float(w2)],
                    #             "mu": [float(mu)],
                    #             "stdev": [float(stdev)],
                    #             "model": [model[0]]}
                    # print("------record check data ------")
                    # check_data = pd.DataFrame(check_data)
                    # output_data = pd.DataFrame(output_data)
                    # check_data.to_csv(f"{path1}check_data_{self.c}.csv")
                    # output_data.to_csv(f"{path2}output_data_{self.c}.csv")
                    # self.c += 1
            if self.position == 0 and self.cointegration_check == True:

                #print(spread_stamp,open_threshold * self.table['stdev'],self.table['mu'])
                if self.table["w1"] < 0 and self.table["w2"] > 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_bid)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_ask)

                elif self.table["w1"] > 0 and self.table["w2"] < 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_ask)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_bid)

                elif self.table["w1"] > 0 and self.table["w2"] > 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_bid)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_ask)

                elif self.table["w1"] < 0 and self.table["w2"] < 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_ask)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_bid)

                if spread_stamp_up > open_threshold * self.table['stdev'] + self.table['mu'] and spread_stamp_up < self.table["mu"] + self.table["stdev"] * stop_loss_threshold:

                    self.position = -1

                    print(
                        f"上開倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                    #self.logger.fill_simulator(f"up open threshold : Ref Size : {self.table['w1'] } Ref Price :{ref_mid_price} Target Size : {self.table['w2'] } Target Price :{target_mid_price}")

                    if self.table['w1'] < 0 and self.table['w2'] > 0:
                        self.open_timestamp = t
                        self.open_Quotes_setting(ref_ask, target_bid, t)
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'topopen')
                        print(ref_ask, target_bid)
                        return self.spread_quotes
                    elif self.table['w1'] > 0 and self.table['w2'] < 0:
                        self.open_timestamp = t
                        self.open_Quotes_setting(ref_bid, target_ask, t)
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'topopen')
                        print(ref_bid, target_ask)
                        return self.spread_quotes
                    elif self.table['w1'] < 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_ask, target_ask, t)

                        print(ref_ask, target_ask)
                        return self.spread_quotes
                    elif self.table['w1'] > 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_bid, target_bid, t)

                        print(ref_bid, target_bid)
                        return self.spread_quotes

                elif spread_stamp_down < self.table['mu'] - open_threshold * self.table['stdev'] and spread_stamp_down > self.table["mu"] - self.table["stdev"] * stop_loss_threshold:
                    self.position = 1

                    print(
                        f"下開倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                    #self.logger.fill_simulator(f"down open threshold : Ref Size : {self.table['w1'] } Ref Price :{ref_mid_price} Target Size : {self.table['w2'] } Target Price :{target_mid_price}")

                    print(f"Ref bid:{ref_bid} ; Target_ask : {target_ask}")
                    if self.table["w1"] < 0 and self.table['w2'] > 0:
                        self.open_timestamp = t
                        self.open_Quotes_setting(ref_bid, target_ask, t)
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'downopen')
                        print(ref_bid, target_ask)
                        return self.spread_quotes
                    elif self.table["w1"] > 0 and self.table['w2'] < 0:
                        self.open_timestamp = t
                        self.open_Quotes_setting(ref_ask, target_bid, t)
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'downopen')
                        print(ref_ask, target_bid)
                        return self.spread_quotes
                    elif self.table["w1"] < 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_bid, target_bid, t)

                        print(ref_bid, target_bid)
                        return self.spread_quotes
                    elif self.table["w1"] > 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_ask, target_ask, t)

                        print(ref_ask, target_ask)
                        return self.spread_quotes
            elif self.position != 0:
                if self.position == -1:
                    if self.ref_size < 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)

                    elif self.ref_size > 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)

                    elif self.ref_size > 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_ask)

                    elif self.ref_size < 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_bid)
                    if spread_stamp < self.table['mu']:
                        print(
                            f"上開倉正常平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        #self.logger.fill_simulator(f"up normal close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        self.cointegration_check = False
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t)

                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t)

                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                    elif spread_stamp > self.table["mu"] + self.table["stdev"] * stop_loss_threshold:
                        self.cointegration_check = False
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        print(
                            f"上開倉停損平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        #self.logger.fill_simulator(f"up stop-loss close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t)

                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t)

                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                    elif t - timedelta(days= self.config.hold_day) >= self.open_timestamp :
                        self.cointegration_check = False
                        self.open_timestamp = 0
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        print(
                            f"上開倉強制平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        #self.logger.fill_simulator(f"up stop-loss close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t)

                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t)

                            print(ref_bid, target_bid)
                            return self.spread_quotes

                elif self.position == 1:
                    if self.ref_size < 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)

                    elif self.ref_size > 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)

                    elif self.ref_size > 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_bid)

                    elif self.ref_size < 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_ask)

                    if spread_stamp > self.table['mu']:
                        self.cointegration_check = False
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        print(
                            f"下開倉正常平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        #self.logger.fill_simulator(f"down normal close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t)
                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t)
                            print(ref_ask, target_ask)
                            return self.spread_quotes
                    elif spread_stamp < self.table["mu"] - self.table["stdev"] * stop_loss_threshold:
                        self.cointegration_check = False
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        print(
                            f"下開倉停損平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")                                                
                        #self.logger.fill_simulator(f"down stop-loss close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")

                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t)
                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t)
                            print(ref_ask, target_ask)
                            return self.spread_quotes
                    elif t - timedelta(days= self.config.hold_day) >= self.open_timestamp :
                        self.cointegration_check = False
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        self.open_timestamp = 0
                        print(
                            f"下開倉強迫平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")                                                
                        #self.logger.fill_simulator(f"down stop-loss close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")

                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t)
                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t)
                            print(ref_ask, target_ask)
                            return self.spread_quotes


if __name__ == "__main__":
    p = round_price(0.0343,Decimal('0'))    
    print(p)