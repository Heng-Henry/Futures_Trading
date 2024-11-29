from asyncio.log import logger
import collections
import time
import pandas as pd
from scipy.ndimage.interpolation import shift
from datetime import timedelta, datetime
import os
from decimal import Decimal
import decimal
import math
import sys
from module.Johanson_class import Johansen
import matplotlib.pyplot as plt
import numpy as np
import json
drawpic = True
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
        #print(self.xs)
        if self.index == self.window_size - 1:
            self.is_warmed_up = True
        self.index += 1


class Predictor:
    c = 0 
    five_min_timestamp_1 = 0
    five_min_timestamp_2 = 0
    sec_timestamp_1 = 0
    sec_timestamp_2 = 0
    drawpic = True
    ADD_POS = 0.3
    level = 10
    def __init__(self, window_size, ref_symbol, target_symbol, slippage,log,config):
        self.window_size = window_size
        self.ref_symbol = ref_symbol
        self.target_symbol = target_symbol
        self.ref_spreads = Spreads(self.window_size)
        self.target_spreads = Spreads(self.window_size)
        self.rf_draw = Spreads(720)
        self.tg_draw = Spreads(720)
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
            "capital": 10000,
        }
        self.ref_size = 0
        self.target_size = 0
        self.cointegration_check = False
        self.timestamp_check = False
        self.count = 0
        self.config = config
        self.open_pos = [0,0]
        self.check = False
        self.increase = 0 
        self.CrossTime = 0
        self.Check_pos = [0,0,0,0,0,0,0,0,0,0,0] #1.5 1.8 2.1 2.4 3  
        self.pos_number = 0
        self.stop_loss = False
    def _reset(self):
        self.position = 0
        self.ref_spreads = Spreads(self.window_size)
        self.target_spreads = Spreads(self.window_size)
        self.rf_draw = Spreads(720)
        self.tg_draw = Spreads(720)
        self.table = {
            "w1": 0,
            "w2": 0,
            "mu": 0,
            "stdev": 0,
            "model": 0,
            "capital": 10000,
        }
        self.ref_size = 0
        self.target_size = 0
        self.cointegration_check = False
        self.check = False
        self.increase = 0 
        self.UPPER_OPEN_THRESHOLD = 0
        self.LOWER_OPEN_THRESHOLD = 0
        self.UPPER_CLOSE_THRESHOLD = 0
        self.LOWER_CLOSE_THRESHOLD = 0
        self.Check_pos = [0,0,0,0,0,0,0,0,0,0,0] #1.5 1.8 2.1 2.4 3 
        self.pos_number = 0 
        self.stop_loss = False
        self.count = 0
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
            
            ref_ask, target_ask = self.get_asks(orderbook)
            ref_bid, target_bid = self.get_bids(orderbook)
            ref_mid_price = (ref_ask + ref_bid) / 2  # ref mid price
            target_mid_price = (target_ask + target_bid) / \
                2  # target mid price
            #print(self.ref_timestamp - self.target_timestamp)
            # print(orderbook[self.ref_symbol]['date'],self.ref_timestamp,orderbook[self.target_symbol]['date'],self.target_timestamp)
            # print(f"ref :{ref_mid_price} , target : {target_mid_price}")
            if ref_ask and target_ask and ref_bid and target_bid:
                self.ref_spreads.update(ref_mid_price)
                self.target_spreads.update(target_mid_price)
                self.rf_draw.update(ref_mid_price)
                self.tg_draw.update(target_mid_price)


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
        Johanson_cointegration = Johansen(price_series)
        return Johanson_cointegration.execute()

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

    def open_Quotes_setting(self, ref_trade_price, target_trade_price,timestamp,date_time):
        slippage = self.slippage
        self.ref_size += 1
        self.target_size += 1
        print(self.ref_size,self.target_size)
        self.spread_quotes.set_price(
            self.ref_symbol, ref_trade_price * (1 + self.slippage_number(slippage, self.ref_size)))
        self.spread_quotes.set_price(
            self.target_symbol, target_trade_price * (1 + self.slippage_number(slippage, self.target_size)))
        self.spread_quotes.set_size(
            self.ref_symbol, abs(self.ref_size))
        self.spread_quotes.set_size(
            self.target_symbol, abs(self.target_size))
        self.spread_quotes.set_side(
            self.ref_symbol,  self.side_determination(self.table['w1'])
        )
        self.spread_quotes.set_side(
            self.target_symbol, self.side_determination(self.table['w2'])
        )
        print(f'reference weight : {self.table["w1"]} , target weight : {self.table["w2"]}')
        print(f'reference_price = {ref_trade_price * (1 + self.slippage_number(slippage,self.ref_size))} . size = {abs(self.ref_size)} , side = {self.side_determination(self.ref_size)}')
        print(f'target_price = {target_trade_price *(1 + self.slippage_number(slippage,self.target_size))} . size = {abs(self.target_size)} , side = {self.side_determination(self.target_size)}')
        # self.logger.fills('America',str(date_time), str(timestamp), self.ref_symbol,
        #                    'LIMIT', self.side_determination(self.ref_size), round_price(ref_trade_price,self.config.REF_PRICE_PRECISION)/200, trunc_amount(abs(self.ref_size),self.config.REF_AMOUNT_PRECISION))
        # self.logger.fills('America', str(date_time), str(timestamp), self.target_symbol,
        #                    'LIMIT', self.side_determination(self.target_size), round_price(target_trade_price,self.config.TARGET_PRICE_PRECISION)/(29*40), trunc_amount(abs(self.target_size),self.config.TARGET_AMOUNT_PRECISION))
        self.logger.fills('America',str(date_time), str(timestamp), self.ref_symbol,
                           'LIMIT', self.side_determination(self.table['w1']), round_price(ref_trade_price,self.config.REF_PRICE_PRECISION), trunc_amount(abs(1),self.config.REF_AMOUNT_PRECISION))
        self.logger.fills('America', str(date_time), str(timestamp), self.target_symbol,
                           'LIMIT', self.side_determination(self.table['w2']), round_price(target_trade_price,self.config.TARGET_PRICE_PRECISION), trunc_amount(abs(1),self.config.TARGET_AMOUNT_PRECISION))
        self.increase += self.ADD_POS
    def close_Quotes_setting(self, ref_trade_price, target_trade_price,timestamp,date_time):
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
        # self.logger.fills('America',str(date_time), str(timestamp), self.ref_symbol,
        #                    'LIMIT', CLOSE_POSITION[self.side_determination(self.ref_size)], round_price(ref_trade_price,self.config.REF_PRICE_PRECISION)/200, trunc_amount(abs(self.ref_size),self.config.REF_AMOUNT_PRECISION))
        # self.logger.fills('America',str(date_time), str(timestamp), self.target_symbol,
        #                    'LIMIT', CLOSE_POSITION[self.side_determination(self.target_size)], round_price(target_trade_price,self.config.TARGET_PRICE_PRECISION)/(29*40), trunc_amount(abs(self.target_size),self.config.TARGET_AMOUNT_PRECISION))
        if self.stop_loss:
            self.logger.fills('America',str(date_time), str(timestamp), self.ref_symbol,
                            'LIMIT', CLOSE_POSITION[self.side_determination(self.table['w1'])], round_price(ref_trade_price,self.config.REF_PRICE_PRECISION), trunc_amount(abs(self.ref_size),self.config.REF_AMOUNT_PRECISION))
            self.logger.fills('America',str(date_time), str(timestamp), self.target_symbol,
                            'LIMIT', CLOSE_POSITION[self.side_determination(self.table['w2'])], round_price(target_trade_price,self.config.TARGET_PRICE_PRECISION), trunc_amount(abs(self.target_size),self.config.TARGET_AMOUNT_PRECISION))
            self.position = 888
            self.stop_loss = False
            self.pos_number = 0
            self.ref_size = 0
            self.target_size = 0
            self.increase = 0
        else :
            self.logger.fills('America',str(date_time), str(timestamp), self.ref_symbol,
                            'LIMIT', CLOSE_POSITION[self.side_determination(self.table['w1'])], round_price(ref_trade_price,self.config.REF_PRICE_PRECISION), trunc_amount(abs(1),self.config.REF_AMOUNT_PRECISION))
            self.logger.fills('America',str(date_time), str(timestamp), self.target_symbol,
                            'LIMIT', CLOSE_POSITION[self.side_determination(self.table['w2'])], round_price(target_trade_price,self.config.TARGET_PRICE_PRECISION), trunc_amount(abs(1),self.config.TARGET_AMOUNT_PRECISION))
            self.ref_size -= 1
            self.target_size -= 1
            self.increase -= self.ADD_POS
            if self.ref_size == 0 and self.target_size == 0:
                self.position = 888
                self.stop_loss = False
                self.pos_number = 0
    def draw_pictrue(self,open_threshold,stop_loss_threshold,stamp,POS,trade_time):
        if self.drawpic :
            print("count :",self.count)
            path_to_image = "./trading_position_pic/"
            path = f'{path_to_image}{self.ref_symbol}_{self.target_symbol}_PIC/' 
            isExist = os.path.exists(path)
            #trade_time = trade_time.date()
            if not isExist:    
                # Create a new directory because it does not exist 
                os.makedirs(path)
                print("The new directory is created!")
            curDT = datetime.now()
            time = curDT.strftime("%Y%m%d%H%M")
            sp =  self.table['w1'] * np.log(self.rf_draw.xs[1:int(self.count /2)+1]) + self.table['w2'] * np.log(self.tg_draw.xs[1:int(self.count /2)+1])
            fig, ax1 = plt.subplots(figsize=(20, 10))
            ax1.plot(sp, color='tab:blue', alpha=0.75)
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.hlines(open_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10,'b')
            #ax1.hlines(stop_loss_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10, 'b') 
            ax1.hlines(self.table['mu'] - open_threshold * self.table['stdev'], 0, len(sp) + 10,'b')
            #ax1.hlines(self.table['mu'] - stop_loss_threshold * self.table['stdev'], 0, len(sp) + 10, 'b') 
            ax1.hlines(self.table['mu'], 0, len(sp) + 10, 'black') 
            ax1.scatter(len(sp)-1 ,stamp, color='g', edgecolors='r', marker='o')
            #ax1.text(3,-3,f"w1 = {self.table['w1']}\nw2 = {self.table['w2']}\nstd = {self.table['stdev']}\nmu = {self.table['mu']}")
            #ax1.text(0,0,f"ref : {ref} , bid : {bid}")
            ax1.text(0,self.table['mu'],f"w1 : {self.table['w1']} , w2 : {self.table['w2']}, stdev : {self.table['stdev']}")
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()
            ax2.plot(self.rf_draw.xs[1:int(self.count /2)+1]/ 200,color='tab:orange',alpha=0.75) #np.diff(a) / a[:-1]
            ax3.plot(self.tg_draw.xs[1:int(self.count /2)+1]/ (2*40*30),color='black', alpha=0.75)
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax3.tick_params(axis='y', labelcolor='black')
            if POS == 'topopen' or POS == 'downopen':
                plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +POS+'spread_'+str(trade_time)+'.png')
            elif POS == 'close':
                plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +POS+'spread_'+ str(trade_time)+'.png')
            elif POS == 'stop_loss':
                plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +POS+'spread_'+ str(trade_time)+'.png')
            
        plt.close()
    def checkCrossTimeFunc(self):
        sp =  self.table['w1'] * np.log(self.ref_spreads.xs) + self.table['w2'] * np.log(self.target_spreads.xs)
        CrossTime = 0
        for ti in  range(1,self.config.MA_WINDOW_SIZE) :
            if (sp[ti-1] < self.table['mu'] < sp[ti]) or (sp[ti-1] > self.table['mu'] > sp[ti]) :
                CrossTime += 1
        return CrossTime
    def save_and_accumulate_array(self,new_array):
        filename = 'data.json'
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                existing_data = json.load(file)
            accumulated_array = [a + b for a, b in zip(existing_data['array'], new_array)]
            existing_data['array'] = accumulated_array
        else:
            existing_data = {'array': new_array}

        with open(filename, 'w') as file:
            json.dump(existing_data, file)
    def forced_close_position(self, orderbook):
        t = orderbook[self.ref_symbol]['timestamp']
        d = orderbook[self.ref_symbol]['date']
        ref_ask, target_ask = self.get_asks(orderbook)
        ref_bid, target_bid = self.get_bids(orderbook)
        self.cointegration_check = False
        self.stop_loss = True
        if self.position == 1 :
            print(f"下開倉強制平倉")
            if self.table['w1'] < 0 and self.table['w2'] > 0:
                self.close_Quotes_setting(ref_ask, target_bid,t,d)
                #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp,'close')
                return self.spread_quotes
            elif self.table['w1'] > 0 and self.table['w2'] < 0:
                self.close_Quotes_setting(ref_bid, target_ask,t,d)
                #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp,'close')
                #self.logger.info(ref_bid, target_ask)
                return self.spread_quotes
            elif self.ref_size > 0 and self.target_size > 0:
                self.close_Quotes_setting(ref_bid, target_bid)
                print(ref_bid, target_bid)
                return self.spread_quotes
            elif self.ref_size < 0 and self.target_size < 0:
                self.close_Quotes_setting(ref_ask, target_ask)
                print(ref_ask, target_ask)
                return self.spread_quotes
        elif self.position == -1 :
            #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
            print(f"上開倉強制平倉")
            if self.table['w1'] < 0 and self.table['w2'] > 0:
                self.close_Quotes_setting(ref_bid, target_ask,t,d)
                #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp,'close')
                return self.spread_quotes
            elif self.table['w1'] > 0 and self.table['w2'] < 0:
                self.close_Quotes_setting(ref_ask, target_bid,t,d)
                #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp,'close')
                #self.logger.info(ref_ask, target_bid)
                return self.spread_quotes
            elif self.ref_size > 0 and self.target_size > 0:
                self.close_Quotes_setting(ref_ask, target_ask)

                print(ref_ask, target_ask)
                return self.spread_quotes
            elif self.ref_size < 0 and self.target_size < 0:
                self.close_Quotes_setting(ref_bid, target_bid)

                print(ref_bid, target_bid)
                return self.spread_quotes

    def get_target_spread_price(self, orderbook, open_threshold, stop_loss_threshold):

        
        if self.ref_spreads.is_warmed_up and self.target_spreads.is_warmed_up and orderbook[self.ref_symbol]['timestamp']  == orderbook[self.target_symbol]['timestamp'] \
            and self.sec_timestamp_1 != orderbook[self.ref_symbol]['timestamp'] and self.sec_timestamp_2 != orderbook[self.target_symbol]['timestamp']:
            
            t = orderbook[self.ref_symbol]['timestamp']
            d = orderbook[self.ref_symbol]['date']
            ref_ask, target_ask = self.get_asks(orderbook)
            ref_bid, target_bid = self.get_bids(orderbook)
            ref_mid_price = (ref_ask + ref_bid) / 2  # ref mid price
            target_mid_price = (target_ask + target_bid) / 2

            self.sec_timestamp_1 = orderbook[self.ref_symbol]['timestamp']
            self.sec_timestamp_2 = orderbook[self.target_symbol]['timestamp']
            # ref_ask, target_ask = self.get_level_asks(orderbook)
            # ref_bid, target_bid = self.get_level_bids(orderbook) 

            if self.position == 0 and self.cointegration_check is False and self.check is False:
                print("in test cointegration")
                mu, stdev, model, w1, w2 = self.cointegration_test()
                if model > 0 and model < 4 and w1 * w2 < 0 :
                    self.cointegration_check = True
                    self.table = {
                        "w1": float(w1),
                        "w2": float(w2),
                        "mu": float(mu),
                        "stdev": float(stdev),
                        "model": model,
                        "capital": 10000
                    }
                    check_cross_time = self.checkCrossTimeFunc()
                    if check_cross_time < 5 :
                        self.cointegration_check = False
                self.check = True
            if self.cointegration_check :
                if self.position == 0 :
                    self.UPPER_OPEN_THRESHOLD = open_threshold* self.table['stdev'] + self.table['mu']
                    self.LOWER_OPEN_THRESHOLD = self.table['mu'] - open_threshold  * self.table['stdev'] 
                    self.UPPER_CLOSE_THRESHOLD = self.table['mu']
                    self.LOWER_CLOSE_THRESHOLD = self.table['mu']
                else :
                    self.UPPER_OPEN_THRESHOLD = (open_threshold + self.increase) * self.table['stdev'] + self.table['mu'] 
                    self.UPPER_CLOSE_THRESHOLD = (self.increase- self.ADD_POS) * self.table['stdev'] + self.table['mu']
                    self.LOWER_OPEN_THRESHOLD = self.table['mu'] - (open_threshold + self.increase) * self.table['stdev']
                    self.LOWER_CLOSE_THRESHOLD = self.table['mu'] - (self.increase-self.ADD_POS) * self.table['stdev']
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
                    self.Check_pos[self.pos_number] += 1
                    self.pos_number += 1
                    self.position = -1
                    print(
                        f"上開倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                    #self.logger.fill_simulator(f"up open threshold : Ref Size : {self.table['w1'] } Ref Price :{ref_mid_price} Target Size : {self.table['w2'] } Target Price :{target_mid_price}")

                    if self.table['w1'] < 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_ask, target_bid, t,d)
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'topopen',d)
                        print(ref_ask, target_bid)
                        return self.spread_quotes
                    elif self.table['w1'] > 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_bid, target_ask, t,d)
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'topopen',d)
                        print(ref_bid, target_ask)
                        return self.spread_quotes
                    elif self.table['w1'] < 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_ask, target_ask, t,d)

                        print(ref_ask, target_ask)
                        return self.spread_quotes
                    elif self.table['w1'] > 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_bid, target_bid, t,d)

                        print(ref_bid, target_bid)
                        return self.spread_quotes

                elif spread_stamp_down < self.table['mu'] - open_threshold * self.table['stdev'] and spread_stamp_down > self.table["mu"] - self.table["stdev"] * stop_loss_threshold:
                    self.position = 1
                    self.Check_pos[self.pos_number] += 1
                    self.pos_number += 1
                    print(
                        f"下開倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                    #self.logger.fill_simulator(f"down open threshold : Ref Size : {self.table['w1'] } Ref Price :{ref_mid_price} Target Size : {self.table['w2'] } Target Price :{target_mid_price}")

                    print(f"Ref bid:{ref_bid} ; Target_ask : {target_ask}")
                    if self.table["w1"] < 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_bid, target_ask, t,d)
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'downopen',d)
                        print(ref_bid, target_ask)
                        return self.spread_quotes
                    elif self.table["w1"] > 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_ask, target_bid, t,d)
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'downopen',d)
                        print(ref_ask, target_bid)
                        return self.spread_quotes
                    elif self.table["w1"] < 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_bid, target_bid, t,d)

                        print(ref_bid, target_bid)
                        return self.spread_quotes
                    elif self.table["w1"] > 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_ask, target_ask, t,d)

                        print(ref_ask, target_ask)
                        return self.spread_quotes
            elif self.position != 0:
                if self.position == -1:
                    if self.table['w1'] < 0 and self.table['w2'] > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)
                        spread_stamp_up = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)
                        spread_stamp_down = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)

                    elif self.table['w1'] > 0 and self.table['w2'] < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)
                        spread_stamp_up = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)
                        spread_stamp_down = self.table["w1"] * \
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
                    
                    if spread_stamp <= self.UPPER_CLOSE_THRESHOLD:
                        print(
                            f"上開倉正常平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        #self.logger.fill_simulator(f"up normal close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        #self.cointegration_check = False
                        #self.save_and_accumulate_array(self.Check_pos)
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close',d)
                        if self.table['w1'] < 0 and self.table['w2']  > 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t,d)

                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.table['w1'] > 0 and self.table['w2'] < 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t,d)

                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t,d)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t,d)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                    elif spread_stamp >= self.table["mu"] + self.table["stdev"] * stop_loss_threshold:
                        self.cointegration_check = False
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'stop_loss',d)
                        print(
                            f"上開倉停損平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        #self.logger.fill_simulator(f"up stop-loss close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        self.stop_loss = True
                        if self.table['w1'] < 0 and self.table['w2'] > 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t,d)

                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.table['w1'] > 0 and self.table['w2'] < 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t,d)

                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t,d)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t,d)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                    elif spread_stamp_up > (open_threshold + self.increase) * self.table['stdev'] + self.table['mu'] and spread_stamp_up < self.table["mu"] + self.table["stdev"] * stop_loss_threshold and self.pos_number < self.level :
                        print(
                            f"上開倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        #self.logger.fill_simulator(f"up open threshold : Ref Size : {self.table['w1'] } Ref Price :{ref_mid_price} Target Size : {self.table['w2'] } Target Price :{target_mid_price}")
                        self.Check_pos[self.pos_number] += 1
                        self.pos_number += 1
                        if self.table['w1'] < 0 and self.table['w2'] > 0:
                            self.open_Quotes_setting(ref_ask, target_bid, t,d)
                            self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'topopen',d)
                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.table['w1'] > 0 and self.table['w2'] < 0:
                            self.open_Quotes_setting(ref_bid, target_ask, t,d)
                            self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'topopen',d)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.table['w1'] < 0 and self.table['w2'] < 0:
                            self.open_Quotes_setting(ref_ask, target_ask, t,d)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.table['w1'] > 0 and self.table['w2'] > 0:
                            self.open_Quotes_setting(ref_bid, target_bid, t,d)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                    # elif t.strftime("%H:%M:%S") == '20:59:00':
                    #     self.cointegration_check = False
                    #     #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                    #     print(
                    #         f"上開倉強制平倉")
                    #     if self.ref_size < 0 and self.target_size > 0:
                    #         self.close_Quotes_setting(ref_bid, target_ask,t,d)
                    #         #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp,'close')
                    #         return self.spread_quotes
                    #     elif self.ref_size > 0 and self.target_size < 0:
                    #         self.close_Quotes_setting(ref_ask, target_bid,t,d)
                    #         #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp,'close')
                    #         #self.logger.info(ref_ask, target_bid)
                    #         return self.spread_quotes
                    #     elif self.ref_size > 0 and self.target_size > 0:
                    #         self.close_Quotes_setting(ref_ask, target_ask)

                    #         print(ref_ask, target_ask)
                    #         return self.spread_quotes
                    #     elif self.ref_size < 0 and self.target_size < 0:
                    #         self.close_Quotes_setting(ref_bid, target_bid)

                    #         print(ref_bid, target_bid)
                    #         return self.spread_quotes

                elif self.position == 1:
                    if self.table['w1'] < 0 and self.table['w2']  > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)
                        spread_stamp_up = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)
                        spread_stamp_down = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)

                    elif self.table['w1']  > 0 and self.table['w2']  < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)
                        spread_stamp_up = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)
                        spread_stamp_down = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)

                    elif self.ref_size > 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_bid)

                    elif self.ref_size < 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_ask)
                    
                        print(t,d)
                        print(spread_stamp)
                    if spread_stamp >=  self.LOWER_CLOSE_THRESHOLD: 
                        #self.save_and_accumulate_array(self.Check_pos)
                        #self.cointegration_check = False
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close',d)
                        print(
                            f"下開倉正常平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        #self.logger.fill_simulator(f"down normal close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        if self.table['w1'] < 0 and self.table['w2'] > 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t,d)
                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.table['w1'] > 0 and self.table['w2'] < 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t,d)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t,d)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t,d)
                            print(ref_ask, target_ask)
                            return self.spread_quotes
                    elif spread_stamp < self.table["mu"] - self.table["stdev"] * stop_loss_threshold:
                        self.cointegration_check = False
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'stop_loss',d)
                        print(
                            f"下開倉停損平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")                                                
                        #self.logger.fill_simulator(f"down stop-loss close threshold : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        self.stop_loss = True
                        if self.table['w1'] < 0 and self.table['w2'] > 0:
                            self.close_Quotes_setting(ref_ask, target_bid, t,d)
                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.table['w1'] > 0 and self.table['w2'] < 0:
                            self.close_Quotes_setting(ref_bid, target_ask, t,d)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid, t,d)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask, t,d)
                            print(ref_ask, target_ask)
                            return self.spread_quotes
                    elif spread_stamp_down < self.table['mu'] - (open_threshold + self.increase) * self.table['stdev'] and spread_stamp_down > self.table["mu"] - self.table["stdev"] * stop_loss_threshold and self.pos_number < self.level:
                        self.Check_pos[self.pos_number] += 1
                        self.pos_number += 1
                        print(
                            f"下開倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        print(f"Ref bid:{ref_bid} ; Target_ask : {target_ask}")
                        if self.table["w1"] < 0 and self.table['w2'] > 0:
                            self.open_Quotes_setting(ref_bid, target_ask, t,d)
                            self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'downopen',d)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.table["w1"] > 0 and self.table['w2'] < 0:
                            self.open_Quotes_setting(ref_ask, target_bid, t,d)
                            self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'downopen',d)
                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.table["w1"] < 0 and self.table['w2'] < 0:
                            self.open_Quotes_setting(ref_bid, target_bid, t,d)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.table["w1"] > 0 and self.table['w2'] > 0:
                            self.open_Quotes_setting(ref_ask, target_ask, t,d)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
        if self.position != 888 : 
            self.count += 1


if __name__ == "__main__":
    p = round_price(0.0343,Decimal('0'))    
    print(p)