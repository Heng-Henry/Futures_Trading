import asyncio
import decimal
import re
from decimal import Decimal

# import logging
import sys
import random
import string

# import telegram
# logger = logging.getLogger(__name__)

PRECISION_AMOUNT = Decimal("0.0000")
PRECISION_PRICE = Decimal("0")

def round_price(x):
    """
    There's probably a faster way to do this...
    """
    return float(Decimal(x).quantize(PRECISION_PRICE))


def trunc_amount(x):
    """
    There's probably a faster way to do this...
    """
    with decimal.localcontext() as c:
        return float(Decimal(x).quantize(PRECISION_AMOUNT))


def side_to_price(side, x):
    neg = x * (-1)
    if side == "BUY":
        return x
    elif side == "SELL":
        return neg


class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


class Pricer:
    active_orders = {}
    open_close_mapping = TwoWayDict()

    def __init__(self,ref_symbol, target_symbol, logger):
        self.ref_symbol = ref_symbol
        self.target_symbol = target_symbol
        self.log = logger

    async def manage_trade(
        self,
        trades,
        spread_prices,
    ):
        # order_key_sell = "manage_SELL_{}".format(0)
        # order_key_buy = "manage_BUY_{}".format(0)
        order_tasks = []
        print("trade length : ", len(trades))
        for trade in trades:
            cl_order_id = trade["clOrderId"]
            size = float(trade["size"])
            symbol = trade["symbol"]
            # logger.info(
            #    "Symbol {} , Order {} filled {} @ {}".format(symbol, cl_order_id, size, trade['price']))
            self.log.fills(
                "BTSE",
                trade["orderId"],
                trade["symbol"],
                "LIMIT",
                trade["side"],
                trade["price"],
                trade["size"],
            )
            origin_size = spread_prices.get_size(symbol)
            origin_size = trunc_amount(origin_size)
            price = spread_prices.get_price(symbol)
            price = round_price(price)
            side = spread_prices.get_side(symbol)
            if origin_size > size:
                print("======= cancel_origin order =========")
                await self.api.cancel_order(symbol, trade["orderId"])
                print("======= sufficient the size =========")
                order_tasks.append(
                    self.api.submit_order(
                        symbol=symbol,
                        cl_order_id=cl_order_id,
                        side=side,
                        price=round_price(price * (1 + side_to_price(side, 0.005))),
                        size=trunc_amount(origin_size - size),
                    )
                )
            result = await asyncio.gather(*order_tasks)
            print("sufficient result :\n", result)

            """
            if symbol == self.ref_symbol :
                origin_size = spread_prices.get_size(self.ref_symbol)
                origin_size = trunc_amount(origin_size)
                price = spread_prices.get_price(self.ref_symbol)
                price = round_price(price)
                side = spread_prices.get_side(self.ref_symbol)
                
                print(origin_size , size)
                if origin_size > size :
                        print("======= cancel_origin order =========")
                        await self.api.cancel_order(symbol,trade['orderId'])

                        print("======= sufficient the size =========")
                        if side == "SELL" :
                            order_tasks.append(self.api.submit_order(
                            symbol= self.ref_symbol, cl_order_id=order_key_sell, side= side, price=price * (1 - 0.005), size= origin_size - size))
                        elif side == "BUY":
                            order_tasks.append(self.api.submit_order(
                            symbol= self.ref_symbol, cl_order_id=order_key_buy, side= side, price=price *(1 + 0.005), size= origin_size - size))
            elif symbol == self.target_symbol :
                origin_size = spread_prices.get_size(self.target_symbol, 0)
                price = spread_prices.get_price(self.target_symbol, 0)
                side = spread_prices.get_side(self.target_symbol , 0)
                if origin_size > size :
                        print("======= cancel_origin order =========")
                        await self.api.cancel_order(symbol,trade['orderId'])

                        print("======= sufficient the size =========")
                        if side == "SELL" :
                            order_tasks.append(self.api.submit_order(
                            symbol= self.target_symbol, cl_order_id=order_key_sell, side= side, price=price * (1-0.005), size= origin_size - size))
                        elif side == "BUY":
                            order_tasks.append(self.api.submit_order(
                            symbol= self.target_symbol, cl_order_id=order_key_buy, side= side, price=price * (1 + 0.005), size= origin_size - size))
            result = await asyncio.gather(*order_tasks)
            print("sufficient result :\n",result)
            """

    async def create_open_orders(self, spread_prices):
        print("===== create open orders =====")
        order_tasks = []

        order_key_sell = "open_SELL_{}".format(0)
        order_key_buy = "open_BUY_{}".format(0)
        price = spread_prices.get_price(self.ref_symbol)
        price = round_price(price)
        size = spread_prices.get_size(self.ref_symbol)
        size = trunc_amount(size)
        side = spread_prices.get_side(self.ref_symbol)
        if side == "BUY":
            order_tasks.append(
                self.api.submit_order(
                    symbol=self.ref_symbol,
                    cl_order_id=order_key_buy,
                    side=side,
                    price=price,
                    size=size,
                )
            )
        elif side == "SELL":
            order_tasks.append(
                self.api.submit_order(
                    symbol=self.ref_symbol,
                    cl_order_id=order_key_sell,
                    side=side,
                    price=price,
                    size=size,
                )
            )

        price = spread_prices.get_price(self.target_symbol)
        price = round_price(price)
        size = spread_prices.get_size(self.target_symbol)
        size = trunc_amount(size)
        side = spread_prices.get_side(self.target_symbol)
        if side == "BUY":
            order_tasks.append(
                self.api.submit_order(
                    symbol=self.target_symbol,
                    cl_order_id=order_key_buy,
                    side=side,
                    price=price,
                    size=size,
                )
            )
        elif side == "SELL":
            order_tasks.append(
                self.api.submit_order(
                    symbol=self.target_symbol,
                    cl_order_id=order_key_sell,
                    side=side,
                    price=price,
                    size=size,
                )
            )

        result = await asyncio.gather(*order_tasks)
        print("order result :\n", result)
        cancelled_task = []
        for r in result:
            if r:
                if r["status"] == 2:
                    cancelled_task.append(
                        self.api.cancel_order(r["symbol"], r["orderID"])
                    )
        cancel_result = await asyncio.gather(*cancelled_task)
        print("cancelled result :\n", cancel_result)
        new_order_task = []
        if cancel_result:
            for c in cancel_result:
                if c["fillSize"] == 0:
                    if c["side"] == "BUY":
                        new_order_task.append(
                            self.api.submit_order(
                                symbol=c["symbol"],
                                cl_order_id=c["clOrderID"],
                                side=c["side"],
                                price=round_price(c["price"] * (1 + 0.001)),
                                size=c["size"],
                            )
                        )
                    elif c["side"] == "SELL":
                        new_order_task.append(
                            self.api.submit_order(
                                symbol=c["symbol"],
                                cl_order_id=c["clOrderID"],
                                side=c["side"],
                                price=round_price(c["price"] * (1 - 0.001)),
                                size=c["size"],
                            )
                        )
        reorder_result = await asyncio.gather(*new_order_task)
        print("reorder result :\n", reorder_result)

    """
    async def amend_order_test(self, spread_prices,):
        print("===== amend open orders =====")
        order_tasks = []

        order_key_sell = "open_SELL_{}".format(0)
        order_key_buy = "open_BUY_{}".format(0)
        price = spread_prices.get_price(self.ref_symbol)
        price = round_price(price)
        size = spread_prices.get_size(self.ref_symbol)
        size = trunc_amount(size)
        side = spread_prices.get_side(self.ref_symbol)
        if side == 'BUY':
            order_tasks.append(self.api.submit_order(
                symbol=self.ref_symbol, cl_order_id=order_key_buy, side=side, price=price, size=size))
        elif side == 'SELL':
            order_tasks.append(self.api.submit_order(
                symbol=self.ref_symbol, cl_order_id=order_key_sell, side=side, price=price, size=size))

        price = spread_prices.get_price(self.target_symbol)
        price = round_price(price)
        size = spread_prices.get_size(self.target_symbol)
        size = trunc_amount(size)
        side = spread_prices.get_side(self.target_symbol)
        if side == 'BUY':
            order_tasks.append(self.api.submit_order(
                symbol=self.target_symbol, cl_order_id=order_key_buy, side=side, price=price, size=size))
        elif side == 'SELL':
            order_tasks.append(self.api.submit_order(
                symbol=self.target_symbol, cl_order_id=order_key_sell, side=side, price=price, size=size))

        result = await asyncio.gather(*order_tasks)
        print('order result :\n', result)
        amend_order_task = []
        for r in result:
            if r:
                if r['status'] == 2:
                    amend_order_task.append(self.api.amend_order(symbol=r['symbol'], type="PRICE", value=round(
                        r['price'] * (1 + side_to_price(side, 0.005))), order_id=r['orderID']))
        cancel_result = await asyncio.gather(*amend_order_task)
        print("cancelled result :\n", cancel_result)
    """
