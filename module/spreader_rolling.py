import logging
import sys
from module.pricer import Pricer
#from module.predictor_jc import Predictor
from datetime import timedelta, datetime
from module.log_format import SaveLog
from module.predictor_jc_rolling import Predictor

#from module.predictor_jc_addpos import Predictor

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Spreader:
    def __init__(self, config, Ref, Target,path,period):
        # Remove all handlers from the root logger to avoid duplicate log entries
        logging.getLogger('').handlers = []

        # Initialize instance variables
        self.config = config
        self.log = SaveLog(
            "Allen",
            "PairTrading",
            f"{Ref}{Target}_{int(self.config.TEST_SECOND/60)}min_{self.config.OPEN_THRESHOLD}_{self.config.STOP_LOSS_THRESHOLD}_{period}",
            path
        )
        self.predictor = Predictor(
            window_size=self.config.MA_WINDOW_SIZE,
            ref_symbol=config.REFERENCE_SYMBOL,
            target_symbol=config.TARGET_SYMBOL,
            slippage=self.config.SLIPPAGE,
            log=self.log,
            config=config
        )
        self.pricer = Pricer(
            config.REFERENCE_SYMBOL,
            config.TARGET_SYMBOL,
            self.log
        )
        self.orderbook = {}
        self.orderbook_5min = {}
        self.trades = {}
        self.spread_prices = None
        self.nowa_date = None

    def local_simulate(self, date_time, timestamp, symbol, asks, bids,time_end_obj):
    # Check if the order book needs to be updated
        #time_end_obj = datetime.strptime("13:45:00", "%H:%M:%S").time()
        if timestamp == time_end_obj:  # if the current date/time is different from the last update
            # If the predictor has an open position, force it to close
            if self.predictor.position == 1 or self.predictor.position == -1:
                self.predictor.forced_close_position(orderbook=self.orderbook)
            # Update the current date/time and reset the predictor
            self.predictor._reset()
            #由於日盤尾盤所以position歸0, cointegration test重算

        # Update the order book with the current quote
        self.orderbook[symbol] = {
            'buyQuote': [{'price': [bids - 100], 'size': [bids]}],
            'sellQuote': [{'price': [asks + 100], 'size': [asks]}],
            'timestamp': timestamp,
            'date': date_time
        }
        # Update the predictor with the current order book
        self.predictor.update_spreads(self.orderbook)
        
        # Get the target spread price based on the current order book and configuration thresholds
        self.spread_prices = self.predictor.get_target_spread_price(
            orderbook=self.orderbook,
            open_threshold=self.config.OPEN_THRESHOLD,
            stop_loss_threshold=self.config.STOP_LOSS_THRESHOLD,
        )

