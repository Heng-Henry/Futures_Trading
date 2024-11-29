from decimal import Decimal
def string_length(precision):
    D = '0.'
    for _ in range(precision):
        D += '0'
    return D
def remove_zeros(string):
    # 先將字符串轉換為浮點數
    num = float(string)
    # 再將浮點數轉換為字符串，並刪除右側的0
    result = str(num).rstrip('0')
    # 如果最後一個字符是點，也一併刪除
    if result[-1] == '.':
        result = result[:-1]
    return result
class Pair_Trading_Config:
    def __init__(
        self, Ref, Target, open_thershold, stop_loss_threshold, test_second, window_size
    ) -> None:

        # Reference symbol is where the MARKET orders are intiated AFTER target symbol's limit orders are filled.
        # REFERENCE_SYMBOL = "BTC-USD"
        self.REFERENCE_SYMBOL = Ref
        self.TARGET_SYMBOL = Target
        self.OPEN_THRESHOLD = float(open_thershold)
        self.STOP_LOSS_THRESHOLD = float(stop_loss_threshold)
        # Window size for calculating spread mean.
        self.MA_WINDOW_SIZE = int(window_size)
        self.SLIPPAGE = 0.000
        self.TEST_SECOND = float(test_second)
        self.REF_PRICE_PRECISION = Decimal('0.01')
        self.REF_AMOUNT_PRECISION = Decimal('0.01')
        self.TARGET_PRICE_PRECISION = Decimal('0.01')
        self.TARGET_AMOUNT_PRECISION = Decimal('0.01')
        