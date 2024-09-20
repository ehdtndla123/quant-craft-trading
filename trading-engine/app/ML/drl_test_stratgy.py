from .DRL.env.drl_backtesting import Strategy, _OutOfMoneyError

class DRLStrategyTest(Strategy):
    def init(self, agent, data_length):
        self.previous_value = 0
        self.liq_cnt = 0
    def next(self):
        # if self.previous_value == self.equity:
        #     if self.liq_cnt > 5:
        #         print('\nraising out of money error\n')
        #         raise _OutOfMoneyError
        #     self.liq_cnt += 1
        # else:
        #     self.liq_cnt = 0

        print(f'{self.data.datetime[-1]}: Open {self.data.Open[-1]}, Close {self.data.Close[-1]}, Balannce {self.equity}')

        action = input()

        if action == '1':
            self.buy()
        elif action == '2':
            self.sell()
        else:
            return
        self.previous_value = self.equity