MODEL_STORE_INTERVAL = 100
GRAPH_DRAW_INTERVAL = 10

N_TRAIN = 500

# Define Actions
LONG = 0
SHORT = 1
CLOSE = 2
HOLD = 3

# Done States
LIQUIFIED = 0
DATA_DONE = 1

# Reward Value
DEMOCRATISATION = -100000

def trade_done(env_self):
    env_self.is_data_done = True

def liquified(env_self):
    env_self.is_liquified = True
    env_self.finish_episode()