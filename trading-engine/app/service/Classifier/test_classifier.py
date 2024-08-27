import os
import sys
import pandas as pd
import random
import numpy as np
import socket

import torch
import torch.nn.functional as F
import torch.optim as optim
from core_optimizer import CoRe

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from app.services.data_loader_service import DataLoaderService as DataLoader
from app.service.Classifier.NN.LSTM_NN import LSTM_NN
LONG = 1
SHORT = 0

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS is available. Using GPU on macOS.")
    else:
        device = torch.device('cpu')
        print("No GPU found. Using CPU.")

    return device

def batch(data, state_size, batch_size):
    state_to = random.randrange(state_size, len(data) - 1)
    state_from = state_to - state_size
    state_tmp = data[state_from:state_to]

    state = torch.from_numpy(np.array([state_tmp.Open.values, state_tmp.Close.values, state_tmp.High.values,
                        state_tmp.Low.values, state_tmp.Volume.values])).unsqueeze(1)
    ans = torch.zeros(2).unsqueeze(0)
    ans[-1][LONG if data.Close[state_to + 1] - data.Close[state_to] > 0 else SHORT] += 1
    for _ in range(batch_size - 1):
        state_to = random.randrange(state_size, len(data) - 1)
        state_from = state_to - state_size
        state_tmp = data[state_from:state_to]
        state = torch.cat((state, torch.from_numpy(np.array([state_tmp.Open.values, state_tmp.Close.values, state_tmp.High.values,
                            state_tmp.Low.values, state_tmp.Volume.values])).unsqueeze(1)), dim=1)
        ans_tmp = torch.zeros(2).unsqueeze(0)
        ans_tmp[-1][LONG if data.Close[state_to + 1] - data.Close[state_to] > 0 else SHORT] += 1
        ans = torch.cat((ans, ans_tmp), dim=0)

    return state, ans

def create_model_dir(base_dir):
    i = 0
    model_dir = os.path.join(base_dir, f"classifier_{i}")
    while os.path.exists(model_dir):
        i += 1
        model_dir = os.path.join(base_dir, f"classifier_{i}")
    os.makedirs(model_dir)
    return model_dir

device = get_device()

state_size = 300
action_size = 2
hidden_size = 2 ** 10
learning_rate = 0.001
epoch = 100
iter_in_epoch = 10000



exchange_name="binance"
symbol="BTC/USDT"
timeframe="1m"
start_time="2019-01-01"
# start_time="2024-07-27"
end_time="2024-07-30"
timezone="Asia/Seoul"

file_name = f"{exchange_name}_{symbol.replace('/', '_')}_{timeframe}_{start_time}_{end_time}.csv"
file_path = os.path.join('data', file_name)


if not os.path.exists(file_path):
    print('downloading new datafile')
    DataLoader.save_data_from_ccxt(
        exchange_name=exchange_name,
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        timezone=timezone
    )

test_data = pd.read_csv(file_path)
classes = ('SHORT', 'LONG')

batch_size = 2 ** 9
model = LSTM_NN(state_size=state_size, action_size=action_size, hidden_size=hidden_size).to(device=device)
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCEWithLogitsLoss()
# optimiser = optim.AdamW(model.parameters(), lr=learning_rate)
optimiser = CoRe(model.parameters(), lr=learning_rate)



i = 0
machine_dir = os.path.join(current_dir, '../../../', 'models/Classifier/', str(socket.gethostname()))
model_dir = create_model_dir(machine_dir)
model_path = os.path.join(machine_dir, 'classifier_0/classifier_net.pth')

model.load_state_dict(torch.load(model_path, weights_only=True))

# Testing
running_loss = 0.0
correct = []
for i in range(len(test_data) - state_size -1):
    state_tmp = test_data[i : state_size + i]
    state = torch.from_numpy(np.array([state_tmp.Open.values, state_tmp.Close.values, state_tmp.High.values,
                        state_tmp.Low.values, state_tmp.Volume.values])).unsqueeze(1).to(device=device, dtype=torch.float32)
    ans = torch.zeros(2)
    ans[LONG if test_data.Close[state_size + i + 1] - test_data.Close[state_size + i] > 0 else SHORT] += 1
    ans = ans.to(device=device, dtype=torch.float32)
    output = model(state).squeeze(1)
    output = torch.argmax(output)
    ans = torch.argmax(ans)
    correct.append(1 if output == ans else 0)
print(f'Test Accuricy = {sum(correct) / len(correct)}')
