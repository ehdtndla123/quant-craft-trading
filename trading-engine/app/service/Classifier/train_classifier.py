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

def get_device():
    import torch
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
    ans = torch.tensor([LONG if data.Close[state_to + 1] - data.Close[state_to] > 0 else SHORT])
    for _ in range(batch_size - 1):
        state_to = random.randrange(state_size, len(data) - 1)
        state_from = state_to - state_size
        state_tmp = data[state_from:state_to]
        state = torch.cat((state, torch.from_numpy(np.array([state_tmp.Open.values, state_tmp.Close.values, state_tmp.High.values,
                            state_tmp.Low.values, state_tmp.Volume.values])).unsqueeze(1)), dim=1)
        ans = torch.cat((ans, torch.from_numpy(np.array([LONG if data.Close[state_to + 1] - data.Close[state_to] > 0 else SHORT]))), dim=0)

    return state, ans

device = get_device()

state_size = 300
action_size = 1
hidden_size = 2 ** 9
learning_rate = 0.001
epoch = 1
iter_in_epoch = 1

LONG = 1
SHORT = -1

exchange_name="binance"
symbol="BTC/USDT"
timeframe="1d"
start_time="2019-01-01"
# start_time="2024-07-27"
end_time="2024-07-30"
timezone="Asia/Seoul"

file_name = f"{exchange_name}_{symbol.replace('/', '_')}_{timeframe}_{start_time}_{end_time}.csv"
file_path = os.path.join('data', file_name)

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    DataLoader.save_data_from_ccxt(
        exchange_name=exchange_name,
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        timezone=timezone
    )
finally:
    data = pd.read_csv(file_path)

train_data = data[0:int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]
classes = ('LONG', 'SHORT')

batch_size = 2 ** 9
model = LSTM_NN(state_size=state_size, action_size=action_size, hidden_size=hidden_size).to(device=device)
criterion = torch.nn.CrossEntropyLoss()
# optimiser = optim.AdamW(model.parameters(), lr=learning_rate)
optimiser = CoRe(model.parameters(), lr=learning_rate)

# Training
for e in range(epoch):
    running_loss = 0.0
    for _ in range(iter_in_epoch):
        optimiser.zero_grad()
        state, ans = batch(train_data, state_size, batch_size)

        if device == torch.device('mps'):
            state = state.to(device=device, dtype=torch.float32)
            ans = ans.to(device=device, dtype=torch.float32)
        else:
            state = state.to(device=device)
            ans = ans.to(device=device)

        outputs = model(state)
        outputs =torch.sign(outputs)
        outputs[outputs == 0] = 1

        loss = criterion(outputs, ans)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()

    print(f'epoch: {e+1}, loss: {running_loss / iter_in_epoch}')

i = 0
machine_dir = os.path.join(current_dir, '../../../', 'models/Classifier/', str(socket.gethostname()))
model_dir = os.path.join(machine_dir, f"classifyer_{i}")

while(os.path.exists(model_dir)):
    i += 1
    model_dir = os.path.join(machine_dir, f"classifyer_{i}")
os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'classifier_net.pth')

torch.save(model.state_dict(), model_path)

# Testing
running_loss = 0.0
correct = []
for i in range(len(test_data) - state_size -1):
    state_tmp = test_data[i : state_size + i]
    state = torch.from_numpy(np.array([state_tmp.Open.values, state_tmp.Close.values, state_tmp.High.values,
                        state_tmp.Low.values, state_tmp.Volume.values])).unsqueeze(1)
    ans = torch.tensor([LONG if data.Close[state_size + 1] - data.Close[state_size] > 0 else SHORT])
    if device == torch.device('mps'):
        state = state.to(device=device, dtype=torch.float32)
        ans = ans.to(device=device, dtype=torch.float32)
    else:
        state = state.to(device=device)
        ans = ans.to(device=device)
    output = model(state)
    output = 1 if output >= 1 else -1
    correct.append(1 if output == ans else 0)
print(f'Test Accuricy = {sum(correct) / len(correct)}')
