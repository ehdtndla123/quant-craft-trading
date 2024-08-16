import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_NN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=state_size, hidden_size=hidden_size, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        self.lstm3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2 ** 2)
        self.fc2 = nn.Linear(hidden_size // 2 ** 2, hidden_size // 2 ** 3)
        self.fc3 = nn.Linear(hidden_size // 2 ** 3, action_size)

        self.silu = nn.SiLU()

    def forward(self, state):
        cat_dim = 0 if len(state.size()) == 2 else 1

        lx, hid = self.lstm1(state)
        lx = self.silu(lx)
        lx, hid = self.lstm2(lx, hid)
        lx = self.silu(lx)
        lx, _ = self.lstm3(lx, hid)
        if cat_dim == 0:
            lx = torch.sigmoid(lx)[-1, :]
            # lx = torch.sigmoid(lx)[0:, -1, :]
        else:
            lx = torch.sigmoid(lx)[-1, :, :]

        lx = torch.flatten(lx, start_dim=cat_dim).squeeze(cat_dim)
        x = self.silu(self.fc1(lx))
        x = self.silu(self.fc2(x))
        x = self.fc3(x)
        return x