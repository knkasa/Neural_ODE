import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

ticker = ‘RELIANCE.NS’
data = yf.download(ticker, start=”2020–01–01", end=”2023–01–01", interval=’1d’)
data[‘Returns’] = data[‘Close’].pct_change()
data[‘Volume_Change’] = data[‘Volume’].pct_change()
data = data[[‘Close’, ‘Returns’, ‘Volume_Change’]].dropna()

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) — seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0] # Predicting only the Close price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
X, y = create_sequences(data_scaled, seq_length)

X_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(y).float()

# Neural ODE function
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
            )

    def forward(self, t, h):
        return self.linear(h)

# Combined LSTM and Neural ODE
class LSTMNeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMNeuralODE, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.ode_func = ODEFunc(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        h_t = lstm_out[:, -1, :] # Use last hidden state for ODE
        t = torch.linspace(0, 1, steps=10).to(x.device) # Time steps for ODE solver
        h_t_ode = odeint(self.ode_func, h_t, t, method=’rk4')[-1] # Final ODE state
        output = self.fc(h_t_ode)
        return output

# Model setup and training
input_dim = 3
hidden_dim = 50
output_dim = 1
model = LSTMNeuralODE(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
    print(f’Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}’)
    
# Predictions and plot
model.eval()
with torch.no_grad():
    predictions = model(X_train).numpy()
    
predictions_padded = np.column_stack((predictions, np.zeros((len(predictions), 2))))
predictions = scaler.inverse_transform(predictions_padded)[:, 0]

actual_padded = np.column_stack((y_train.numpy().reshape(-1, 1), np.zeros((len(y_train), 2))))
actual = scaler.inverse_transform(actual_padded)[:, 0]    

plt.plot(actual, label=’Actual Prices’)
plt.plot(predictions, label=’Predicted Prices’)
plt.legend()
plt.show()

