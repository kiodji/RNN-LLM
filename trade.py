import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import time
import os
from timeit import default_timer as timer    
import tkinter as tk

# Definirea modelului RNN în PyTorch
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Strat LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Strat fully connected
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Funcție pentru conectare la MetaTrader 5
def conectare_mt5(username, password, server, path):
    if not mt5.initialize(path=path):
        print("Inițializare eșuată!")
        return False
    
    autorizat = mt5.login(username, password, server)
    if not autorizat:
        print("Autentificare eșuată! Verifică username, parola sau serverul.")
        return False
    
    print("Conectat cu succes la MetaTrader 5!")
    return True

# Funcție pentru colectarea datelor istorice
def obtinere_date(simbol, timeframe, data_start, data_end):
    timeframe_map = {
        "1m": mt5.TIMEFRAME_M1,
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1
    }
    tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)  # Default la 1 minut
    
    baruri = mt5.copy_rates_range(simbol, tf, data_start, data_end)
    if baruri is None or len(baruri) == 0:
        print(f"Nu s-au obținut date pentru {simbol}!")
        return None
    
    df = pd.DataFrame(baruri)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Coloane disponibile: {df.columns.tolist()}")
    return df

# Funcție pentru prepararea datelor
def preparare_date(df, secventa_length=60):
    available_columns = df.columns.tolist()
    required_columns = []
    
    if 'open' in available_columns:
        required_columns.append('open')
    if 'high' in available_columns:
        required_columns.append('high')
    if 'low' in available_columns:
        required_columns.append('low')
    if 'close' in available_columns:
        required_columns.append('close')
    else:
        raise ValueError("Coloana 'close' este obligatorie!")
    if 'tick_volume' in available_columns:
        required_columns.append('tick_volume')
    if 'rsi' in available_columns:
        required_columns.append('rsi')
    
    print(f"Folosim coloanele: {required_columns}")
    
    data = df[required_columns].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    price_index = required_columns.index('close')
    
    for i in range(secventa_length, len(data_scaled)):
        X.append(data_scaled[i-secventa_length:i, :])
        y.append(data_scaled[i, price_index])
    
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return X_train, X_test, y_train, y_test, scaler, required_columns, price_index

# Funcție pentru crearea și antrenarea modelului RNN
def creare_model_pytorch(X_train, y_train, X_test, y_test, input_dim, hidden_dim=50, num_layers=3, learning_rate=0.001, epochs=25, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Folosim dispozitivul: {device}")
    
    model = RNNModel(input_dim, hidden_dim, num_layers, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                test_loss += criterion(outputs, batch_y).item()
        
        print(f'Epoca {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}')
    
    torch.save(model.state_dict(), "model_rnn_pytorch.pth")
    print("Model salvat cu succes!")
    return model, device

# Funcție pentru predicție
def predictie_pret_pytorch(model, date_actuale, scaler, secventa_length, device, required_columns, price_index):
    model.eval()
    with torch.no_grad():
        date_scaled = scaler.transform(date_actuale)
        X = date_scaled[-secventa_length:].reshape(1, secventa_length, date_actuale.shape[1])
        X = torch.tensor(X, dtype=torch.float32).to(device)
        predictie_scaled = model(X).cpu().numpy()
        
        temp = np.zeros((1, date_actuale.shape[1]))
        temp[0, price_index] = predictie_scaled[0, 0]
        predictie = scaler.inverse_transform(temp)[0, price_index]
        return predictie

# Funcție pentru executarea tranzacției
def executare_tranzactie(simbol, directie, volum, stop_loss_pips, take_profit_pips):
    # Verificăm conectivitatea
    if mt5.terminal_info() is None:
        print("MT5 nu este conectat!")
        return False

    # Verificăm simbolul
    info = mt5.symbol_info(simbol)
    if info is None:
        print(f"Simbolul {simbol} nu este disponibil!")
        return False

    # Verificăm datele tick
    tick = mt5.symbol_info_tick(simbol)
    if tick is None:
        print(f"Nu s-au obținut date tick pentru {simbol}! Piața este posibil închisă.")
        return False

    point = info.point  # ex. 0.0001 pentru EUR/USD
    digits = info.digits  # Numărul de zecimale

    if directie == "BUY":
        tip_tranzactie = mt5.ORDER_TYPE_BUY
        pret = tick.ask
        sl = pret - stop_loss_pips * point
        tp = pret + take_profit_pips * point
    else:
        tip_tranzactie = mt5.ORDER_TYPE_SELL
        pret = tick.bid
        sl = pret + stop_loss_pips * point
        tp = pret - take_profit_pips * point

    cerere = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": simbol,
        "volume": float(volum),  # Asigurăm că volumul e float
        "type": tip_tranzactie,
        "price": pret,
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "magic": 234000,
        "comment": "Tranzacție RNN PyTorch",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    print(f"Trimit cerere: {cerere}")
    rezultat = mt5.order_send(cerere)
    
    if rezultat.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Eroare la executarea tranzacției: {rezultat.retcode} - {rezultat.comment}")
        return False
    
    print(f"Tranzacție executată: {directie} {simbol} {volum} loturi")
    return True

# Funcție principală pentru strategia de tranzacționare
def train(username, password, server, path, simbol, timeframe, perioada_secventa=60):
    if not conectare_mt5(username, password, server, path):
        return
    
    # Date istorice pentru antrenare
    data_end = datetime.datetime.now()
    data_start = data_end - datetime.timedelta(days=365)
    print("Colectare date istorice...")
    df = obtinere_date(simbol, timeframe, data_start, data_end)
    if df is None:
        return

    # Preparare date
    print("Preparare date pentru antrenare...")
    X_train, X_test, y_train, y_test, scaler, required_columns, price_index = preparare_date(df, perioada_secventa)
    
    # Antrenare model
    print("Antrenare model RNN în PyTorch...")
    model, device = creare_model_pytorch(
        X_train, y_train, X_test, y_test,
        input_dim=len(required_columns),
        hidden_dim=64,
        num_layers=3,
        learning_rate=0.001,
        epochs=60,
        batch_size=32
    )


# Funcție principală pentru strategia de tranzacționare
def strategie_tranzactionare(username, password, server, path, simbol, timeframe, perioada_secventa=60):
    if not conectare_mt5(username, password, server, path):
        return
    
    # Date istorice pentru antrenare
    data_end = datetime.datetime.now()
    data_start = data_end - datetime.timedelta(days=365)
    print("Colectare date istorice...")
    df = obtinere_date(simbol, timeframe, data_start, data_end)
    if df is None:
        return

    # Preparare date
    print("Preparare date pentru antrenare...")
    X_train, X_test, y_train, y_test, scaler, required_columns, price_index = preparare_date(df, perioada_secventa)
    
    # Antrenare model
    print("Antrenare model RNN în PyTorch...")
    model, device = creare_model_pytorch(
        X_train, y_train, X_test, y_test,
        input_dim=len(required_columns),
        hidden_dim=64,
        num_layers=3,
        learning_rate=0.001,
        epochs=60,
        batch_size=32
    )
    
    # Buclă de tranzacționare
    print("Începere tranzacționare...")
    profit_threshold = 0.0005  # 0.15%
    stop_loss_pips = 20  # 20 pips
    take_profit_pips = 30  # 30 pips
    previous_direction = None
    
    try:
        while True:
            # Verificăm conectivitatea înainte de fiecare ciclu
            if mt5.terminal_info() is None:
                print("Reconectare la MT5...")
                if not conectare_mt5(username, password, server, path):
                    time.sleep(60)
                    continue

            # Obținere date actualizate
            data_end_current = datetime.datetime.now()
            data_start_current = data_end_current - datetime.timedelta(days=5)
            df_current = obtinere_date(simbol, timeframe, data_start_current, data_end_current)
            if df_current is None or len(df_current) < perioada_secventa:
                print("Nu sunt suficiente date pentru predicție!")
                time.sleep(60)
                continue
            
            date_actuale = df_current[required_columns].values
            pret_prezis = predictie_pret_pytorch(model, date_actuale, scaler, perioada_secventa, device, required_columns, price_index)
            tick = mt5.symbol_info_tick(simbol)
            if tick is None:
                print("Nu s-au obținut date tick!")
                time.sleep(60)
                continue
            pret_actual = tick.bid
            
            diferenta_procentuala = (pret_prezis - pret_actual) / pret_actual
            print(f"Preț actual: {pret_actual}, Preț prezis: {pret_prezis}, Diferență: {diferenta_procentuala*100:.3f}%")
            
            # if diferenta_procentuala > profit_threshold:
            #     directie = "BUY"
            # elif diferenta_procentuala < -profit_threshold:
            #     directie = "SELL"
            if pret_prezis > pret_actual:
                directie = "BUY"
            elif pret_prezis < pret_actual:
                directie = "SELL"
            else:
                directie = None
                print("Fără semnal de tranzacționare.")
            
            if directie and directie != previous_direction:
                volum = 0.1
                if executare_tranzactie(simbol, directie, volum, stop_loss_pips, take_profit_pips):
                    previous_direction = directie
            
            print("Așteptare până la următoarea verificare...")
            time.sleep(300)  # 5 minute
            
    except KeyboardInterrupt:
        print("\nOprire strategie de tranzacționare...")
        mt5.shutdown()
        
        
def runButton():
    username = 101724117  # Înlocuiți cu ID-ul dvs.
    password = "u>9yC8@N"  # Înlocuiți cu parola dvs.
    server = "FBS-Demo"  # Înlocuiți cu serverul brokerului dvs.
    terminal_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # Înlocuiți cu calea către MT5
    
    # Setări tranzacționare
    simbol = "EURUSDw"
    timeframe = "1h"
    
    # Pornire strategie
    train(username, password, server, terminal_path, simbol, timeframe)

def testButton():
    username = 101724117  # Înlocuiți cu ID-ul dvs.
    password = "u>9yC8@N"  # Înlocuiți cu parola dvs.
    server = "FBS-Demo"  # Înlocuiți cu serverul brokerului dvs.
    terminal_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # Înlocuiți cu calea către MT5
    
    # Setări tranzacționare
    simbol = "EURUSD"
    timeframe = "1h"
    
    # Pornire strategie
    strategie_tranzactionare(username, password, server, terminal_path, simbol, timeframe)

root = tk.Tk()
root.title("Model")
root.geometry("800x600")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# TextArea pentru afișare log-uri/rezultate
text_area = tk.Text(root, height=10, width=80)
text_area.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
text_area.insert(tk.END, "Hello world")

button = tk.Button(root, text="Run", command=runButton)
button.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

button = tk.Button(root, text="Test", command=testButton)
button.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

scrollbar_y = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar_x = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=tk.TOP, fill=tk.X)

canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

second_frame = tk.Frame(canvas)

canvas.create_window((0,0), window=second_frame, anchor="nw")

root.mainloop()

