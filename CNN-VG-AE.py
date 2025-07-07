# -*- coding: utf-8 -*-

#torch==2.0.1          
#pandas==2.0.3         
#numpy==1.24.3         
#matplotlib==3.7.2     
#scipy==1.10.1         
#scikit-learn==1.3.0   
#openpyxl==3.1.2       

#conda create -n soil_model python=3.9
#conda activate soil_model
#pip install torch pandas numpy matplotlib scipy scikit-learn openpyxl

"""
The code implements a soil moisture transport model based on physical constraints, combining deep learning and physical equations.
Key features:
1. Load and preprocess soil moisture data
2. Build a physical information neural network model
3. Train the model and save the results
4. Visualize model predictions and physical parameters
Please cite the relevant paper when citing this code.
"""

# ===================== Parameter Configuration =====================
# 1. File path configuration
DATA_PATH = r"D:\桌面"  
EXCEL_FILE = "Soil moisture content.xlsx"  
SHEET_NAME = "30min"  

# 2. Output directory configuration
SAVE_DIR = r"D:\桌面\CNN-VG-AE\Hydraulic parameters"  
TIME_DIR = r"D:\桌面\CNN-VG-AE\Time"  
PLOT_DIR = r"D:\桌面\CNN-VG-AE\Training_plot"  
MOISTURE_DIR = r"D:\桌面\CNN-VG-AE\Moisture Content"  
LOSS_DIR = r"D:\桌面\CNN-VG-AE\Loss"  

# 3. Physical parameter configuration
THETA_R = 0.01464  # Residual water content
THETA_S = 0.298  # Saturated water content
ALPHA = 0.08267192  # van Genuchten parameter α
N = 1.484226  # van Genuchten parameter n
K_S = 407.3903  # Saturated water conductivity (mm/h)
EPSILON = 1e-5  # Numerical stability constant

# 4. Space configuration
# Note: The raw data column names are "3C", "5C", etc., which indicate the centimeter depth
X_ORIGINAL_CM = [3, 5, 10, 30, 50]  # Original Measurement Depth (cm)
X_ORIGINAL = [d/100 for d in X_ORIGINAL_CM]  
X_NEW_STEP = 0.005  # Interpolation step size (m)
X_PLOT = [0.03, 0.05, 0.1, 0.3, 0.5]  # Drawing Depth (m)
X_RANGE = (0.03, 0.5)  # Spatial extent (m)

# 5. Time configuration
TIME_WINDOW = "2D"  # The data grouping time window
VALID_RANGE = (0.01465, 0.298)  # Valid data range

# 6. Model parameters
MODEL_HIDDEN_LAYERS = [64 * 2 , 20 , 1]  # Hidden layer configuration
CK_HIDDEN_LAYERS = [16, 8, 2]  # Autoencoder hidden layer

# 7. Training parameters
NUM_OUTER_LOOPS = 200  # Number of external cycles
NUM_INNER_EPOCHS = 1000  # The number of times the water content model was trained
NUM_INNER_EPOCHS_CK = 1000  # Number of training iterations for hydraulic parameters
BATCH_SIZE = 32  
INITIAL_LR = 1e-2  # Initial learning rate
CK_LR = 0.001  # Hydraulic parameter learning rate
RECONSTRUCTION_WEIGHT = 0.01  # Reconstruction loss weight

# 8. Random seeds
SEED = 42  # Global random seeds

# ===================== Importing dependency libraries =====================
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ===================== Model Definition =====================
#Initialize the autoencoder, layers: network layer structure, e.g. [1, 16, 8, 2]
class Autoencoder1D(nn.Module):
    def __init__(self, layers):
        super(Autoencoder1D, self).__init__()
        encoder_layers = []
        for i in range(len(layers) - 1):
            encoder_layers.append(nn.Linear(layers[i], layers[i+1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers[:-1])  
        
        decoder_layers = []
        rev_layers = list(reversed(layers))
        for i in range(len(rev_layers) - 1):
            decoder_layers.append(nn.Linear(rev_layers[i], rev_layers[i+1]))
            if i < len(rev_layers) - 2:  
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
#Complex linear transformation layers
class ComplexLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexLinear, self).__init__()
        self.real_weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.real_bias = nn.Parameter(torch.randn(output_dim) * 0.001 + 0.01)
        self.imag_bias = nn.Parameter(torch.randn(output_dim) * 0.001 - 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(output_dim) * 0.01 + 0.05)
        self.lambda_k2 = nn.Parameter(torch.randn(output_dim) * 0.01 + 0.05)
    
    def forward(self, x):
        if torch.is_complex(x):
            x_real = torch.real(x)
            x_imag = torch.imag(x)
        else:
            x_real = x
            x_imag = torch.zeros_like(x)
        
        real_1 = torch.matmul(x_real, self.real_weights) + self.real_bias
        real = torch.square(real_1)
        complex_output = real + torch.square(self.imag_bias)
        
        complex_output_1 = real_1 / complex_output
        output_1 = complex_output_1 * self.lambda_k1
        
        complex_output_2 = self.imag_bias / complex_output
        output_2 = complex_output_2 * self.lambda_k2
        
        x = output_1 + output_2
        x = torch.mean(x, dim=-1)
        return x.view(-1, 1)

#Soil moisture prediction model
class ConvNet(nn.Module):
    def __init__(self, hidden_layers):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1) 
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1) 
        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(hidden_layers[0], hidden_layers[1]) 
        self.fc2 = nn.Linear(hidden_layers[1], hidden_layers[2]) 

        self.relu = nn.ReLU() 
        self.tanh = nn.Tanh() 

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x)) 
        x = self.relu(self.conv2(x)) 
        x = self.flatten(x) 
        x = self.tanh(self.fc1(x)) 
        x = self.fc2(x) 
        return x

# ===================== Functions of physical equations =====================
def Theta(u):
    return (u - THETA_R) / (THETA_S - THETA_R)

def h_star(u):
    m = 1 - 1/N
    return (1 / ALPHA) * (((THETA_S - THETA_R) / (u - THETA_R))**(1/m) - 1)**(1/N)

def k_star(u):
    theta_val = Theta(u)
    m = 1 - 1/N
    return K_S * (theta_val**(0.5)) * (1 - (1 - theta_val**(1/m))**m)**2

# ===================== Utility functions =====================
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Create a directory: {path}")
    return path

def load_and_preprocess_data(file_path, sheet_name):
    print(f"Load data: {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet_name, parse_dates=['date'])
    columns_to_keep = ["date"] + [f"{depth}centimeter" for depth in X_ORIGINAL_CM]
    print(f"Selected columns: {columns_to_keep}")
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing in the data: {missing_cols}")
    df = df[columns_to_keep]
    print(f"Data Preview:\n{df.head()}")
    return df

def interpolate_data(df, x_original_cm, x_new_step):
    print("Execute spatial interpolation...")
    x_original_m = [d/100 for d in x_original_cm]
    x_new = np.round(np.arange(min(x_original_m), max(x_original_m)+x_new_step, x_new_step), 3)
    interpolated_values = []
    y_values = df[[f"{depth}centimeter" for depth in x_original_cm]].values
    for i in range(y_values.shape[0]):
        interp_fn = interp1d(x_original_m, y_values[i, :], kind='linear', fill_value="extrapolate")
        y_interpolated = interp_fn(x_new)
        interpolated_values.append(y_interpolated)
    interpolated_df = pd.DataFrame(interpolated_values, columns=x_new)
    interpolated_df['date'] = df['date'].values
    return interpolated_df[['date'] + list(x_new)]

def filter_valid_data(df):
    print("Filter out invalid data...")
    float_columns = [col for col in df.columns if isinstance(col, float)]
    if not float_columns:
        print("Warning: There are no floating-point number columns available for filtering")
        return df
    valid_plot_columns = [col for col in X_PLOT if col in float_columns]
    if not valid_plot_columns:
        print(f"Warning: Columns in X_PLOT {X_PLOT} Not present in the DataFrame")
        return df
    condition = (df[valid_plot_columns] > VALID_RANGE[0]) & (df[valid_plot_columns] < VALID_RANGE[1])
    return df[condition.all(axis=1)]

def plot_results(t_obs, x_obs, u_obs, u_pred, epoch):
    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(t_obs, x_obs, u_obs, c=u_obs, cmap='viridis', marker='o', s=20)
    ax1.set_title(f'Observations - Epoch {epoch+1}')
    ax1.set_xlabel('Normalized time')
    ax1.set_ylabel('Depth (m)')
    ax1.set_zlabel('moisture content')
    fig.colorbar(sc1, ax=ax1, label='moisture content')

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(t_obs, x_obs, u_pred, c=u_pred, cmap='viridis', marker='o', s=20)
    ax2.set_title(f'Observations - Epoch {epoch+1}')
    ax2.set_xlabel('Normalized time')
    ax2.set_ylabel('Depth (m)')
    ax2.set_zlabel('moisture content')
    fig.colorbar(sc2, ax=ax2, label='moisture content')
    
    plot_path = os.path.join(PLOT_DIR, f"3d_comparison_epoch_{epoch+1}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Save the 3D comparison chart to: {plot_path}")

def prepare_datasets(interpolated_df):
    print("Prepare the training data set...")
    grouped = interpolated_df.groupby(pd.Grouper(key='date', freq=TIME_WINDOW))
    dataframes = [group for _, group in grouped]
    if dataframes:
        dataframes.pop(0)
        if dataframes:  
            dataframes.pop(-1)
    valid_dataframes = []
    for df in dataframes:
        if not df.empty:
            try:
                filtered_df = filter_valid_data(df)
                if not filtered_df.empty:
                    valid_dataframes.append(filtered_df)
            except Exception as e:
                print(f"An error occurred while filtering the data: {e}")
    for df in valid_dataframes:
        min_date = df['date'].min()
        max_date = df['date'].max()
        time_span = (max_date - min_date).total_seconds()
        if time_span > 0:
            df['normalized_date'] = (df['date'] - min_date).dt.total_seconds() / time_span
        else:
            df['normalized_date'] = 0.0
    return valid_dataframes

# ===================== Loss function =====================
def loss_function(model, c_net, k_net, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    xt = torch.cat([t, x], dim=1)
    u = model(xt)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    k_star_val = k_star(u)
    c_star_val = -h_star(u)
    k_u = k_net(k_star_val)
    c_u = c_net(c_star_val)
    k_x = torch.autograd.grad(k_net(k_star_val), x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    c_x = torch.autograd.grad(c_net(c_star_val), x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    pde_loss = u_t - torch.autograd.grad(k_u * c_x * u_x, x, torch.ones_like(u_x), create_graph=True)[0] + k_x
    return torch.mean(pde_loss ** 2)

def generate_interior_samples(t_values, num_samples=1000):
    if len(t_values) == 0:
        t_values = [0.0]  
    t_sample = np.random.choice(t_values, num_samples, replace=True)
    x_sample = np.random.uniform(X_RANGE[0], X_RANGE[1], num_samples)
    t_tensor = torch.tensor(t_sample, dtype=torch.float32).reshape(-1, 1)
    x_tensor = torch.tensor(x_sample, dtype=torch.float32).reshape(-1, 1)
    return t_tensor, x_tensor

def observed_data_loss(model, t_observed, x_observed, u_observed):
    t_observed.requires_grad_(True)
    x_observed.requires_grad_(True)
    xt = torch.cat([t_observed, x_observed], dim=1)
    u_pred = model(xt)
    return torch.mean((u_pred - u_observed) ** 2)

def reconstruction_loss(c_net, k_net):
    u_values = np.linspace(THETA_R + EPSILON, THETA_S - EPSILON, 200)
    u_tensor = torch.tensor(u_values, dtype=torch.float32).reshape(-1, 1)
    k_star_val = k_star(u_tensor)
    c_star_val = -h_star(u_tensor)
    k_u = k_net(k_star_val)
    c_u = c_net(c_star_val)
    criterion = nn.MSELoss()
    return criterion(k_u, k_star_val) + criterion(c_u, c_star_val)

# ===================== Data generation function =====================
def get_observed_data(dataframes, index):
    if index >= len(dataframes):
        index = 0
    df = dataframes[index]
    depth_cols = [col for col in df.columns if isinstance(col, float)]
    if not depth_cols:
        depth_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col not in ['normalized_date']]
    if not depth_cols:
        depth_cols = X_PLOT
        print(f"Warning: Use the default drawing columns {depth_cols}")
    depth_cols = sorted(depth_cols)
    observed_data = df[depth_cols].values
    t_values = df['normalized_date'].values
    t_dates = df['date'].values  # Original date and time
    t_obs, x_obs, u_obs, date_obs = [], [], [], []
    
    for i, t in enumerate(t_values):
        current_date = t_dates[i]  # Gets the original datetime of the current record
        for j, x in enumerate(depth_cols):
            t_obs.append(t)
            x_obs.append(x)
            u_obs.append(observed_data[i, j])
            date_obs.append(current_date)  # Add the original datetime for each sample point
    
    t_tensor = torch.tensor(t_obs, dtype=torch.float32).reshape(-1, 1)
    x_tensor = torch.tensor(x_obs, dtype=torch.float32).reshape(-1, 1)
    u_tensor = torch.tensor(u_obs, dtype=torch.float32).reshape(-1, 1)
    
    return t_dates, t_tensor, x_tensor, u_tensor, date_obs, index

# ===================== Plotting functions =====================
def plot_results(date_obs, x_obs, u_obs, u_pred, epoch):
    # Convert datetime to numeric format (keep chronological)
    try:
        date_numeric = pd.to_datetime(date_obs).astype('int64') / 10**9
    except:
        date_numeric = np.arange(len(date_obs))  # Fallback scenario: Use indexes
    
    fig = plt.figure(figsize=(18, 9))

    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(date_numeric, x_obs, u_obs, c=u_obs, cmap='viridis', marker='o', s=20)
    ax1.set_title(f'Observed values - Epoch {epoch+1}')
    ax1.set_xlabel('Date and Time')
    ax1.set_ylabel('Depth (m)')
    ax1.set_zlabel('moisture content')
    fig.colorbar(sc1, ax=ax1, label='moisture content')
    
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(date_numeric, x_obs, u_pred, c=u_pred, cmap='viridis', marker='o', s=20)
    ax2.set_title(f'Observed values - Epoch {epoch+1}')
    ax2.set_xlabel('Date and Time')
    ax2.set_ylabel('Depth (m)')
    ax2.set_zlabel('moisture content')
    fig.colorbar(sc2, ax=ax2, label='moisture content')
    
    plot_path = os.path.join(PLOT_DIR, f"3d_comparison_epoch_{epoch+1}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"The 3D comparison image is saved to: {plot_path}")

# ===================== Main Function =====================
def main():
    create_directory(SAVE_DIR)
    create_directory(PLOT_DIR)
    create_directory(MOISTURE_DIR)
    create_directory(TIME_DIR)
    create_directory(LOSS_DIR)

    time_log_path = os.path.join(TIME_DIR, "training_time.csv")
    loss_log_path = os.path.join(LOSS_DIR, "training_loss.csv")

    time_log = pd.DataFrame(columns=["epoch", "start_time", "end_time", "duration"])
    loss_log = pd.DataFrame(columns=["epoch", "water_loss", "pde_loss", "recon_loss"])

    time_log.to_csv(time_log_path, index=False)
    loss_log.to_csv(loss_log_path, index=False)
    
    try:
        data_path = os.path.join(DATA_PATH, EXCEL_FILE)
        df = load_and_preprocess_data(data_path, SHEET_NAME)
        interpolated_df = interpolate_data(df, X_ORIGINAL_CM, X_NEW_STEP)
        print(f"Preview of the interpolated data:\n{interpolated_df.head()}")
        
        dataframes = prepare_datasets(interpolated_df)
        print(f"The number of datasets prepared: {len(dataframes)}")
        
        if not dataframes:
            print("Error: No valid dataset available for training")
            return
        
        sorted_dfs = sorted(dataframes, key=lambda df: df['date'].min())
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Use the device: {device}")
        
        c_net = Autoencoder1D([1] + CK_HIDDEN_LAYERS).to(device)
        k_net = Autoencoder1D([1] + CK_HIDDEN_LAYERS).to(device)
        model = ConvNet(MODEL_HIDDEN_LAYERS).to(device)
        
        optimizer_model = optim.Adam(model.parameters(), lr=INITIAL_LR)
        optimizer_ck = optim.Adam(
            list(c_net.parameters()) + list(k_net.parameters()), 
            lr=CK_LR
        )
        
        current_idx = 0
        for epoch in range(NUM_OUTER_LOOPS):
            epoch_start = time.time()
            
            t_dates, t_obs, x_obs, u_obs, date_obs, current_idx = get_observed_data(
                sorted_dfs, current_idx
            )
            
            t_obs = t_obs.to(device)
            x_obs = x_obs.to(device)
            u_obs = u_obs.to(device)
            
            # ========== Train a moisture model ==========
            model.train()
            water_losses = []
            for inner_epoch in range(NUM_INNER_EPOCHS):
                optimizer_model.zero_grad()
                loss = observed_data_loss(model, t_obs, x_obs, u_obs)
                loss.backward()
                optimizer_model.step()
                water_losses.append(loss.item())
                
                if (inner_epoch + 1) % 100 == 0:
                    print(f"Outer circulation {epoch+1}/{NUM_OUTER_LOOPS}, "
                          f"internal circulation {inner_epoch+1}/{NUM_INNER_EPOCHS}, "
                          f"loss: {loss.item():.4e}")
            
            avg_water_loss = np.mean(water_losses)
            
            # ========== Train a hydrological parameter model ==========
            t_np = t_obs.detach().cpu().numpy().flatten()
            t_int, x_int = generate_interior_samples(t_np)  # Use the added function
            t_int = t_int.to(device)
            x_int = x_int.to(device)
            
            c_net.train()
            k_net.train()
            pde_losses = []
            recon_losses = []
            for ck_epoch in range(NUM_INNER_EPOCHS_CK):
                optimizer_ck.zero_grad()
                pde_loss = loss_function(model, c_net, k_net, t_int, x_int)
                recon_loss = reconstruction_loss(c_net, k_net)
                total_loss = pde_loss + RECONSTRUCTION_WEIGHT * recon_loss
                total_loss.backward()
                optimizer_ck.step()
                
                pde_losses.append(pde_loss.item())
                recon_losses.append(recon_loss.item())
                
                if (ck_epoch + 1) % 100 == 0:
                    print(f"Outer circulation {epoch+1}/{NUM_OUTER_LOOPS}, "
                          f"Hydraulic parameters training {ck_epoch+1}/{NUM_INNER_EPOCHS_CK}, "
                          f"PDE loss: {pde_loss.item():.4e}, "
                          f"Reconstruction loss: {recon_loss.item():.4e}")
            
            avg_pde_loss = np.mean(pde_losses)
            avg_recon_loss = np.mean(recon_losses)
            
            # ========== Save results and logs ==========
            epoch_end = time.time()
            duration = round(epoch_end - epoch_start, 2)
            
            # 1. Keep a time log
            time_log = pd.concat([time_log, pd.DataFrame({
                "epoch": [epoch+1],
                "start_time": [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start))],
                "end_time": [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_end))],
                "duration": [duration]
            })], ignore_index=True)
            time_log.to_csv(time_log_path, index=False)
            
            # 2. Save loss log
            loss_log = pd.concat([loss_log, pd.DataFrame({
                "epoch": [epoch+1],
                "water_loss": [avg_water_loss],
                "pde_loss": [avg_pde_loss],
                "recon_loss": [avg_recon_loss]
            })], ignore_index=True)
            loss_log.to_csv(loss_log_path, index=False)
            
            # 3. Save hydraulic parameters
            u_values = np.linspace(THETA_R + EPSILON, THETA_S - EPSILON, 500)
            u_tensor = torch.tensor(u_values, dtype=torch.float32).reshape(-1, 1).to(device)
            
            with torch.no_grad():
                h_star_val = -h_star(u_tensor)
                k_star_val = k_star(u_tensor)
                h_pred = c_net(h_star_val).cpu().numpy()
                k_pred = k_net(k_star_val).cpu().numpy()
            
            param_df = pd.DataFrame({
                'u': u_values,
                'h_star': h_star_val.cpu().numpy().flatten(),
                'h_pred': h_pred.flatten(),
                'k_star': k_star_val.cpu().numpy().flatten(),
                'k_pred': k_pred.flatten()
            })
            
            param_save_path = os.path.join(SAVE_DIR, f"hydraulic_params_epoch_{epoch+1}.csv")
            param_df.to_csv(param_save_path, index=False)
            print(f"Save hydraulic parameters to: {param_save_path}")
            
            # 4. Save the prediction results
            with torch.no_grad():
                model.eval()
                xt = torch.cat([t_obs, x_obs], dim=1)
                u_pred = model(xt).cpu().numpy()
            
            pred_df = pd.DataFrame({
                't_normalized': t_obs.cpu().detach().numpy().flatten(),  # Standard Time
                'depth': x_obs.cpu().detach().numpy().flatten(),  # Depth (m)
                'u_observed': u_obs.cpu().detach().numpy().flatten(),  # Measure the water content
                'u_predicted': u_pred.flatten()  # Predict moisture content
            })
            
            pred_save_path = os.path.join(MOISTURE_DIR, f"predictions_epoch_{epoch+1}.csv")
            pred_df.to_csv(pred_save_path, index=False)
            print(f"Prediction results are saved to: {pred_save_path}")
            
            # 5. Visualize and save
            if (epoch + 1) % 10 == 0 or epoch == NUM_OUTER_LOOPS - 1:
                try:
                    t_obs_np = t_obs.detach().cpu().numpy()
                    x_obs_np = x_obs.detach().cpu().numpy()
                    u_obs_np = u_obs.detach().cpu().numpy()
                    
                    plot_results(date_obs, x_obs_np, u_obs_np, u_pred, epoch)
                    plot_path = os.path.join(PLOT_DIR, f"timeseries_epoch_{epoch+1}.png")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    plot_df = pred_df.copy()
                    plot_df['date'] = pd.to_datetime(plot_df['date'])
                    
                    plot_df = plot_df.sort_values('date')
                    
                    for depth in X_PLOT:
                        depth_df = plot_df[np.isclose(plot_df['depth'], depth, atol=0.001)]
                        if not depth_df.empty:
                            ax.plot(depth_df['date'], depth_df['u_observed'], 
                                    label=f"Observed values (x={depth}m)")
                            ax.plot(depth_df['date'], depth_df['u_predicted'], '--', 
                                    label=f"Predicted value (x={depth}m)")
                    
                    ax.set_title(f"Moisture content time series - Epoch {epoch+1}")
                    ax.set_xlabel("Date and Time")
                    ax.set_ylabel("moisture content")
                    ax.legend()
                    ax.grid(True)
                    
                    plt.gcf().autofmt_xdate()
                    
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300)
                    plt.close()
                    print(f"The time series graph is saved to: {plot_path}")
                    
                except Exception as e:
                    print(f"Failed to save visualization: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            print(f"Epoch {epoch+1}/{NUM_OUTER_LOOPS} Completion, time-consuming: {duration} second")
            current_idx = (current_idx + 1) % len(sorted_dfs)  
        
        # ========== Final save after training is completed ==========
        print("Training completed! Save the results...")
        
        # 1. Save loss curve
        loss_curve_path = os.path.join(PLOT_DIR, "loss_curves.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_log['epoch'], loss_log['water_loss'], label='Moisture model loss')
        ax.plot(loss_log['epoch'], loss_log['pde_loss'], label='PDE loss')
        ax.plot(loss_log['epoch'], loss_log['recon_loss'], label='Reconstruction loss')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(loss_curve_path, dpi=300)
        plt.close()
        print(f"The loss curve is saved to: {loss_curve_path}")
        
        # 2. Save the final hydraulic parameter chart
        final_param_path = os.path.join(PLOT_DIR, "final_hydraulic_params.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pressure head diagram
        ax1.plot(param_df['u'], param_df['h_star'], 'b-', label='Actual value')
        ax1.plot(param_df['u'], param_df['h_pred'], 'r--', label='Predicted value')
        ax1.set_title("Pressure head h(u)")
        ax1.set_xlabel("u")
        ax1.set_ylabel("h(u)")
        ax1.legend()
        ax1.grid(True)
        
        # Hydraulic conductivity chart
        ax2.plot(param_df['u'], param_df['k_star'], 'b-', label='Actual value')
        ax2.plot(param_df['u'], param_df['k_pred'], 'r--', label='Predicted value')
        ax2.set_title("Hydraulic conductivity k(u)")
        ax2.set_xlabel("u")
        ax2.set_ylabel("k(u)")
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(final_param_path, dpi=300)
        plt.close()
        print(f"The final hydraulic diagram is saved to: {final_param_path}")
        
        # 3. Save all logs
        time_log.to_csv(time_log_path, index=False)
        loss_log.to_csv(loss_log_path, index=False)

        print("All results have been saved!")
        
    except Exception as e:
        print(f"An error occurred during the training process: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to save the current log
        try:
            time_log.to_csv(time_log_path, index=False)
            loss_log.to_csv(loss_log_path, index=False)

            print("The current log has been saved")
        except:
            print("Failed to save the logs")

if __name__ == "__main__":
    print("="*50)
    print("Soil moisture migration model training begins")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    main()
    
    print("="*50)
    print("Model training completed")
    print(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)