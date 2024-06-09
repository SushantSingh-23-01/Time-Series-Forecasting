import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from RoPE_tf import RoPEModelBlock

plt.style.use('ggplot')

####################################################################
#                       Note
# Model often overfits or goes off very quickly.
# Hyperparameters can be further tuned to obtain far better results.
####################################################################


@dataclass
class ModelArgs:
    batch_size: int = 16
    seq_len: int = 64
    embed_dim: int = 128
    num_q_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int  = embed_dim // num_q_heads
    num_blocks: int = 4
    dropout: float = 0.1
    proj_factor:int = 4
    window_size:float = 25
    
    
@dataclass
class TrainingArgs:
    epochs: int = 1e1
    learning_rate: float = 1e-3
    sc_gamma: float = 0.999
    use_mixed_precision: bool = True
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_model_path: str = r'models/tf_ts_chkpt_1.pt'
    load_model_path: str = r'models/tf_ts_chkpt_1.pt' # None on the first run


class TimeSeriesModel(nn.Module):
    def __init__(self, seq_len, embed_dim, head_dim, num_q_heads, num_kv_heads, window_size, dropout, proj_factor, num_blocks):
        super(TimeSeriesModel, self).__init__()
        self.lin = nn.Linear(1, embed_dim)
        self.transformer_blocks = nn.Sequential(*[RoPEModelBlock(seq_len, embed_dim, head_dim, num_q_heads, num_kv_heads, window_size, dropout, proj_factor) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.out_layer = nn.Linear(seq_len * embed_dim, seq_len)
        
    def forward(self, x):
        x = self.lin(x.unsqueeze(-1))
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        x = self.out_layer(torch.flatten(x, -2, -1))
        return x

class MinMaxScaler:
    def normalize(self, x, new_min=0, new_max=1):
        self.x = x
        self.x_min = torch.min(x)
        self.x_max = torch.max(x)

        self.new_max = new_max
        self.new_min = new_min
        return (self.x - self.x_min) / (self.x_max - self.x_min) * (new_max - new_min) + new_min
    
    def renormalize(self, x):
        return (x - self.new_min) / (self.new_max - self.new_min) * (self.x_max - self.x_min) + self.x_min

class Loader:      
    def get_label_target(self, data, seq_len):
        label, target = [], []
        for i in range(0, len(data) - seq_len - 1):
            label.append(data[i:i+seq_len])
            target.append(data[i+1:i+seq_len+1])
        x = torch.stack(label)
        y = torch.stack(target)
        return x, y  
        
    def get_dataloader(self, x, y, batch_size)->torch.tensor:
        datasize = len(x)
        train_dataset = TensorDataset(x[:int(0.8*datasize)], y[:int(0.8*datasize)])
        val_dataset = TensorDataset(x[int(0.8*datasize):int(0.9*datasize)], y[int(0.8*datasize):int(0.9*datasize)])
        test_dataset = TensorDataset(x[int(0.9*datasize):],y[int(0.9*datasize):])
        
        train_loader = DataLoader(train_dataset,batch_size,shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
        return train_loader, val_loader, test_loader
              
    def __call__(self, data, batch_size, seq_len):
        x, y = self.get_label_target(data, seq_len)
        train_loader, val_loader, test_loader = self.get_dataloader(x, y, batch_size)
        return train_loader, val_loader, test_loader

def save_checkpoint(model, optimizer, epoch:int, save_path:str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, save_path)

def load_checkpoint(model, optimizer, load_path:str, device,):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']    
    return model, optimizer, epoch

def train(model, optimizer, scheduler, train_loader, valid_loader, args:ModelArgs, targs:TrainingArgs):
    if targs.load_model_path is not None:
        model, optimizer, epoch = load_checkpoint(model, optimizer, targs.load_model_path, targs.device)
    try:
        model.train()
        for epoch in tqdm(range(int(targs.epochs)), desc = 'Epochs' , total = int(targs.epochs), maxinterval= targs.epochs // 10):
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                y_pred = model(x)
                loss = F.mse_loss(y_pred, y)
                nn.utils.clip_grad_norm_(model.parameters(),5)
                loss.backward()
                optimizer.step()
              
            model.eval()
            with torch.no_grad():
                for x, y in valid_loader:
                    y_pred = model(x)
                    val_loss = F.mse_loss(y_pred, y)
                
            if epoch % (targs.epochs / 10) == 0:
                gpu_usage = round(torch.cuda.memory_reserved(0)/1024**3,1)
                #print(f'\nEpoch: {epoch} | Train_loss : {loss.item():.4f} | lr : {scheduler.get_last_lr()[0]:.2e} | GPU usage : {gpu_usage}\n')
                print(f'\nEpoch: {epoch} | Train_loss : {loss.item():.6f} | Val_loss : {val_loss.item():.4f} | lr : {scheduler.get_last_lr()[0]:.2e} | GPU usage : {gpu_usage}\n')
            model.train()
            scheduler.step()
        
    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, epoch, targs.save_model_path)
    if targs.save_model_path is not None:
        save_checkpoint(model, optimizer, epoch, targs.save_model_path)

def diagnose_model(model, batch_size, seq_len, device):
    x = torch.randn(batch_size, seq_len, device=device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True, profile_memory=True) as prof:
        with record_function('model_inference'):
            model(x)

    print(prof.key_averages().table(sort_by="cuda_cltime_total", row_limit=10))
    print('Parmameters:',sum(param.numel() for param in model.parameters()))
    
def predict(model, x, gen_len):
    _,seq_len = x.shape
    model.eval()
    with torch.inference_mode():
        for _ in range(gen_len):
            x_trunc = x[:,-seq_len:]
            y_pred = model(x_trunc)   
            x = torch.cat((x_trunc, y_pred[:,-1].view(1,-1)), dim=-1)   
    return x[:,-gen_len:]


def main():
    model = TimeSeriesModel(
        ModelArgs.seq_len,
        ModelArgs.embed_dim,
        ModelArgs.head_dim,
        ModelArgs.num_q_heads,
        ModelArgs.num_kv_heads,
        ModelArgs.window_size,
        ModelArgs.dropout,
        ModelArgs.proj_factor,
        ModelArgs.num_blocks
        )
    model.to(TrainingArgs.device)
    #diagnose_model(model, ModelArgs.batch_size, ModelArgs.seq_len, TrainingArgs.device)
    
    path = r'datasets\Microsoft_Stock.csv'
    df = pd.read_csv(path)
    df.set_index('Date',inplace=True)
    df.index = pd.to_datetime(df.index).date
    data = torch.tensor(df['High'].tolist(),device=TrainingArgs.device)
    scaler = MinMaxScaler()
    scaled_data = scaler.normalize(data)
    
    loader = Loader()
    train_loader, val_loader, test_loader = loader(scaled_data, ModelArgs.batch_size, ModelArgs.seq_len)
    optimizer = torch.optim.AdamW(model.parameters(),TrainingArgs.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=TrainingArgs.sc_gamma)
    
    train(model, optimizer, scheduler, train_loader, val_loader, ModelArgs, TrainingArgs)
    if TrainingArgs.load_model_path is not None:
        model,_,_ = load_checkpoint(model, optimizer, TrainingArgs.load_model_path, TrainingArgs.device)


    gen_len = 64
    x = scaled_data[-2*ModelArgs.seq_len:-ModelArgs.seq_len].view(1,-1)
    pred = predict(model, x , gen_len)
    pred = scaler.renormalize(pred.view(-1,))
    
    datasize = data.shape[0]
    
    plt.plot(df.index[0:datasize-gen_len], data[:-gen_len].cpu().detach().numpy(),label='prev_true')
    plt.plot(df.index[datasize-gen_len-1: datasize], data[-gen_len-1:].cpu().detach().numpy(), label='true')
    plt.plot(df.index[datasize-gen_len:datasize], pred.cpu().detach().numpy(), label='predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Microsoft Stock Price Overtime')
    plt.legend()
    plt.show()
    
if __name__ =='__main__':
    main()
