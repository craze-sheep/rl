import numpy as np
import torch
from torch import nn
from pathlib import Path
PATH_DIR = Path(__file__).parent

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = Model()
model.step = 321
model_state_dict_cpu = {k: v.clone().cpu() for k, v in model.state_dict().items()}
data = {
    'model_state_dict': model_state_dict_cpu,
    'step': model.step,
    'obstacles': np.zeros([128, 128])
}

path_model = PATH_DIR / 'model.pkl'
torch.save(data, path_model)
del(model)
del(data)

model = Model()
data = torch.load(path_model, map_location='cpu')
model.load_state_dict(data['model_state_dict'])
model.step = data['step']
print(type(data['model_state_dict']))
print(data['model_state_dict'])
print(type(data['step']))
print(type(data['obstacles']))
print(data['obstacles'].shape)
