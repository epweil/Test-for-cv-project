import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset

import os
import torch
import pytorch_lightning as pl
from share import *
from cldm.model import create_model
import torch.multiprocessing as mp
from share import *


from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./procedural_generation_training_data/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
    
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
    
        source_path = './procedural_generation_training_data/' + source_filename
        target_path = './procedural_generation_training_data/' + target_filename
    
        # Check file existence
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target file not found: {target_path}")
    
        # Load images
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)
    
        # Validate image loading
        if source is None:
            raise ValueError(f"Failed to load source image: {source_path}")
        if target is None:
            raise ValueError(f"Failed to load target image: {target_path}")
    
        # Convert from BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    
        # Normalize source images to [0, 1]
        source = source.astype(np.float32) / 255.0
    
        # Normalize target images to [-1, 1]
        target = (target.astype(np.float32) / 127.5) - 1.0
    
        return dict( txt=prompt, hint=source, jpg=target)
dataset = MyDataset()
print(len(dataset))

item = dataset[0]
txt = item['txt']
hint = item['hint']
jpg = item['jpg']
print(txt)
print(hint.shape)
print(jpg.shape)

input_path = './models/v2-1_512-ema-pruned.ckpt'
output_path = './models/control_sd21_ini.ckpt'

# Ensure paths are valid
# assert os.path.exists(input_path), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
# assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'





def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models/cldm_v21.yaml')

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
print('Done.')

# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)

