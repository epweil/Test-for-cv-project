
import os
import torch
import pytorch_lightning as pl
from share import *
from cldm.model import create_model

# Hard-coded paths
input_path = './models/v2-1_512-ema-pruned.ckpt'
output_path = './models/control_sd21_ini.ckpt'

# Ensure paths are valid
assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

# Function to extract node names
def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

# Load model configuration
model = create_model(config_path='./models/cldm_v21.yaml')

# Load pretrained weights
pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

# Initialize the state dictionary for the new model
scratch_dict = model.state_dict()
target_dict = {}

# Map weights from the pretrained model
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

# Load the updated state dictionary into the model
model.load_state_dict(target_dict, strict=True)

# Save the updated model
torch.save(model.state_dict(), output_path)
print('Done.')

# should see something like
# These weights are newly added ...
# These weights are newly added ...
# Done.


# # Training!

# In[ ]:


# Taken from tutorial_train_sd21.py
# https://github.com/lllyasviel/ControlNet/blob/main/tutorial_train_sd21.py
from share import *


from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = './models/control_sd21_ini.ckpt'  # Pretrained model checkpoint path
batch_size = 4
logger_freq = 250  # Log progress every 100 steps
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First, use CPU to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()  # Replace with your dataset implementation
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

# Define a custom logger callback for logging steps
class StepLogger(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.global_step % logger_freq == 0:
            print(f"Step {trainer.global_step}/{trainer.max_steps}")

# Add ModelCheckpoint callback to save the model every 2500 steps
checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints',  # Directory to save checkpoints
    filename='control_sd21-step={step}-loss={loss:.2f}',  # Filename template
    save_top_k=-1,  # Save all checkpoints
    every_n_train_steps=2500  # Save every 2500 steps
)

# Trainer with StepLogger and ModelCheckpoint
trainer = pl.Trainer(
    gpus=1,
    precision=32,
    callbacks=[ImageLogger(batch_frequency=logger_freq), StepLogger(), checkpoint_callback],
    max_steps=20000  # Set a maximum number of steps if needed
)

# Train!
trainer.fit(model, dataloader)

