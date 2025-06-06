from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="./ControlNet/lightning_logs/version_0/checkpoints",
    filename="model-{epoch:02d}-{step}",
    save_last=True,
    every_n_train_steps=300
)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
torch.cuda.empty_cache()
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=32, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(devices=1, accelerator="gpu", precision="32", callbacks=[logger, checkpoint_callback], max_epochs=3)


# Train!
trainer.fit(model, dataloader) #Use this to train the model from the start

#Use the below command if you want to resume training form a checkpoint
# trainer.fit(model, dataloader, ckpt_path="./path/to/checkpoint.ckpt") #Update ckpt_path
