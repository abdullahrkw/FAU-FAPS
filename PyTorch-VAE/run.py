import os
import yaml
import argparse
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.strategies import DDPStrategy
from sklearn.metrics import confusion_matrix, classification_report


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
print(model)
experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=False)

data.setup()
runner = Trainer(logger=tb_logger,
                 log_every_n_steps=20,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                #  strategy=DDPStrategy(find_unused_parameters=False),
                 accelerator="auto",
                 devices=1 if torch.cuda.is_available() else None,
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
print(f"======= Testing {config['model_params']['name']} =======")
runner.test(dataloaders=data.test_dataloader(), ckpt_path="last")
print(f"======= Predicting {config['model_params']['name']} =======")

experiment.eval()  # Set the model to evaluation mode
preds = runner.predict(experiment, dataloaders=data.val_dataloader(), ckpt_path="last")

batch_size = preds[0][0].shape[0]
labels_list = []
recon_error_list = []
for idx, pred  in enumerate(preds):
    recon_error, labels = pred
    labels = labels.data.cpu().numpy()
    recon_error = recon_error.data.cpu().numpy()
    labels = np.argmax(labels, axis=-1)
    labels_list = np.append(labels_list, labels)
    recon_error_list = np.append(recon_error_list, recon_error)
plt.scatter(range(len(labels_list)), recon_error_list, c=labels_list)
plt.colorbar()
error_class = (1-labels_list)*recon_error_list
# threshold = np.unique(np.sort(error_class))[-2]
threshold = np.mean(np.unique(np.sort(error_class))[-10:-1])
plt.hlines(threshold, 0, len(labels_list), label=f"Threshold = {threshold:.2f}")
plt.legend()
plt.savefig(f"recon_error_{config['data_params']['problem']}_{config['trainer_params']['max_epochs']}_val.png")

# Confusion metrics
print("Validation Dataset Matrics")
recon_error_list[recon_error_list > threshold] = 1
recon_error_list[recon_error_list <= threshold] = 0
recon_error_list = recon_error_list.astype(np.int8)
print(np.unique(recon_error_list))
cm = confusion_matrix(labels_list, recon_error_list)
print(cm)
cr = classification_report(labels_list, recon_error_list)
print(cr)
plt.clf()
# Test Dataset , but use threshold from validation dataset
preds = runner.predict(experiment, dataloaders=data.test_dataloader(), ckpt_path="last")

batch_size = preds[0][0].shape[0]
labels_list = []
recon_error_list = []
for idx, pred  in enumerate(preds):
    recon_error, labels = pred
    labels = labels.data.cpu().numpy()
    recon_error = recon_error.data.cpu().numpy()
    labels = np.argmax(labels, axis=-1)
    labels_list = np.append(labels_list, labels)
    recon_error_list = np.append(recon_error_list, recon_error)
plt.scatter(range(len(labels_list)), recon_error_list, c=labels_list)
plt.colorbar()
error_class = (1-labels_list)*recon_error_list
# threshold = np.unique(np.sort(error_class))[-2]
plt.hlines(threshold, 0, len(labels_list), label=f"Threshold = {threshold:.2f}")
plt.legend()
plt.savefig(f"recon_error_{config['data_params']['problem']}_{config['trainer_params']['max_epochs']}_test.png")

# Confusion metrics
print("Test Dataset Matrics")
recon_error_list[recon_error_list > threshold] = 1
recon_error_list[recon_error_list <= threshold] = 0
recon_error_list = recon_error_list.astype(np.int8)
print(np.unique(recon_error_list))
cm = confusion_matrix(labels_list, recon_error_list)
print(cm)
cr = classification_report(labels_list, recon_error_list)
print(cr)