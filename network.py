from collections import OrderedDict

from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
import torch.optim

from torchvision.models import resnet18

class ResNet(LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.resnet = model
        self.lr = lr

    def forward(self, x):
        x = self.resnet(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x = train_batch["view"]
        target = train_batch["label"]

        pred = self(x)
        loss = F.binary_cross_entropy(torch.sigmoid(pred), target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["view"]
        target = val_batch["label"]
        pred = self(x)
        loss = F.binary_cross_entropy(torch.sigmoid(pred), target)
        self.log("val_loss", loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x = test_batch["view"]
        target = test_batch["label"]
        pred = self(x)
        loss = F.binary_cross_entropy(torch.sigmoid(pred), target)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["view"]
        y = self(x)
        return y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LateFusionNetwork(LightningModule):
    def __init__(self, backbone=None, backbone_out=1000, num_classes=None, lr=1e-4, output_activation=None, loss_func=None):
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.output_activation = output_activation
        self.loss_func = loss_func
        self.num_classes = num_classes
        self.late_fc = torch.nn.Sequential(OrderedDict([
          ('fc2', torch.nn.Linear(in_features=2*backbone_out, out_features=1024)),
          ('relu2', torch.nn.ReLU()),
          ('bn2', torch.nn.BatchNorm1d(num_features=1024)),
          ('dropout2', torch.nn.Dropout(p=0.5, inplace=False)),
          ('fc3', torch.nn.Linear(in_features=1024, out_features=512)),
          ('relu3', torch.nn.ReLU()),
          ('bn3', torch.nn.BatchNorm1d(num_features=512)),
          ('dropout3', torch.nn.Dropout(p=0.5, inplace=False)),
          ('fc4', torch.nn.Linear(in_features=512,out_features=256)),
          ('relu4', torch.nn.ReLU()),
          ('bn4', torch.nn.BatchNorm1d(num_features=256)),
          ('dropout4', torch.nn.Dropout(p=0.5, inplace=False)),
          ('fc5', torch.nn.Linear(in_features=256,out_features=num_classes)),
        ]))

    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x_cat = torch.cat((x1, x2), dim=1)
        x = self.late_fc(x_cat)
        return x

    def training_step(self, train_batch, batch_idx):
        x1 = train_batch["view1"]
        x2 = train_batch["view2"]
        target = train_batch["label"]

        pred = self(x1, x2)
        pred = self.output_activation(pred)
        loss = self.loss_func(pred, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x1 = val_batch["view1"]
        x2 = val_batch["view2"]
        target = val_batch["label"]

        pred = self(x1, x2)
        pred = self.output_activation(pred)
        loss = self.loss_func(pred, target)
        self.log("val_loss", loss)
        target = target.cpu().argmax(axis=1).numpy()
        pred = pred.cpu().argmax(axis=1).numpy()
        accuracy = accuracy_score(target, pred)
        self.log("val_acc", accuracy)
        return loss

    def test_step(self, test_batch, batch_idx):
        x1 = test_batch["view1"]
        x2 = test_batch["view2"]
        target = test_batch["label"]

        pred = self(x1, x2)
        pred = self.output_activation(pred)
        loss = self.loss_func(pred, target)
        self.log("test_loss", loss)
        target = target.cpu().argmax(axis=1).numpy()
        pred = pred.cpu().argmax(axis=1).numpy()
        accuracy = accuracy_score(target, pred)
        self.log("test_acc", accuracy)
        return loss

    def predict_step(self, batch, batch_idx):
        x1 = batch["view1"]
        x2 = batch["view2"]

        y = self(x1, x2)

        return y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
