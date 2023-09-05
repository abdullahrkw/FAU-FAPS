from collections import OrderedDict

from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, confusion_matrix
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
    def __init__(self, backbone=None,
                        backbone_out=1000,
                        num_classes=None,
                        lr=1e-4,
                        output_activation=None,
                        loss_func=None,
                        views=None,
                        labels=None):
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.output_activation = output_activation
        self.loss_func = loss_func
        self.num_classes = num_classes
        self.views = views
        self.labels = labels
        self.late_fc = torch.nn.Sequential(OrderedDict([
          ('fc2', torch.nn.Linear(in_features=len(self.views)*backbone_out, out_features=1024)),
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

    def forward(self, *views):
        x_cat = None
        for view in views:
            x = self.backbone(view)
            x_cat = x if x_cat is None else torch.cat((x_cat, x), dim=1)
        x = self.late_fc(x_cat)
        return x

    def training_step(self, train_batch, batch_idx):
        views = []
        for view in self.views:
            x = train_batch[view]
            views.append(x)
        target = train_batch["label"]
        # print(torch.bincount(torch.argmax(target, dim=1)))
        pred = self(*views)
        pred = self.output_activation(pred)
        loss = self.loss_func(pred, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        views = []
        for view in self.views:
            x = val_batch[view]
            views.append(x)
        target = val_batch["label"]

        pred = self(*views)
        pred = self.output_activation(pred)
        loss = self.loss_func(pred, target)
        self.log("val_loss", loss)
        target = target.cpu().argmax(axis=1).numpy()
        pred = pred.cpu().argmax(axis=1).numpy()
        accuracy = accuracy_score(target, pred)
        # cm = confusion_matrix(target, pred)
        # tp, fn, fp, tn = cm.ravel()
        # self.log("val_true_pos", torch.tensor(tp, dtype=torch.float32), on_epoch=True, reduce_fx="sum")
        # self.log("val_false_neg", torch.tensor(fn, dtype=torch.float32), on_epoch=True, reduce_fx="sum")
        self.log("val_acc", accuracy, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        views = []
        for view in self.views:
            x = test_batch[view]
            views.append(x)
        target = test_batch["label"]

        pred = self(*views)
        pred = self.output_activation(pred)
        loss = self.loss_func(pred, target)
        self.log("test_loss", loss)
        target = target.cpu().argmax(axis=1).numpy()
        pred = pred.cpu().argmax(axis=1).numpy()
        accuracy = accuracy_score(target, pred)
        self.log("test_acc", accuracy)
        return loss

    def predict_step(self, batch, batch_idx):
        views = []
        for view in self.views:
            x = batch[view]
            views.append(x)

        y = self(*views)

        return y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }      
