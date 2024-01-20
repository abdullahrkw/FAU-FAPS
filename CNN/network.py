from collections import OrderedDict

from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
import torch
import torch.optim


class DeepCNN(LightningModule):
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
          ('fc2', torch.nn.Linear(in_features=len(self.views)*backbone_out, out_features=num_classes)),
        ]))

    def forward(self, view):
        x = self.backbone(view)
        x = self.late_fc(x)
        return x

    def training_step(self, train_batch, batch_idx):
        view = self.views[0]
        x = train_batch[view]
        target = train_batch["label"]
        pred = self(x)
        pred = self.output_activation(pred)
        loss = self.loss_func(pred, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        view = self.views[0]
        x = val_batch[view]
        target = val_batch["label"]
        pred = self(x)
        pred = self.output_activation(pred)
        loss = self.loss_func(pred, target)
        self.log("val_loss", loss)
        target = target.cpu().argmax(axis=1).numpy()
        pred = pred.cpu().argmax(axis=1).numpy()
        accuracy = accuracy_score(target, pred)
        self.log("val_acc", accuracy, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        view = self.views[0]
        x = test_batch[view]
        target = test_batch["label"]
        pred = self(x)
        pred = self.output_activation(pred)
        loss = self.loss_func(pred, target)
        self.log("test_loss", loss)
        target = target.cpu().argmax(axis=1).numpy()
        pred = pred.cpu().argmax(axis=1).numpy()
        accuracy = accuracy_score(target, pred)
        self.log("test_acc", accuracy)
        return loss

    def predict_step(self, batch, batch_idx):
        view = self.views[0]
        x = batch[view]
        y = self(x)

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
