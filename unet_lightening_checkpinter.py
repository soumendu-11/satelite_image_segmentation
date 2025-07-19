import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Jaccard metric
def jaccard_coef(y_true, y_pred, smooth=1.0):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = (y_true * y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

# Double conv block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# UNet architecture
class UNet(nn.Module):
    def __init__(self, n_classes=4, in_channels=1):
        super(UNet, self).__init__()
        self.n_classes = n_classes

        self.c1 = DoubleConv(in_channels, 16)
        self.p1 = nn.MaxPool2d(2)

        self.c2 = DoubleConv(16, 32)
        self.p2 = nn.MaxPool2d(2)

        self.c3 = DoubleConv(32, 64)
        self.p3 = nn.MaxPool2d(2)

        self.c4 = DoubleConv(64, 128)
        self.p4 = nn.MaxPool2d(2)

        self.c5 = DoubleConv(128, 256, dropout=0.3)

        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c6 = DoubleConv(256, 128)

        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c7 = DoubleConv(128, 64)

        self.up8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c8 = DoubleConv(64, 32)

        self.up9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c9 = DoubleConv(32, 16)

        self.out = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        c5 = self.c5(p4)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        return self.out(c9)  # raw logits

# LightningModule for UNet

###############################################
# WITHOUT CHECKPOINTER 
# class LitUNet(pl.LightningModule):
#     def __init__(self, n_classes=4, in_channels=1, lr=1e-3):
#         super().__init__()
#         self.model = UNet(n_classes=n_classes, in_channels=in_channels)
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.lr = lr

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss_fn(logits, y)
#         preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
#         jaccard = jaccard_coef(y.float(), preds.float())

#         self.log("train_loss", loss)
#         self.log("train_jaccard", jaccard)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss_fn(logits, y)
#         preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
#         jaccard = jaccard_coef(y.float(), preds.float())

#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_jaccard", jaccard, prog_bar=True)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.lr)


###############################################
# WITH CHECKPOINTER 
###############################################

class LitUNet(pl.LightningModule):
    def __init__(self, n_classes=4, in_channels=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  # saves n_classes, in_channels, lr automatically
        self.model = UNet(in_channels=self.hparams.in_channels, n_classes=self.hparams.n_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()  # define your loss function here

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        jaccard = jaccard_coef(y.float(), preds.float())

        self.log("train_loss", loss)
        self.log("train_jaccard", jaccard)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        jaccard = jaccard_coef(y.float(), preds.float())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_jaccard", jaccard, prog_bar=True)

    def configure_optimizers(self):
        # Use the learning rate stored in hyperparameters
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

