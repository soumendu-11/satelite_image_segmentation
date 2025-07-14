# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def jaccard_coef(y_true, y_pred, smooth=1.0):
#     y_true = y_true.view(-1)
#     y_pred = y_pred.view(-1)

#     intersection = (y_true * y_pred).sum()
#     total = y_true.sum() + y_pred.sum()

#     union = total - intersection

#     return (intersection + smooth) / (union + smooth)


# # Double convolution block: Conv2d -> ReLU -> Conv2d -> ReLU
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# # UNet model
# class UNet(nn.Module):
#     def __init__(self, n_classes, in_channels=3):
#         super(UNet, self).__init__()

#         # Encoder
#         self.enc1 = DoubleConv(in_channels, 64)
#         self.pool1 = nn.MaxPool2d(2)

#         self.enc2 = DoubleConv(64, 128)
#         self.pool2 = nn.MaxPool2d(2)

#         self.enc3 = DoubleConv(128, 256)
#         self.pool3 = nn.MaxPool2d(2)

#         self.enc4 = DoubleConv(256, 512)
#         self.pool4 = nn.MaxPool2d(2)

#         # Bottleneck
#         self.bottleneck = DoubleConv(512, 1024)

#         # Decoder
#         self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.dec4 = DoubleConv(1024, 512)

#         self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec3 = DoubleConv(512, 256)

#         self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec2 = DoubleConv(256, 128)

#         self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec1 = DoubleConv(128, 64)

#         # Final layer - CAM-relevant
#         self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool1(enc1))
#         enc3 = self.enc3(self.pool2(enc2))
#         enc4 = self.enc4(self.pool3(enc3))

#         # Bottleneck
#         bottleneck = self.bottleneck(self.pool4(enc4))

#         # Decoder
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((enc4, dec4), dim=1)
#         dec4 = self.dec4(dec4)

#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((enc3, dec3), dim=1)
#         dec3 = self.dec3(dec3)

#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((enc2, dec2), dim=1)
#         dec2 = self.dec2(dec2)

#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((enc1, dec1), dim=1)
#         dec1 = self.dec1(dec1)

#         out = self.final_conv(dec1)
#         return out



import torch
import torch.nn as nn
import torch.nn.functional as F

def jaccard_coef(y_true, y_pred, smooth=1.0):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    intersection = (y_true * y_pred).sum()
    total = y_true.sum() + y_pred.sum()

    union = total - intersection

    return (intersection + smooth) / (union + smooth)

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

class UNet(nn.Module):
    def __init__(self, n_classes, in_channels=3):
        super(UNet, self).__init__()

        # Encoder
        self.c1 = DoubleConv(in_channels, 16)
        self.p1 = nn.MaxPool2d(2)

        self.c2 = DoubleConv(16, 32)
        self.p2 = nn.MaxPool2d(2)

        self.c3 = DoubleConv(32, 64)
        self.p3 = nn.MaxPool2d(2)

        self.c4 = DoubleConv(64, 128)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.c5 = DoubleConv(128, 256, dropout=0.3)

        # Decoder
        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c6 = DoubleConv(256, 128)

        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c7 = DoubleConv(128, 64)

        self.up8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c8 = DoubleConv(64, 32)

        self.up9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c9 = DoubleConv(32, 16)

        # Output
        self.out = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        # Bottleneck
        c5 = self.c5(p4)

        # Decoder
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

        # Output with softmax activation along class dim (dim=1)
        return F.softmax(self.out(c9), dim=1)

# Example usage:
# model = UNet(n_classes=6, in_channels=3)
