import torch
import torchsummary
import torch.nn as nn
import torchvision.models


class cnn_model(nn.Module):
    def __init__(self, in_features, out_features, size, backbone="resnet"):
        super().__init__()
        self.backbone = backbone_model(backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.in_features = in_features
        self.out_features = out_features

        self.size = size // 32
        # 2048
        if backbone == "resnet":
            self.first_size = 2048
        self.layer_1 = self.conv_block(self.first_size)
        self.layer_2 = self.conv_block(self.first_size//2)
        self.layer_3 = self.conv_block(self.first_size//4)

        self.classifier_layer = self.classifier(self.out_features)
    def forward(self, x):
        print(x.shape)
        x = self.backbone(x)
        res = x.clone()
        print(x.shape,res.shape)

        x = self.layer_1(x)
        res = nn.Conv2d(2048,1024,kernel_size=1)(res)
        x = x + res
        res = x
        print(x.shape,res.shape)

        x = self.layer_2(x)
        res = nn.Conv2d(1024,512,kernel_size=1)(res)
        x = x + res
        res = x
        print(x.shape,res.shape)

        x = self.layer_3(x)
        res = nn.Conv2d(512,256,kernel_size=1)(res)
        x = x + res
        res = x
        print(x.shape,res.shape)

        x = self.classifier_layer(x)
        print(x.shape)
        return x

    def classifier(self, out):
        input_value = int(self.first_size//8 * self.size * self.size)
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_value, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out)
        )

    def conv_block(self, num):
        return nn.Sequential(
            nn.Conv2d(num, num // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num // 2),
            nn.Conv2d(num // 2, num//2 , kernel_size=3, padding=1),
        )


def backbone_model(name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (name == "resnet"):
        backbone = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT).to(device)
        return nn.Sequential(*list(backbone.children())[:-2])
    if (name == "mobile_net"):
        return 0
    if (name == "inception"):
        return 0

model = cnn_model(3,5,512)

torchsummary.summary(model=model,input_size=(3,512,512))

import Utils.utils_fn as utils
conv = nn.Conv2d(2038,1024,kernel_size=3,padding=1)
print(utils.convolution_calculation.conv2d_hw_calculation(256,256,conv,pool=None))