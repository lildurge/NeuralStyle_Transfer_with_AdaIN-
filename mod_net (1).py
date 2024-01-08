import torch
import torch.nn as nn


class encoder_decoder:
    vgg_backend = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU()  # relu4-1, this is the last layer used
    )

    frontend = nn.Sequential(
        nn.Linear(512 * 8 * 8, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),  # Added layer for skip connection
        nn.ReLU(),
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Linear(256, 100)
    )


class ResidualBlock(nn.Module):
    def __init__(self, input_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_features, input_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.relu(out)
        out += residual  # Skip Connection
        out = self.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, num_classes=100):
        super(Model, self).__init__()
        self.backend = encoder_decoder.vgg_backend
        self.frontend = encoder_decoder.frontend

        # Replace a frontend layer with residual block
        self.res_block1 = ResidualBlock(4096)
        self.res_block2 = ResidualBlock(256)

        # Keep encoder weights fixed (if you want to)
        for param in self.backend.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, X):
        enc = self.backend(X)

        # Ensure the encoder output has the correct shape
        if enc.size(1) == 512 and enc.size(2) == 8 and enc.size(3) == 8:
            enc_flatten = enc.view(enc.size(0), -1)
            dec = self.frontend(enc_flatten)
            return dec
        else:
            raise ValueError(f"Expected encoder output shape [batch, 512, 8, 8], got {enc.size()}")
