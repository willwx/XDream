import torch
from torch import nn


class DeePSiMFc6(nn.Module):
    def __init__(self):
        super().__init__()
        # reusable activation funcs
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        # layers
        self.defc7 = nn.Linear(4096, 4096)
        self.defc6 = nn.Linear(4096, 4096)
        self.defc5 = nn.Linear(4096, 4096)
        self.deconv5_0 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.deconv5_1 = nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1)
        self.deconv4_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.deconv4_1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1)
        self.deconv3_0 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv3_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv0 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
        
    def forward(self, x):
        p = self.parameters().__iter__().__next__()
        x = torch.tensor(x, device=p.device, dtype=p.dtype)
        lrelu = self.lrelu
        x = lrelu(self.defc7(x))
        x = lrelu(self.defc6(x))
        x = lrelu(self.defc5(x))
        x = x.view(-1, 256, 4, 4)
        x = lrelu(self.deconv5_0(x))
        x = lrelu(self.deconv5_1(x))
        x = lrelu(self.deconv4_0(x))
        x = lrelu(self.deconv4_1(x))
        x = lrelu(self.deconv3_0(x))
        x = lrelu(self.deconv3_1(x))
        x = lrelu(self.deconv2(x))
        x = lrelu(self.deconv1(x))
        x = self.deconv0(x)
        return x


get_net = {
    'deepsim-fc6': DeePSiMFc6
}


def load_net(net_name):
    return get_net[net_name]()
