import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import kornia.geometry.transform as kt

class PerspectiveSTNPerImage(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_shape[0], 12, kernel_size=3),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True),
            nn.Conv2d(12, 16, kernel_size=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        with torch.no_grad():
            x = torch.randn((1, *in_shape))
            loc_out_shape = self.localization(x).numel()
        
        self.fc_loc = nn.Sequential(
            nn.Linear(loc_out_shape, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 3)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.eye(3, dtype=torch.float).flatten())

    def forward(self, x, return_theta=False):
        xs = self.localization(x)
        # xs = xs.view(-1, 10 * 3 * 3)
        xs = xs.view(x.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 3)
        x = kt.warp_perspective(x, theta, dsize=x.shape[-2:])
        # grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)
        if return_theta:
            return x, theta
        else:
            return x

class PerspectiveSTN(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_shape[0], 12, kernel_size=3),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True),
            nn.Conv2d(12, 16, kernel_size=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        with torch.no_grad():
            x = torch.randn((1, *in_shape))
            loc_out_shape = self.localization(x).numel()
        
        self.fc_loc = nn.Sequential(
            nn.Linear(loc_out_shape, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 3)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.eye(3, dtype=torch.float).flatten())

    def forward(self, x):
        xs = self.localization(x)
        # xs = xs.view(-1, 10 * 3 * 3)
        xs = xs.view(x.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 3)
        x = kt.warp_perspective(x, theta, dsize=x.shape[-2:])
        # grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)
        return x

class TransformNet_STN_PerImage(nn.Module):
    def __init__(self, num_channels):
        super(TransformNet_STN_PerImage, self).__init__()
        self.num_channals = num_channels
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, return_theta=False):
        x = x.view(-1, 3, x.size()[-2], x.size()[-1])
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        if return_theta:
            return x.view(-1, self.num_channals, 84, 84), theta
        else:
            return x.view(-1, self.num_channals, 84, 84)
    
class TransformNet_STN(nn.Module):
    def __init__(self, num_channels):
        super(TransformNet_STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=7),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class TransformNet_STN1(nn.Module):
    def __init__(self, in_shape):
        super(TransformNet_STN1, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_shape[0], 8, kernel_size=7),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        with torch.no_grad():
            x = torch.randn((1, *in_shape))
            loc_out_shape = self.localization(x).numel()  # 160
        
        self.fc_loc = nn.Sequential(
            nn.Linear(loc_out_shape, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        # xs = xs.view(-1, 10 * 3 * 3)
        xs = xs.view(x.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class TransformNet_STN2(nn.Module):
    def __init__(self, num_channels):
        super(TransformNet_STN2, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
    
class TransformNet(nn.Module):
    def __init__(self, lr=0.01):
        super(TransformNet, self).__init__()
        self.lr = lr
        self.theta = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]).detach().cuda()
        self.theta.requires_grad = True
        self.optimizer = optim.Adam([self.theta], lr=self.lr)

    def forward(self, x):
        grid = F.affine_grid(self.theta.expand(x.size()[0], -1, -1), x.size())
        x = F.grid_sample(x, grid)
        return x
    
    def to_device(self, device):
        self.theta = self.theta.to(device).detach()
        self.theta.requires_grad = True
        self.optimizer = optim.Adam([self.theta], lr=self.lr)
