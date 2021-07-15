import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import inspect
import sys
import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model_tools import GradualWarmupScheduler
from src.criterions import CXE

# from src.mobilenetV2 import MobileNetV2
# from src.vgg import VGG16


def get_model(name, params, n_epochs):
    """Looks for a model class with the specified name, instantiates it with the
    provided parameters and defines the loss and the optimizer.

    Args:
        name (str): name of the model, defined as a class with that same name in the
        src.models module.
        params (dict): dictionary of parameters to pass to the model when instantiated.
        n_epochs(int): number of epochs to run. Used to adjust the lr_scheduler

    Raises:
        ModuleNotFoundError: if the model does not exist, this exception is raised.

    Returns:
        tuple: model, loss function, optimizer and lr_scheduler.
    """
    # Find the requested model by name
    cls_members = dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))
    if name not in cls_members:
        raise ModuleNotFoundError(f"Class {name} not found in module {__name__}")
    model_class = cls_members[name]

    # Instantiate the model
    net = model_class(**params)

    # Define the loss and the optimizer
    criterion = CXE

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    # lr_scheduler = CosineAnnealingLR(
    #    optimizer, T_max=n_epochs + 1, eta_min=1e-6, last_epoch=-1, verbose=False
    # )
    # lr_scheduler = GradualWarmupScheduler(
    #    optimizer, multiplier=10, total_epoch=5, after_scheduler=lr_scheduler
    # )
    lr_scheduler = None
    return net, criterion, optimizer, lr_scheduler


class ModelProto(nn.Module):
    def __init__(self):
        super(ModelProto, self).__init__()

    def reset_masks(self):
        self.masks = self.get_dropout_masks(self.sizes, self.p, self.device)

    def invert_masks(self):
        self.masks = [m.logical_not() for m in self.masks]

    @staticmethod
    def get_dropout_masks(layer_sizes, p, device):
        masks = []
        for size in layer_sizes:
            masks.append(torch.rand(size=[size]).to(device) < p)
        return masks

    def mask_conv(self, h, idx):
        return h * self.masks[idx].view(1, -1, 1, 1) / self.masks[idx].float().mean()

    def mask_fc(self, h, idx):
        return h * self.masks[idx].view(1, -1) / self.masks[idx].float().mean()


class Conv6(ModelProto):
    def __init__(self, n_outputs, input_size, input_channels, p, device):
        super(Conv6, self).__init__()
        self.p = p  # Keep prob
        self.device = device
        self.n_outputs = n_outputs
        self.sizes = [128, 128, 256, 256, 512, 512, 512, 512]
        self.activation = F.relu
        self.flatten_img_dim = int(input_size / 2 / 2 / 2)
        self.pool = nn.MaxPool2d(2, 2)
        # Block conv 1
        self.conv1_1 = nn.Conv2d(input_channels, self.sizes[0], 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(self.sizes[0], self.sizes[1], 3, padding=(1, 1))
        # Block conv 2
        self.conv2_1 = nn.Conv2d(self.sizes[1], self.sizes[2], 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(self.sizes[2], self.sizes[3], 3, padding=(1, 1))
        # Block conv 3
        self.conv3_1 = nn.Conv2d(self.sizes[3], self.sizes[4], 3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(self.sizes[4], self.sizes[5], 3, padding=(1, 1))
        # FC top
        self.fc1 = nn.Linear(self.sizes[5] * self.flatten_img_dim ** 2, self.sizes[6])
        self.fc2 = nn.Linear(self.sizes[6], self.sizes[7])
        self.fc3 = nn.Linear(self.sizes[7], n_outputs)

    def forward(self, x, mask=False):
        # Block conv 1
        h = self.activation(self.conv1_1(x))
        h = self.mask_conv(h, 0) if mask else h
        h = self.pool(self.activation(self.conv1_2(h)))
        h = self.mask_conv(h, 1) if mask else h
        # Block conv 2
        h = self.activation(self.conv2_1(h))
        h = self.mask_conv(h, 2) if mask else h
        h = self.pool(self.activation(self.conv2_2(h)))
        h = self.mask_conv(h, 3) if mask else h
        # Block conv 3
        h = self.activation(self.conv3_1(h))
        h = self.mask_conv(h, 4) if mask else h
        h = self.pool(self.activation(self.conv3_2(h)))
        h = self.mask_conv(h, 5) if mask else h
        # Flatten
        h = h.view(-1, self.sizes[5] * self.flatten_img_dim ** 2)
        # FC top
        h = self.activation(self.fc1(h))
        h = self.mask_fc(h, 6) if mask else h
        h = self.activation(self.fc2(h))
        h = self.mask_fc(h, 7) if mask else h
        h = self.fc3(h)
        return h


class Conv4(ModelProto):
    def __init__(self, n_outputs, input_size, input_channels, p, device):
        super(Conv4, self).__init__()
        self.p = p  # Keep prob
        self.device = device
        self.n_outputs = n_outputs
        self.sizes = [128, 128, 256, 256, 256, 256]
        self.activation = F.relu
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_img_dim = int(input_size / 2 / 2)
        # Block conv 1
        self.conv1_1 = nn.Conv2d(input_channels, self.sizes[0], 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(self.sizes[0], self.sizes[1], 3, padding=(1, 1))
        # Block conv 2
        self.conv2_1 = nn.Conv2d(self.sizes[1], self.sizes[2], 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(self.sizes[2], self.sizes[3], 3, padding=(1, 1))
        # FC top
        self.fc1 = nn.Linear(self.sizes[3] * self.flatten_img_dim ** 2, self.sizes[4])
        self.fc2 = nn.Linear(self.sizes[4], self.sizes[5])
        self.fc3 = nn.Linear(self.sizes[5], n_outputs)

    def forward(self, x, mask):
        # Block conv 1
        h = self.activation(self.conv1_1(x))
        h = self.mask_conv(h, 0) if mask else h
        h = self.pool(self.activation(self.conv1_2(h)))
        h = self.mask_conv(h, 1) if mask else h
        # Block conv 2
        h = self.activation(self.conv2_1(h))
        h = self.mask_conv(h, 2) if mask else h
        h = self.pool(self.activation(self.conv2_2(h)))
        h = self.mask_conv(h, 3) if mask else h
        # Flatten
        h = h.view(-1, self.sizes[3] * self.flatten_img_dim ** 2)
        # FC top
        h = self.activation(self.fc1(h))
        h = self.mask_fc(h, 4) if mask else h
        h = self.activation(self.fc2(h))
        h = self.mask_fc(h, 5) if mask else h
        h = self.fc3(h)
        return h


class Conv2(ModelProto):
    def __init__(self, n_outputs, input_size, input_channels, p, device):
        super(Conv2, self).__init__()
        self.p = p  # Keep prob
        self.device = device
        self.n_outputs = n_outputs
        self.sizes = [128, 128, 256, 256]
        self.activation = F.relu
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_img_dim = int(input_size / 2)
        # Block conv 1
        self.conv1_1 = nn.Conv2d(input_channels, self.sizes[0], 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(self.sizes[0], self.sizes[1], 3, padding=(1, 1))
        # FC top
        self.fc1 = nn.Linear(self.sizes[1] * self.flatten_img_dim ** 2, self.sizes[2])
        self.fc2 = nn.Linear(self.sizes[2], self.sizes[3])
        self.fc3 = nn.Linear(self.sizes[3], n_outputs)

    def forward(self, x, mask):
        # Block conv 1
        h = self.activation(self.conv1_1(x))
        h = self.mask_conv(h, 0) if mask else h
        h = self.pool(self.activation(self.conv1_2(h)))
        h = self.mask_conv(h, 1) if mask else h
        # Flatten
        h = h.view(-1, self.sizes[1] * self.flatten_img_dim ** 2)
        # FC top
        h = self.activation(self.fc1(h))
        h = self.mask_fc(h, 2) if mask else h
        h = self.activation(self.fc2(h))
        h = self.mask_fc(h, 3) if mask else h
        h = self.fc3(h)
        return h


class FC(ModelProto):
    def __init__(self, n_outputs, input_size, input_channels, p, device):
        super(FC, self).__init__()
        self.p = p  # Keep prob
        self.device = device
        self.n_outputs = n_outputs
        self.sizes = [256, 256]
        self.activation = F.relu
        self.input_size = (input_size ** 2) * input_channels
        # MLP block
        self.fc1 = nn.Linear(self.input_size, self.sizes[0])
        self.fc2 = nn.Linear(self.sizes[0], self.sizes[1])
        self.fc3 = nn.Linear(self.sizes[1], n_outputs)

    def forward(self, x, mask):
        # Flatten
        h = x.view(x.shape[0], -1)
        # MLP Block
        h = self.activation(self.fc1(h))
        h = self.mask_fc(h, 0) if mask else h
        h = self.activation(self.fc2(h))
        h = self.mask_fc(h, 1) if mask else h
        h = self.fc3(h)
        return h

