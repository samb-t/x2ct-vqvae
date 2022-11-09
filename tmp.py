import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm

def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32 if 32 < in_channels else 1, num_channels=in_channels, eps=1e-6, affine=True)

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = LayerNorm(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm2 = LayerNorm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.conv_out = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        
        return x + x_in

    
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels = in_channels if out_channels is None else out_channels

        self.conv_in = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.norm1 = LayerNorm(in_channels)
        self.norm2 = LayerNorm(out_channels)

        self.branch1 = nn.Sequential(
            nn.Conv3d(out_channels // 3, out_channels // 3, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.Conv3d(out_channels // 3, out_channels // 3, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0))
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(out_channels // 3, out_channels // 3, kernel_size=(3, 3, 1), stride=1, padding=(1,1,0)),
            nn.Conv3d(out_channels // 3, out_channels // 3, kernel_size=(1, 1, 3), stride=1, padding=(0,0,1))
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(out_channels // 3, out_channels // 3, kernel_size=(3, 1, 3), stride=1, padding=(1,0,1)),
            nn.Conv3d(out_channels // 3, out_channels // 3, kernel_size=(1, 3, 1), stride=1, padding=(0,1,0))
        )

        self.conv_out = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv_in(x)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3 = self.branch3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.norm2(x)
        x = F.gelu(x)

        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        
        return x + x_in


def test_block(x, block):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = block(x)
    torch.cuda.synchronize()
    return time.time() - start_time

def repeat_test(num_iters, test_fn, *args, **kwargs):
    total_time = 0
    for _ in tqdm(range(num_iters)):
        total_time += test_fn(*args, **kwargs)
    return total_time / num_iters

device = 'cuda'
num_iters = 100
x = torch.randn(4, 32, 128, 128, 128, device=device)
resblock = ResBlock(32).to(device)
depthwise_block = DepthwiseBlock(32).to(device)
separable_block = SeparableConvBlock(30).to(device)

print(f"Res Block: {repeat_test(num_iters, test_block, x, resblock)}")
print(f"Depthwise Block: {repeat_test(num_iters, test_block, x, depthwise_block)}")
x = torch.randn(4, 30, 128, 128, 128, device=device)
print(f"Separable Block: {repeat_test(num_iters, test_block, x, separable_block)}")

x = torch.randn(4, 16, 128, 128, 128, device=device)
resblock = ResBlock(16).to(device)
depthwise_block = DepthwiseBlock(16).to(device)
print(f"Res Block - 16: {repeat_test(num_iters, test_block, x, resblock)}")
print(f"Depthwise Block - 16: {repeat_test(num_iters, test_block, x, depthwise_block)}")