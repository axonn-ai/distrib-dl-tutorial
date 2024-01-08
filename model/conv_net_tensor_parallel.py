import torch.nn as nn
import torch
from axonn.intra_layer import Tensor_Parallel_Conv2d, Linear, drop, gather
from axonn import axonn as ax

def init_method_normal():
    def init_(tensor):
        return torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu')

    return init_

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()
        self.embed1 = torch.nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 1)

        self.conv1 = ConvLayer(in_channels = [4, 8], out_channels = [8, 16], kernel_size = [5, 5])
        self.fc1 = FCNetLayer((input_size - 8) * (input_size - 8) * 16, 120, 120)
        self.clf = nn.Linear(120, output_size)
        
    def forward(self, x, checkpoint_activations=False):
        h = self.embed1(x)

        h = self.conv1(h, scatter_input = True)
        h = h.view(h.shape[0], -1)
        h = self.fc1(h, gather_output = True)
        #h = gather(h, dim=1)

        h = self.clf(h)
        return h

class FCNetLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNetLayer, self).__init__()
        #self.norm = nn.LayerNorm(hidden_size)

        ## replace nn.Linear with Tensor Parallel Linear
        self.linear_1 = Linear(in_features= input_size, out_features= hidden_size, init_method = init_method_normal())
        self.relu = nn.ReLU()
        ## replace nn.Linear with Tensor Parallel Linear
        ## every alternate layer needs to be 'transposed'
        self.linear_2 = Linear(in_features = hidden_size, out_features = output_size, transpose = True, init_method = init_method_normal())

    def forward(self, x, gather_output = False):
        #h = self.norm(x)
        #print (x.size())
        h = self.linear_1(x, scatter_input=False, gather_output=False)
        h = self.relu(h)
        h = self.linear_2(h, scatter_input=False, gather_output=gather_output)
        #print (h.size())
        return h

class ConvLayer(nn.Module):
    def __init__(self, in_channels : list[int], out_channels : list[int], kernel_size : list[int]):
        super(ConvLayer, self).__init__()

        self.conv2d_1 = Tensor_Parallel_Conv2d(in_channels = in_channels[0], out_channels = out_channels[0], kernel_size = kernel_size[0], init_method = init_method_normal()) 
        self.relu = nn.ReLU()

        ## every alternate layer needs to be 'transposed'
        self.conv2d_2 = Tensor_Parallel_Conv2d(in_channels = in_channels[1], out_channels = out_channels[1], kernel_size = kernel_size[1], transpose=True, init_method = init_method_normal())

    def forward(self, x, scatter_input = True):
        #print (x.size())
        #h = self.norm(x)
        h = self.conv2d_1(x, scatter_input=scatter_input, gather_output=False)
        h = self.relu(h)
        h = self.conv2d_2(h, scatter_input=False, gather_output=False)
        #print (h.size())
        return h


if __name__ == "__main__":
    net = ConvNet(input_size=64, output_size=10).cuda()
    x = torch.rand(64, 1, 64, 64).cuda()
    y = net(x)
    print(y.shape, y.device)
