import torch.nn as nn
import torch
from axonn.intra_layer import Linear, drop, gather
from axonn import axonn as ax

class FC_Net(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super(FC_Net, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([FC_Net_Layer(hidden_size) for _ in range(num_layers)])
        self.clf = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        
    def forward(self, x, checkpoint_activations=False):
        x = x.view(x.shape[0], -1)
        x = self.embed(x)
        ## drop partitions x across the tensor parallel GPUs
        ## needs to be done before the first tensor parallel layer
        x = drop(x)
        for layer in self.layers:
            if not checkpoint_activations:
                x = layer(x)
            else:
                x = torch.utils.checkpoint.checkpoint(layer, x)
        ## gather recovers x from the tensor parallel GPUs
        ## needs to be done after the last tensor parallel layer
        x = gather(x)
        x = self.clf(x)
        return x


class FC_Net_Layer(nn.Module):
    def __init__(self, hidden_size):
        super(FC_Net_Layer, self).__init__()
        #self.norm = nn.LayerNorm(hidden_size)

        ## replace nn.Linear with Tensor Parallel Linear
        self.linear_1 = Linear(in_features=hidden_size, out_features=4 * hidden_size, scatter_input=False, gather_output=False)
        self.relu = nn.ReLU()
        ## replace nn.Linear with Tensor Parallel Linear
        ## every alternate layer needs to be 'transposed'
        self.linear_2 = Linear(in_features = 4 * hidden_size, out_features = hidden_size, transpose=True, scatter_input=False, gather_output=False)

    def forward(self, x):
        #h = self.norm(x)
        h = self.linear_1(x)
        h = self.relu(h)
        h = self.linear_2(h)
        return h + x

if __name__ == "__main__":
    net = FC_Net(num_layers=2, input_size=256, hidden_size=1024, output_size=10).cuda()
    x = torch.rand(64, 256).cuda()
    y = net(x)
    print(y.shape, y.device)
