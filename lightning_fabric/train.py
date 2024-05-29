import torch
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np
from axonn import axonn as ax
from torch.cuda.amp import GradScaler
import random

from lightning.fabric import Fabric
from axonn_strategy import AxonnStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.fc_net_easy_tensor_parallel import FC_Net
from utils import print_memory_stats, num_params, log_dist
from args import create_parser

NUM_EPOCHS=2
PRINT_EVERY=200

seed=123

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    ## Step 1 - Initialize AxoNN
    torch.distributed.init_process_group(backend='nccl')
    
    axonn_strat = AxonnStrategy(
            G_data=args.G_data,
            G_inter=1,
            G_intra_r=args.G_intra_r,
            G_intra_c=args.G_intra_c,
            G_intra_d=args.G_intra_d)

    fabric = Fabric(strategy=axonn_strat, accelerator="gpu", devices=2)
    fabric.launch()

    log_dist('initialized AxoNN', ranks=[0])

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    ## Step 2 - Create dataset with augmentations
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )


    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, drop_last=True, num_workers=1)

    ## Step 4 - Create Neural Network 
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10).cuda()
    params = num_params(net) / 1e9

    ## Step 5 - Create Optimizer 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    net, optimizer = fabric.setup(net, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    ## Step 6 - Create Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    ## Step 7 - Scales the loss prior to the backward pass to prevent underflow of gradients
    scaler = GradScaler()

    ## Step 8 - Train
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    log_dist(f"Model Params = {num_params(net)/1e9} B", [0])
    log_dist(f"Start Training with AxoNN's Intra-Layer Parallelism", [0])

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        iter_ = 0
        iter_times = []
        for img, label in train_loader:
            start_event.record()
            optimizer.zero_grad()

            #with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = net(img)
            iter_loss = loss_fn(logits, label)

            ## scaling loss before doing the backward pass
            fabric.backward(scaler.scale(iter_loss))

            ## unscale gradients and run optimizer
            scaler.step(optimizer)
            ## update loss scale 
            scaler.update()

            epoch_loss += iter_loss
            stop_event.record()
            torch.cuda.synchronize()
            iter_time = start_event.elapsed_time(stop_event)
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0:
                log_dist(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time/1000:.6f} s", [0])
            iter_ += 1
        print_memory_stats()
        log_dist(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times)/1000:.6f} s", [0])
        log_dist(f"End Training ...", [0])
