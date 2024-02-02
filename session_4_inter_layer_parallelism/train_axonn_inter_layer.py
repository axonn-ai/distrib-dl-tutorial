import torch
import torchvision
import sys
import time
import os
from torchvision import transforms
import numpy as np
from axonn import axonn as ax

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from model.fc_net_pipeline_parallel import FC_Net
from utils import print_memory_stats, num_params, log_dist
from args import create_parser

NUM_EPOCHS=2
PRINT_EVERY=200

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
    ## Step 1 - Initialize Pytorch Distributed
    ax.init(
                G_data=args.G_data,
                G_inter=args.G_inter,
                G_intra_r=1,
                G_intra_c=1,
                mixed_precision=False,
                fp16_allreduce=False,
                cpu_offload=False
            )
    
    log_dist('initialized AxoNN for CPU execution', ranks=[0])

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    ## Step 2 - Create Dataloaders with sampler
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )

    train_loader = ax.create_dataloader(
        train_dataset,
        args.batch_size,
        args.micro_batch_size,
        num_workers=1,
    )

    ## Step 3 - Create Neural Network 
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10)
    params = num_params(net) / 1e9 
    
    ## Step 4 - Create Optimizer 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    ## Step 4 - register model and optimizer with AxoNN
    net, optimizer = ax.register_model_and_optimizer(net, optimizer)

    ## Step 6 - Create Loss Function and register it
    loss_fn = torch.nn.CrossEntropyLoss()
    ax.register_loss_fn(loss_fn)

    ## Step 7 - Train
    log_dist(f"Model Params = {num_params(net)*ax.config.G_inter/1e9} B", [0])

    if args.G_data == 1:
        log_dist(f"Begin Training with AxoNN's Inter-Layer Parallelism ... \n", [0])
    else:
        log_dist(f"Begin Training with AxoNN's Hybrid Inter-Layer+Data Parallelism ... \n", [0])

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        iter_ = 0
        iter_times = []
        for img, label in train_loader:
            start_time = time.time()
            optimizer.zero_grad()
            iter_loss = ax.run_batch(img, label)
            optimizer.step()
            
            epoch_loss += iter_loss
            end_time = time.time()
            iter_time = end_time - start_time
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0 and ax.config.inter_layer_parallel_rank == ax.config.G_inter-1 and ax.config.data_parallel_rank == 0:
                ax.print_status(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time:.6f} s")
                print_memory_stats()
            iter_ += 1
        if ax.config.inter_layer_parallel_rank == ax.config.G_inter-1 and ax.config.data_parallel_rank == 0:
            ax.print_status(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times):.6f} s")
        
    log_dist(f"\n End Training ...", [0])
