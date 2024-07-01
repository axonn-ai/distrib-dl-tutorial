import torch
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np
from axonn import axonn as ax
from axonn.inter_layer import AxoNN_Inter_Layer_Engine

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
            )
    
    log_dist('initialized AxoNN', ranks=[0])

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
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10).cuda()
    params = num_params(net) / 1e9 
    
    ## Step 4 - Create Optimizer 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    ## Step 5 - Create Loss Function 
    loss_fn = torch.nn.CrossEntropyLoss()

    ## Step 6 - Declare PipeEngine
    engine = AxoNN_Inter_Layer_Engine(net, loss_fn, computation_dtype=torch.bfloat16)

    ## Step 7 - Train
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
  
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
            start_event.record()
            optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()
            iter_loss = engine.forward_backward_optimizer(img, label, optimizer)
            epoch_loss += iter_loss
            stop_event.record()
            torch.cuda.synchronize()
            iter_time = start_event.elapsed_time(stop_event)
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0 and ax.config.inter_layer_parallel_rank == ax.config.G_inter-1 and ax.config.data_parallel_rank == 0:
                print(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time/1000:.6f} s")
                print_memory_stats()
            iter_ += 1
        if ax.config.inter_layer_parallel_rank == ax.config.G_inter-1 and ax.config.data_parallel_rank == 0:
            print(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times)/1000:.6f} s")
        
    log_dist(f"\n End Training ...", [0])
