import torch
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np
from axonn import axonn as ax
import matplotlib.pyplot as plt
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.conv_net_tensor_parallel import ConvNet
from utils import print_memory_stats, num_params, log_dist
from args import create_parser

NUM_EPOCHS=10000
PRINT_EVERY=200

torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# This is required because TF32 cores only look at the first 10 bits of mantissa
torch.backends.cudnn.allow_tf32 = False

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
    ## Step 1 - Initialize AxoNN
    ax.init(
                G_data=args.G_data,
                G_inter=1,
                G_intra_r=args.G_intra_r,
                G_intra_c=args.G_intra_c,
                G_intra_d=args.G_intra_d,
                mixed_precision=True,
                fp16_allreduce=True,
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

    ## Step 2 - Create dataset with augmentations
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )

    ## Step 3 - Create dataloader using AxoNN
    train_loader = ax.create_dataloader(
        train_dataset,
        args.batch_size,
        args.micro_batch_size,
        num_workers=1,
    )

    ## Step 4 - Create Neural Network 
    net = ConvNet(args.image_size, 10).cuda()
    #params = num_params(net) / 1e9 
    #for param in net.parameters():
    #    torch.nn.init.constant_(param, 0.01)

    
    ## Step 5 - Create Optimizer 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    ## Step 6 - register model and optimizer with AxoNN
    ## This creates the required data structures for 
    ## mixed precision
    net, optimizer = ax.register_model_and_optimizer(net, optimizer)

    ## Step 7 - Create Loss Function and register it
    loss_fn = torch.nn.CrossEntropyLoss()
    ax.register_loss_fn(loss_fn)

    ## Step 8 - Train
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
  
    log_dist(f"Model Params = {num_params(net)*ax.config.G_intra/1e9} B", [0])
    log_dist(f"Start Training with AxoNN's Intra-Layer Parallelism", [0])

    epoch_losses = []
    iter_losses = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        iter_ = 0
        iter_times = []
        for img, label in train_loader:
            start_event.record()
            optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()
            iter_loss = ax.run_batch(img, label)
            optimizer.step()

            epoch_loss += iter_loss

            stop_event.record()
            torch.cuda.synchronize()
            iter_losses.append(iter_loss)
            iter_time = start_event.elapsed_time(stop_event)
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0:
                log_dist(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time/1000:.6f} s", [0])
            iter_ += 1
        print_memory_stats()
        epoch_losses.append(epoch_loss)
        log_dist(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times)/1000:.6f} s", [0])


    log_dist(f"End Training ...", [0])

    if dist.is_initialized():
        if dist.get_rank() == 0:

            epochs = list(range(1, len(epoch_losses) + 1))
    
            # Plot the loss curve
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, epoch_losses, marker='.', linestyle='-')
            plt.title('Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('EpochLoss' + str(args.G_intra_r) + 'x' + str(args.G_intra_c) + 'x' + str(args.G_intra_d) + '.png')
            plt.show()

            iterations = list(range(1, len(iter_losses) + 1))
    
            # Plot the loss curve
            plt.figure(figsize=(8, 6))
            plt.plot(iterations, iter_losses, marker='.', linestyle='-')
            plt.title('Loss Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('IterLoss' + str(args.G_intra_r) + 'x' + str(args.G_intra_c) + 'x' + str(args.G_intra_d) + '.png')
            plt.show()

            with open('EpochLoss' + str(args.G_intra_r) + 'x' + str(args.G_intra_c) + 'x' + str(args.G_intra_d) + '.txt', 'w') as f:
                f.write("\n".join(map(str, epoch_losses)))

            with open('IterLoss' + str(args.G_intra_r) + 'x' + str(args.G_intra_c) + 'x' + str(args.G_intra_d) + '.txt', 'w') as f:
                f.write("\n".join(map(str, iter_losses)))

            





