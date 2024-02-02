import torch
import torchvision
import sys
import os
import time
from torchvision import transforms
import numpy as np
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.fc_net_sequential import FC_Net
from utils import print_memory_stats, num_params
from args import create_parser

NUM_EPOCHS=2
PRINT_EVERY=200

if __name__ == "__main__":
    data_json = {
        "iter_times": []
        "val_acc": [],
        "val_loss": [],
        "train_acc": [],
        "train_loss": []
    }
    parser = create_parser()
    args = parser.parse_args()
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    ## Step 1 - Create Dataloaders
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=args.batch_size, drop_last=True, num_workers=1)
    
    ## Step 2 - Create Neural Network 
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10)
    params = num_params(net) / 1e9 
    
    ## Step 3 - Create Optimizer and LR scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    ## Step 4 - Create Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()
   
    print("Start training on CPU ...\n")
    print(f"Model Size = {params} B")

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        total = 0
        correct = 0
        iter_ = 0
        iter_times = []
        for inputs, targets in train_loader:
            start_time = time.time()
            optimizer.zero_grad()

            output = net(inputs)
            iter_loss = loss_fn(output, targets)
            iter_loss.backward()
            optimizer.step()
            
            epoch_loss += iter_loss.item()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            end_time = time.time()
            iter_time = end_time - start_time
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0:
                print(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time:.6f} s")
                # print_memory_stats()
            iter_ += 1
        data_json["train_acc"].append(100.*correct/total)
        data_json["train_loss"].append(epoch_loss/len(train_loader))
        data_json["iter_times"].append(iter_times)
        print(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times):.6f} s")
    
    json.dump(data_json, open("file.json", 'w'), indent=2)
    print("\nEnd of training.")
