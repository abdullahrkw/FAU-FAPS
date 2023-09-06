import argparse
import os

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler

from torch import optim

from dataloader import MultiViewDataset
from models import ViT, CrossViT   # rename the skeleton file for your implementation / comment before testing for ResNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    parser.add_argument('--model', type=str, default='cvit', help='model to train (default: cvit)')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--problem', type=str, default="Cover", help='Problem for classification')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()

def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)/len(output)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    cm = np.zeros((2,2))
    batch_size = 1
    with torch.no_grad():
        for data, target in test_loader:
            batch_size = len(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            cm += confusion_matrix(target.cpu(), pred.cpu(), labels=[0, 1])

    print(cm)
    test_loss /= len(test_loader)*batch_size
    acc = 100. * correct / (len(test_loader)*batch_size)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader)*batch_size,
        acc))
    return acc


def run(args):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.GaussianBlur(kernel_size=7, sigma=40),

                                    # ImageNet mean/std values
									transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ) 
                                    ])
    
    target_transform = None
    ROOT_DIR = "/home/vault/iwfa/iwfa018h/FAPS/NewMotorsDataset/AugClassification1/"
    ROOT_DIR = os.path.join(ROOT_DIR, args.problem)
    print(ROOT_DIR)
    train_csv_path = os.path.join(ROOT_DIR, "train.csv")
    test_csv_path = os.path.join(ROOT_DIR, "test.csv")
    val_csv_path = os.path.join(ROOT_DIR, "val.csv")

    views = ["file_name"]
    # Order matter for labels
    labels = ["label", "~label"]
    num_classes = 2

    train_dataset = MultiViewDataset(train_csv_path,
                                        views=views,
                                        labels=labels,
                                        base_dir=ROOT_DIR,
                                        transform=transform,
                                        target_transform=target_transform)
    val_dataset = MultiViewDataset(val_csv_path,
                                        views=views,
                                        labels=labels,
                                        base_dir=ROOT_DIR,
                                        transform=transform,
                                        target_transform=target_transform)
    test_dataset = MultiViewDataset(test_csv_path,
                                        views=views,
                                        labels=labels,
                                        base_dir=ROOT_DIR,
                                        transform=transform,
                                        target_transform=target_transform)

    # Sampler to for oversampling/undersampling to counter class imbalance
    class_counts = np.ones(num_classes)
    for i, val in enumerate(train_dataset.data):
        label = np.asarray(val["label"])
        # assuming one-hot encoding
        class_ = np.argmax(label)
        class_counts[class_] += 1

    sample_weights = np.zeros(len(train_dataset.data))
    for i, val in enumerate(train_dataset.data):
        label = np.asarray(val["label"])
        # assuming one-hot encoding
        class_ = np.argmax(label)
        sample_weights[i] = 1/class_counts[class_]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=False, batch_size=16, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=16, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16, num_workers=4, drop_last=True)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(weights=None, num_classes=2)
    elif args.model == "vit":
        model = ViT(image_size = 256, patch_size = 16, num_classes = 2, dim = 64,
                    depth = 2, heads = 8, mlp_dim = 128, dropout = 0.2,
                    emb_dropout = 0.1) 
    elif args.model == "cvit":
        model = CrossViT(image_size = 256, num_classes = 2, sm_dim = 192, 
                         lg_dim = 384, sm_patch_size = 16, sm_enc_depth = 2,
                         sm_enc_heads = 8, sm_enc_mlp_dim = 2048, 
                         sm_enc_dim_head = 64, lg_patch_size = 64, 
                         lg_enc_depth = 3, lg_enc_heads = 8, 
                         lg_enc_mlp_dim = 2048, lg_enc_dim_head = 64,
                         cross_attn_depth = 2, cross_attn_heads = 8,
                         cross_attn_dim_head = 64, depth = 3, dropout = 0.1,
                         emb_dropout = 0.1)

    # Define the loss
    criterion = nn.CrossEntropyLoss(reduction="sum")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        test(model, device, val_loader, criterion, set="Validation")

    acc = test(model, device, test_loader, criterion)
    if args.save_model:
        torch.save(model, f"models/{args.model}_{args.epochs}_{int(acc)}.pth")

if __name__ == '__main__':
    args = parse_args()
    run(args)
