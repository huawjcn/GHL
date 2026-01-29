# This is the demo code for training GHL on FastHebb and DeepHebb.
import argparse
import logging
import torch
import torchvision
from torch import nn, optim

from HebbConv2d import HebbConv2d
from FastHebb import FastHebb
from DeepHebb import DeepHebb


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hebbian is not all you need.")
    # Dataset
    parser.add_argument('--data_path', type=str, default='path/to/data', help='Path to the dataset')

    # Model hyperparameters
    parser.add_argument('--model', type=str, default='DeepHebb', choices=['DeepHebb', 'FastHebb'], help='Choose the model type')

    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=int, default='0', help='CUDA device to use, -1 for CPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Default learning rate / Classifier layer learning rate')
    parser.add_argument('--lr_conv', type=float, default=1e-2, help='Learning rate for convolutional layers')
    parser.add_argument('--wd_conv', type=float, default=1e-2, help='Weight decay for convolutional layers')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Set device
    if torch.cuda.is_available() and args.device != -1:
        device = torch.device('cuda', args.device)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Data loading
    transform_train = transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Instantiate model
    if args.model == 'DeepHebb':
        model = DeepHebb()
    elif args.model == 'FastHebb':
        model = FastHebb()

    conv_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in ['conv', 'Conv', 'conv2d', 'Conv2d']):
            conv_params.append(param)
            logging.info(f"Adding conv param: {name}")
        elif any(keyword in name for keyword in ['classifier', 'Classifier', 'fc', 'FC']):
            classifier_params.append(param)
            logging.info(f"Adding classifier param: {name}")
        else:
            logging.info(f"Adding other param: {name} to classifier group")
            classifier_params.append(param)

    optim_params = [
        {'params': conv_params, 'lr': args.lr_conv, 'weight_decay': args.wd_conv},
        {'params': classifier_params, 'lr': args.lr},
    ]

    model.to(device)
    optimizer = optim.SGD(optim_params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(trainloader))
    criterion = nn.CrossEntropyLoss()

    # Initialize training state variables
    start_epoch = 0
    best_test_acc = 0.0
    best_epoch = 0
    current_lr = 0
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    # Start supervised training loop
    print(f'Start supervised training')
    print(f'Epoch\tTrain\tTest\tTrain\tTest')
    print(f'\tloss\tloss\tacc\tacc\tLR')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Iterate over training data
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # --- Forward pass ---
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # --- Backward pass ---
            loss.backward()

            # --- Gradient update ---
            for module in model.modules():
                if isinstance(module, HebbConv2d):
                    module.update_gradients()  # In-place update of .weight.grad

            # --- Optimizer step ---
            optimizer.step()

            # --- Learning rate adjustment ---
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            # --- Training statistics ---
            _, predicted_train = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted_train == labels).sum().item()
            train_loss += loss.item() * inputs.size(0)

        # --- Evaluation phase ---
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # Top1
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_loss += criterion(outputs, labels).item() * images.size(0)
        # Print training and testing statistics
        train_loss_epoch = train_loss / train_total
        test_loss_epoch = test_loss / test_total
        train_acc_epoch = 100 * train_correct / train_total
        test_acc_epoch = 100 * test_correct / test_total
        print(f'{epoch + 1}\t{train_loss_epoch:.3f}\t{test_loss_epoch:.3f}\t'
              f'{train_acc_epoch:.2f}\t{test_acc_epoch:.2f}\t'
              f'{current_lr:.2e}')

        train_losses.append(train_loss_epoch)
        test_losses.append(test_loss_epoch)
        train_accs.append(train_acc_epoch)
        test_accs.append(test_acc_epoch)

        # Update best results
        if test_acc_epoch > best_test_acc:
            best_test_acc = test_acc_epoch
            best_epoch = epoch

    # Training finished, print final results
    print(f'Finished Training, best accuracy: {best_test_acc:.3f} at epoch {best_epoch + 1}.')
    print(f'------------------- SUMMARY -------------------')
    print(f'Train\tTest\tTrain\tTest\tBest')
    print(f'loss\tloss\tacc\tacc\tacc\tepoch')
    print(f'{train_losses[-1]:.3f}\t{test_losses[-1]:.3f}\t'
          f'{train_accs[-1]:.2f}\t{test_accs[-1]:.2f}\t'
          f'{best_test_acc:.3f}\t{best_epoch + 1}\t')
    print(f'-----------------------------------------------')