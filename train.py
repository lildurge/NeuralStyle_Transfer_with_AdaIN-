from torch import nn
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
import torch
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
import vanilla_net
import mod_net
import datetime
import matplotlib.pyplot as plt
import ssl
from torch.backends import cudnn

# Was getting certificate error
ssl._create_default_https_context = ssl._create_unverified_context

cudnn.benchmark = True


def train_transform():
    transform_list = [
        transforms.Resize((64, 64)),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transform_list)


def plot_loss(losses_train, plot_path):
    plt.plot(losses_train)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(plot_path)


def train(train_loader, encoder_path, epochs, cuda):
    # Make sure we are using gpu or cpu
    print('Using device: ', cuda)

    # Print training start
    print('---------------Training Start---------------')

    # Loading in the model depending on the argument chosen
    args.m = args.m.lower()
    if args.m == 'v':
        encoder = vanilla_net.encoder_decoder.vgg_backend
        encoder.load_state_dict(torch.load(encoder_path))
        decoder = vanilla_net.encoder_decoder.frontend
        model = vanilla_net.Model().to(device=device)
    else:  # else just use the mod_net
        encoder = mod_net.encoder_decoder.vgg_backend
        encoder.load_state_dict(torch.load(encoder_path))
        decoder = mod_net.encoder_decoder.frontend
        layers_config = [2, 2, 2, 2]
        model = mod_net.Model(mod_net.ResNet, layers_config, num_classes=100).to(device=device)

    model.train()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    losses_train = []

    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        start_time = datetime.datetime.now()

        for batch in train_loader:
            # Get the inputs and labels
            inputs, labels = batch[0].to(device=device), batch[1].to(device=device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            training_loss += loss.item()

        scheduler.step()

        # Print the average loss after each epoch
        average_loss = training_loss / len(train_loader)
        losses_train.append(average_loss)

        # Print out info to terminal
        print(f'Epoch [{epoch}/{epochs}] | Average Training Loss: {average_loss:.4f} | Time elapsed: '
              f'{datetime.datetime.now() - start_time}')

    # Save the model
    torch.save(decoder.state_dict(), args.s)

    # Plot the training loss
    plot_loss(losses_train, args.p)

    return losses_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, required=True, help='Number of epochs')
    parser.add_argument('-b', type=int, required=True, help='Batch size')
    parser.add_argument('-l', type=str, required=True, help='Encoder path')
    parser.add_argument('-s', type=str, required=True, help='Save path')
    parser.add_argument('-p', type=str, required=True, help='Plot path')
    parser.add_argument('-m', type=str, required=True, help='[V/M]')
    parser.add_argument('-cuda', type=str, default='N', help='[Y/N]')
    args = parser.parse_args()

    # Loading in the dataset
    train_transforms = train_transform()
    train_data = CIFAR100(root='./data', train=True, download=True, transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.b, shuffle=True, num_workers=2)

    # Setting up the device
    if args.cuda == 'y' or args.cuda == 'Y':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Test for flattened size
    # dummy_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    # model = vanilla_net.Model()
    # print(model.encode(dummy_input).size())

    train(train_loader, args.l, args.e, device)
