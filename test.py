import argparse
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import vanilla_net
import mod_net


def topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Testing loop
def test(model, test_loader):
    top1_accuracy = 0.0
    top5_accuracy = 0.0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            top1, top5 = topk_accuracy(outputs, labels, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
            total += labels.size(0)

    top1_accuracy = top1_accuracy / total
    top5_accuracy = top5_accuracy / total
    return top1_accuracy, top5_accuracy


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, required=True, help='Decoder path')
    parser.add_argument('-m', type=str, required=True, help='[V/M]')  # V for vanilla, M for modified
    parser.add_argument('-cuda', type=str, default='N', help='[Y/N]')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the test dataset and loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR100 mean and std
    ])

    test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    # Loading in the model depending on the argument chosen
    args.m = args.m.lower()
    if args.m == 'v':
        frontend = vanilla_net.encoder_decoder.frontend
        frontend.load_state_dict(torch.load(args.s))
        model = vanilla_net.Model(num_classes=100).to(device)
        model.eval()
        vanilla_top1, vanilla_top5 = test(model, test_loader)
        print(f'Vanilla Model - Top-1 Accuracy: {vanilla_top1:.2f}%, Top-5 Accuracy: {vanilla_top5:.2f}%')

    else:  # Load modified model
        frontend = mod_net.encoder_decoder.frontend
        frontend.load_state_dict(torch.load(args.s))
        model = mod_net.Model(mod_net.ResNet, [2, 2, 2, 2]).to(device)
        model.eval()
        modified_top1, modified_top5 = test(model, test_loader)
        print(f'Modified Model - Top-1 Accuracy: {modified_top1:.2f}%, Top-5 Accuracy: {modified_top5:.2f}%')
