from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description = 'Pytorch MNIST example')
parser.add_argument('--batch-size', type=int, default = 256, metavar='N', \
                    help = 'batch size')
parser.add_argument('--test-batch-size', type = int, default = 1000, \
                    metavar = 'N', help = 'test batch size' )
parser.add_argument('--epochs', type = int, default = 10, metavar = 'N', \
                    help = 'number of epochs to train.')
parser.add_argument('--lr', type = float, default = 0.01, metavar = 'LR', \
                    help = 'learning rate')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(x, 2))#conv->max_pool->relu
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))#conv->dropout->max_pool->relu
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))#fc->relu
        x = F.dropout(x, training=self.training)#dropout
        x = self.fc2(x)
        #return x
        return F.log_softmax(x)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(\
                epoch, batch_idx * len(data), len(train_loader.dataset),\
                100. * batch_idx / len(train_loader), loss.item()))#loss.data[0]

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    kwargs = {'num_workers':8, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader( \
        datasets.MNIST('data', train = True, download = False, \
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size = args.batch_size, shuffle = True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Net()
    if args.cuda:
        #normal bn.
        model = nn.DataParallel(model, device_ids=[0,1,2,3, 4, 5,6 ,7]).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
