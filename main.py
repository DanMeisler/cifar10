import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models import resnet18
import torch.backends.cudnn as cudnn
import torch

INPUT_SIZE = 32 * 32
NUM_OF_CLASSES = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleCnnModel(nn.Module):
    BATCH_SIZE = 32
    DROPOUT_PROBABILITY = 0.5

    def __init__(self, input_size):
        super(SimpleCnnModel, self).__init__()
        self.input_size = input_size
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_layer_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=2, stride=2),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_layer_1 = nn.Sequential(nn.Dropout(self.DROPOUT_PROBABILITY),
                                        nn.Linear(4 * 4 * 128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU())

        self.fc_layer_2 = nn.Sequential(nn.Dropout(self.DROPOUT_PROBABILITY),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU())

        self.fc_layer_3 = nn.Sequential(nn.Dropout(self.DROPOUT_PROBABILITY),
                                        nn.Linear(128, NUM_OF_CLASSES))

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        return self.fc_layer_3(x)


def save_test_prediction(output_path, test_predictions):
    with open(output_path, "w") as output_file:
        output_file.write("\n".join(map(str, test_predictions)))


def split_train_set(train_set, batch_size):
    num_train = len(train_set)
    validation_percent = 80
    validation_slice_size = int((validation_percent / 100.0) * num_train)

    validation_idx = np.random.choice(num_train, size=validation_slice_size, replace=False)
    train_idx = list(set(np.arange(num_train)) - set(validation_idx))

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_idx),
                                               num_workers=2,
                                               pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=batch_size,
                                                    sampler=SubsetRandomSampler(validation_idx),
                                                    num_workers=2,
                                                    pin_memory=True)

    return train_loader, validation_loader


def plot_average_loss(train_average_loss_per_epoch, validation_average_loss_per_epoch, num_of_epochs):
    plt.plot(range(1, num_of_epochs + 1), train_average_loss_per_epoch, label="train")
    plt.plot(range(1, num_of_epochs + 1), validation_average_loss_per_epoch, label="validation")
    plt.xlabel("epoch number")
    plt.ylabel("average loss")
    plt.legend()
    plt.show()


def train(net, optimizer, train_loader, validation_loader, num_of_epochs):
    train_average_loss_per_epoch = []
    validation_average_loss_per_epoch = []

    for epoch_number in range(num_of_epochs):

        train_correct_count = 0
        train_loss = 0.0

        net.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels, size_average=False)
            train_correct_count += outputs.max(dim=1)[1].eq(labels).sum()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        validation_correct_count = 0
        validation_loss = 0.0
        net.eval()
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels, size_average=False)
            validation_correct_count += outputs.max(dim=1)[1].eq(labels).sum()

            validation_loss += loss.item()

        train_average_loss = train_loss / len(train_loader.sampler)
        validation_average_loss = validation_loss / len(validation_loader.sampler)

        train_average_loss_per_epoch.append(train_average_loss)
        validation_average_loss_per_epoch.append(validation_average_loss)
        print("[%d train] accuracy: %.3f, average loss: %.3f" %
              (epoch_number + 1, float(train_correct_count) / len(train_loader.sampler), train_average_loss))
        print("[%d validation] accuracy: %.3f, average loss: %.3f" %
              (epoch_number + 1, float(validation_correct_count) / len(validation_loader.sampler),
               validation_average_loss))

    plot_average_loss(train_average_loss_per_epoch, validation_average_loss_per_epoch, num_of_epochs)
    print("Finished Training")


def test(net, test_loader):
    predictions = []
    test_correct_count = 0
    test_loss = 0.0
    net.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels, size_average=False)
        predictions.extend(outputs.max(dim=1)[1].data.tolist())
        test_correct_count += outputs.max(dim=1)[1].eq(labels).sum()

        test_loss += loss.item()

    print("[test] accuracy: %.3f, average loss: %.3f" % (float(test_correct_count) / len(test_loader.dataset),
                                                         test_loss / len(test_loader.dataset)))

    print(get_confusion_matrix(list(map(lambda x: x[1], test_loader.dataset)), predictions))
    save_test_prediction("test.pred", predictions)


def get_confusion_matrix(labels, predictions):
    return confusion_matrix(labels, predictions)


def get_resnet18_model():
    resnet18_model = resnet18(pretrained=True)
    for param in resnet18_model.parameters():
        param.requires_grad = False
    resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, NUM_OF_CLASSES)
    return resnet18_model


def get_data(transformer):
    root = './resources'

    if not os.path.exists(root):
        os.mkdir(root)

    train_set = CIFAR10(root=root, train=True, transform=transformer, download=True)
    test_set = CIFAR10(root=root, train=False, transform=transformer, download=True)

    return train_set, test_set


def check_simple_cnn_model():
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_set, test_set = get_data(trans)
    train_loader, validation_loader = split_train_set(train_set, batch_size=SimpleCnnModel.BATCH_SIZE)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=SimpleCnnModel.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=2)
    net = SimpleCnnModel(INPUT_SIZE).to(device)

    print("parameters count = %d" % sum(p.numel() for p in net.parameters() if p.requires_grad))

    optimizer = optim.Adam(net.parameters())

    train(net, optimizer, train_loader, validation_loader, num_of_epochs=30)
    test(net, test_loader)


def check_resnet_18_model():
    trans = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_set, test_set = get_data(trans)
    train_loader, validation_loader = split_train_set(train_set, batch_size=128)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=2)
    net = get_resnet18_model().to(device)

    print("parameters count = %d" % sum(p.numel() for p in net.parameters() if p.requires_grad))

    optimizer = optim.Adam(net.fc.parameters())

    train(net, optimizer, train_loader, validation_loader, num_of_epochs=5)
    test(net, test_loader)


if __name__ == "__main__":
    if device == "cuda":
        cudnn.benchmark = True

    #  check_resnet_18_model()
    check_simple_cnn_model()
