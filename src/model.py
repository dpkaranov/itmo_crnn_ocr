import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class DeepBiLSTM(nn.Module):

    def __init__(self, inF, hidden, out):
        super(DeepBiLSTM, self).__init__()
        self.rnn = nn.LSTM(inF, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden * 2, out)

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        l, b, h = x.size()
        x = x.view(l * b, h)
        x = self.embedding(x)
        x = x.view(l, b, -1)
        return x

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 3, stride=1, padding=1)
        self.pooling1 = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride=1, padding=1)
        self.pooling2 = nn.MaxPool2d((2, 2), stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size = 3, stride=1, padding=1)
        self.pooling3 = nn.MaxPool2d((1, 2), stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size = 3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size = 3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.pooling4 = nn.MaxPool2d((1, 2), stride=2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size = 2, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling3(x)
        x = self.conv5(x)
        x = self.batchnorm1(x)
        x = self.conv6(x)
        x = self.batchnorm2(x)
        x = self.pooling4(x)
        x = self.conv7(x)
        return x

class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn = CNN()
        self.mapping = nn.Linear(256, 11)
        self.bilstm1 = DeepBiLSTM(512,256,256)
        self.bilstm2 = DeepBiLSTM(256,256,256)

    def forward(self, x):
        x = x.permute(1,0,2,3)
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        x = self.mapping(x)
        x = nn.functional.log_softmax(x, 2)
        return x
