from torch import nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.Norm1 = nn.BatchNorm2d(64)
        self.Norm2 = nn.BatchNorm2d(128)
        self.Norm3 = nn.BatchNorm2d(256)
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.Norm4 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)

        self.bmi_class = nn.Linear(64, 7)
        self.bmi_out = nn.Linear(7, 1)
        self.weights_init()

    def forward(self, x):
        x = self.Norm1(self.pool(F.relu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.Norm2(self.pool(F.relu(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.Norm3(self.pool(F.relu(self.conv5(x))))
        x = self.flat(x)
        x = self.Norm4(self.drop(F.relu(self.fc1(x))))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.bmi_class(x)
        y = self.bmi_out(x)
        x = F.softmax(x)
        y = F.sigmoid(y)

        return x, y


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)



