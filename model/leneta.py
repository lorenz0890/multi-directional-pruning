import torch.nn as nn
import torch.nn.functional as F


class LeNetA(nn.Module):
    def __init__(self):
        super(LeNetA, self).__init__()
        self.fc1 = nn.Linear(in_features=28 * 28 * 1, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=100)
        self.output = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.output(x))
        return F.log_softmax(x, dim=1)