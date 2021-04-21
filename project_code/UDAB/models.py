import torch
import torch.nn as nn
import sys

sys.path.append('./')

from torch.utils.tensorboard import SummaryWriter
from UDAB.functions import ReverseLayer

class UDAB(nn.Module):

    def __init__(self, track_running_stats, momentum):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(310, 256), 
            nn.BatchNorm1d(num_features=256, momentum=momentum, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128, momentum=momentum, track_running_stats=track_running_stats),
            nn.Dropout(),
            nn.ReLU()
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64, momentum=momentum, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32, momentum=momentum, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32, momentum=momentum, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha=0):
        feature = self.feature_extractor(input_data)
        reverse_feature = ReverseLayer.apply(feature, alpha)
        class_pred = self.class_classifier(feature)
        
        domain_pred = self.domain_classifier(reverse_feature)

        return class_pred, domain_pred

if __name__ == '__main__':
    writer = SummaryWriter()
    model = UDAB(True, 0.5) 
    writer.add_graph(model, input_to_model=torch.zeros((10, 310)), verbose=True)
    writer.close()



