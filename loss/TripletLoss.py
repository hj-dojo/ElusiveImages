import torch.nn as nn
import torch
import torch.nn.functional as F

# Implementation based on https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, similarity_type='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.similarity_type = similarity_type

    def calc_euclidean(self, x1, x2):
        return (x1-x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative):
        if self.similarity_type == 'euclidean':
            distance_positive = self.calc_euclidean(anchor, positive)
            distance_negative = self.calc_euclidean(anchor, negative)
        else:
            distance_positive = F.cosine_similarity(anchor, positive)
            distance_negative = F.cosine_similarity(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
