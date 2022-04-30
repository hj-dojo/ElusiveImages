import torch
import torch.nn.functional as F


# Implementation based on https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0, similarity_type = 'euclidean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.similarity_type = similarity_type

    def calc_euclidean(self, x1, x2):
        return torch.sqrt((x1-x2).pow(2).sum(1))

    def forward(self, output1, output2, label):
        if self.similarity_type == 'euclidean':
            # Find the pairwise distance or euclidean distance of two output feature vectors
            distance = self.calc_euclidean(output1, output2)
        else:
            distance = F.cosine_similarity(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive
