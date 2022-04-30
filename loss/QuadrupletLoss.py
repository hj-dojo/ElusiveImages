import torch
import torch.nn.functional as F


# Implementation based on https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
class QuadrupletLoss(torch.nn.Module):
    """
    Quadruplet loss function.
    Builds on the Triplet Loss and takes 4 data input: one anchor, one positive and two negative examples. The negative examples needs not to be matching the anchor, the positive and each other.
    """
    def __init__(self, margin1=2.0, margin2=1.0, similarity_type = 'euclidean'):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.similarity_type = similarity_type

    def calc_euclidean(self, x1, x2):
        return (x1-x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative1, negative2):

        if self.similarity_type == 'euclidean':
            squarred_distance_pos = self.calc_euclidean(anchor, positive)
            squarred_distance_neg = self.calc_euclidean(anchor, negative1)
            squarred_distance_neg_b = self.calc_euclidean(negative1, negative2)
        else:
            squarred_distance_pos = F.cosine_similarity(anchor, positive)
            squarred_distance_neg = F.cosine_similarity(anchor, negative1)
            squarred_distance_neg_b = F.cosine_similarity(negative1, negative2)

        quadruplet_loss = \
            F.relu(self.margin1 + squarred_distance_pos - squarred_distance_neg) \
            + F.relu(self.margin2 + squarred_distance_pos - squarred_distance_neg_b)

        return quadruplet_loss.mean()