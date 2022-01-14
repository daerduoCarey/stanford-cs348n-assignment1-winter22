import os
import sys
import torch
from torch import nn
from cd.chamfer import chamfer_distance


# the PointNet point cloud shape encoder
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        
        # we can use conv1d with kernel-size=1 to implement the per-point MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)

        # batch-norm can help stabilize and speed up the learning
        # perform batch-norm over per-point features across all points in all shapes
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)

        # a 2-layer MLP to further transform the global shape feature
        self.fc1 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.bn6 = nn.BatchNorm1d(128)

    """
        Input: B x N x 3
        Output: B x F
    """
    def forward(self, pcs):
        net = pcs.permute(0, 2, 1)

        # STUDENT CODE START
        # use self.conv1-4 and self.bn1-4 to extract per-point features
        # do not forget about the non-linear activation functions (e.g. ReLU)


        # get the global shape feature using a max-pooling operation


        # use self.fc1/2 and self.bn5/6 for the global feature MLP
        # do not forget about the non-linear activation functions (e.g. ReLU)

        # STUDENT CODE END
        
        return net


# a small MLP for variational encoding
class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()
        self.probabilistic = probabilistic

        # first transform the feature from the encoder into a hidden space
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        # use a FC-layer to output the mu of the Gaussian
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        # use another FC-layer to output the log-variance of the Gaussian
        self.mlp2var = nn.Linear(hidden_size, feature_size)
            
        if probabilistic:
            print('[Sampler] Variational Sampler is activated!')

    """
        Input: B x F
        Output: B x F if probabilistic = False
                B x 2F otherwise
    """
    def forward(self, x):
        encode = torch.relu(self.mlp1(x))
        mu = self.mlp2mu(encode)

        if self.probabilistic:
            logvar = self.mlp2var(encode)

            # STUDENT CODE START


            # STUDENT CODE END

            # return a concatenation of the jittered code z and the kld loss term
            # Check Eq (10) in https://arxiv.org/pdf/1312.6114.pdf if you need help for the kld loss term
            return torch.cat([ret, kld], 1)
        else:
            return mu


# a small MLP for variational decoding (just 2 FC-layers)
class SampleDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    """
        Input: B x F
        Output: B x F
    """
    def forward(self, input_feature):
        output = torch.relu(self.mlp1(input_feature))
        output = torch.relu(self.mlp2(output))
        return output


# a simple MLP-based point cloud shape decoder
class FCDecoder(nn.Module):

    def __init__(self, num_point=1024):
        super(FCDecoder, self).__init__()

        self.mlp1 = nn.Linear(128, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, num_point*3)

    """
        Input: B x F
        Output: B x NumPoint x 3
    """
    def forward(self, feat):
        batch_size = feat.shape[0]

        # STUDENT CODE START
        # we do not use batch-norm for decoding
        # but we do need non-linear activation functions (e.g. ReLU)


        # STUDENT CODE END

        return net


class Network(nn.Module):

    def __init__(self, conf):
        super(Network, self).__init__()
        self.conf = conf

        self.encoder = PointNet()
        self.sample_encoder = Sampler(128, 256, probabilistic=conf.probabilistic)

        self.sample_decoder = SampleDecoder(128, 256)
        self.decoder = FCDecoder(num_point=conf.num_point)

    """
        Input: B x N x 3
        Output: B x N x 3, B x F
    """
    def forward(self, input_pcs):
        feats = self.encoder(input_pcs)
        feats = self.sample_encoder(feats)

        ret_list = dict()
        if self.conf.probabilistic:
            feats, obj_kldiv_loss = torch.chunk(feats, 2, 1)
            ret_list['kldiv_loss'] = -obj_kldiv_loss.sum(dim=1)

        feats = self.sample_decoder(feats)
        output_pcs = self.decoder(feats)

        return output_pcs, feats, ret_list
    

    """
        Input: B x N x 3, B x M x 3
        Output: B
    """
    def get_loss(self, pc1, pc2):
        dist1, dist2 = chamfer_distance(pc1, pc2, transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

        return loss_per_data
    
