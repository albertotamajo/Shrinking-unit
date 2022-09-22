import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch_geometric.nn as gnn
import networkx as netx
from torch_geometric.utils import sort_edge_index, add_self_loops
from torch_scatter import scatter_max
from threading import Thread
import copy


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ShrinkingUnit(nn.Module):
    """
    This class implements the Shrinking unit.
    """
    def __init__(self, mlp: nn.Module, learning_rate: int, k: int, kmeansInit, n_init, sigma: nn.Module, F: nn.Module, W: nn.Module,
                 M: nn.Module, C, P, mlp1: nn.Module, mlp2: nn.Module):
        """
        Instantiate a shrinking unit object.
        :param mlp: a R^C -> R^C MLP, where C is the dimensionality of the input points. It is used in the
                    Self-correlation module. It is denoted as $f$ in the paper.
        :param learning_rate: this is the initial value for the $\lambda$ parameter. It is used in the Self-correlation
                              module.
        :param k: the number of clusters. It is used in the K-Means-Conv module. It is denoted as $K$ in the paper.
        :param kmeansInit: method for initialization of the k-means algorithm. "k-means++" is the default method
                           proposed in the paper. It is used in the K-Means-Conv module.
        :param n_init: number of time the k-means algorithm will be run with different centroid seeds. It is used in the
                       K-Means-Conv module.
        :param sigma: activation function. It is denoted as $\sigma$ in the paper. The sigmoid function is the
                      default activation function proposed in the paper. It is used in the K-Means-Conv module.
        :param F: F is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{F}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons. It is used in the K-Means-Conv module.
        :param W: W is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{W}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons. It is used in the K-Means-Conv module.
        :param M: M is a R^(C+P) -> R MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{M}$ in the paper. It is used in the K-Means-Conv module.
        :param C: the dimensionality of the points to be convolved.
        :param P: the difference in dimensionality between the points to be convolved and the output points.
        :param mlp1: a R^(C+P) -> R^(C+P) MLP, where C is the dimensionality of the points prior to the convolution
                    operation and P is the difference in dimensionality between the points prior to the convolution
                    operation and the output points of the convolution operation. It is denoted as $f_1$ in the paper.
                    It is used in the Aggregation module.
        :param mlp2: a R^(C+P) -> R^(C+P) MLP, where C is the dimensionality of the points prior to the convolution
                    operation and P is the difference in dimensionality between the points prior to the convolution
                    operation and the output points of the convolution operation. It is denoted as $f_2$ in the paper.
                    It is used in the Aggregation module.
        """
        super().__init__()
        self.selfCorr = SelfCorrelation(mlp, learning_rate)
        self.kmeansConv = KMeansConv(k, kmeansInit, n_init, sigma, F, W, M, C, P)
        self.localAdaptFeaAggre = Aggregation(mlp1, mlp2)
        self.graphMaxPool = MaxPool(k)

    def forward(self, feature_matrix_batch):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = self.selfCorr(feature_matrix_batch)
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        feature_matrix_batch, conv_feature_matrix_batch, cluster_index = self.kmeansConv(feature_matrix_batch)
        feature_matrix_batch = self.localAdaptFeaAggre(feature_matrix_batch, conv_feature_matrix_batch)
        output = self.graphMaxPool(feature_matrix_batch, cluster_index)
        # output size = (N,K,D) where N=batch number, K=members, D=member dimensionality
        return output


class SelfCorrelation(nn.Module):
    """
    This class implements the Self-correlation module of the Shrinking unit.
    """
    def __init__(self, mlp: nn.Module, learning_rate: int = 1.0):
        """
        Instantiate a Self-correlation module object.
        :param mlp: a R^C -> R^C MLP, where C is the dimensionality of the input points.
                    It is denoted as $f$ in the paper.
        :param learning_rate: this is the initial value for the $\lambda$ parameter
        """
        super().__init__()
        self.mlp = mlp
        self.learning_rate = torch.tensor(learning_rate, requires_grad=True)

    def forward(self, feature_matrix_batch):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        N, I, D = feature_matrix_batch.size()
        feature_matrix_batch = feature_matrix_batch.view(-1, D)
        # feature_matrix_batch size = (L,D) where L=N*I, D=member dimensionality
        Weight = self.mlp(feature_matrix_batch)
        # Weight size = (L,D) where L=N*I, D=member dimensionality
        output = (self.learning_rate * feature_matrix_batch * Weight) + feature_matrix_batch
        # output size = (L,D) where L=N*I, D=member dimensionality
        output = output.view(N, I, D)
        # output size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        # output size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        return output


class KMeansConv(nn.Module):
    """
    This class implements the K-Means-Conv module of the Shrinking unit.
    """
    def __init__(self, k: int, kmeansInit, n_init: int, sigma: nn.Module, F: nn.Module, W: nn.Module, M: nn.Module, C: int, P: int):
        """
        Instantiate a K-Means-Conv module object.
        :param k: the number of clusters. It is denoted as $K$ in the paper.
        :param kmeansInit: method for initialization of the k-means algorithm. "k-means++" is the default method
                           proposed in the paper
        :param n_init: number of time the k-means algorithm will be run with different centroid seeds.
        :param sigma: activation function. It is denoted as $\sigma$ in the paper. The sigmoid function is the
                      default activation function proposed in the paper.
        :param F: F is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{F}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons.
        :param W: W is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{W}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons.
        :param M: M is a R^(C+P) -> R MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{M}$ in the paper.
        :param C: the dimensionality of the points to be convolved.
        :param P: the difference in dimensionality between the points to be convolved and the output points.
        """
        super().__init__()
        self.k = k
        self.kmeansInit = kmeansInit
        self.n_init = n_init
        self.conv = Conv(sigma, F, W, M, C, P)

    def forward(self, feature_matrix_batch):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        N, I, D = feature_matrix_batch.size()
        clusters = []
        for i, feature_matrix in enumerate(feature_matrix_batch):
            kmeans = KMeans(n_clusters=self.k, init=self.kmeansInit, n_init=self.n_init)
            labels = np.apply_along_axis(lambda x: x + (i*self.k), axis=0, arr=kmeans.labels_)
            clusters.extend(labels)
        clusters = np.asarray(clusters)
        list1 = []
        list2 = []
        for i in range(self.k*N):
            indices = np.argwhere(clusters == i).flatten().tolist()
            if len(indices) != 1:
                edges = [e for e in netx.complete_graph(indices).edges]
                inverse_edges = list(map(lambda x: (x[1], x[0]), edges))
                edges.extend(inverse_edges)
                unzip = list(zip(*edges))
                list1.extend(unzip[0])
                list2.extend(unzip[1])
            else:
                list1.append(indices[0])
                list2.append(indices[0])

        edge_index = torch.tensor([list1, list2], dtype=torch.long, device=getDevice(feature_matrix_batch))
        edge_index = sort_edge_index(add_self_loops(edge_index)[0])[0]
        conv_feature_matrix_batch = self.conv(feature_matrix_batch.view(-1, D), edge_index).view(N, I, -1)
        # conv_feature_matrix_batch size = (N,I,L) where N=batch number, I=members, L=C+P
        return feature_matrix_batch, conv_feature_matrix_batch, torch.tensor(clusters, dtype=torch.long, device=getDevice(feature_matrix_batch))


def getDevice(t: torch.Tensor):
    if t.is_cuda:
        return f"cuda:{t.get_device()}"
    else:
        return "cpu"


class KMeansInitMostDistantFromMean:
    def __call__(self, *args, **kwargs):
        X, k = args
        mean = np.mean(X, axis=0)
        arg_sorted = np.argsort(np.apply_along_axis(lambda y: euclidean(mean, y), 1, X))
        output = X[np.flip(arg_sorted)[:k]]
        return output


class KMeansInit:
    def __call__(self, *args, **kwargs):
        X, k = args
        current_centroids = np.expand_dims(np.mean(X, axis=0), 0)
        for i in range(k - 1):
            X, current_centroids = self.next_centroid(X, current_centroids)

        return current_centroids

    def next_centroid(self, X, curr_centroids):
        highest_dist = 0.0
        next_centroid = None
        next_centroid_index = None
        for i, x in enumerate(X):
            max_dist = np.amax(np.apply_along_axis(lambda y: euclidean(x, y), 1, curr_centroids))
            if max_dist > highest_dist:
                next_centroid = x
                highest_dist = max_dist
                next_centroid_index = i

        return np.delete(X, next_centroid_index, 0), np.append(curr_centroids, np.expand_dims(next_centroid, 0), 0)


class Conv(gnn.MessagePassing):
    """
    Convolution operation used in the K-Means-Conv module
    """
    def __init__(self, sigma: nn.Module, F: nn.Module, W: nn.Module, M: nn.Module, C: int, P: int):
        """
        Instantiate a convolution operation object.
        :param sigma: activation function. It is denoted as $\sigma$ in the paper. The sigmoid function is the
                      default activation function proposed in the paper.
        :param F: F is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{F}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons.
        :param W: W is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{W}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons.
        :param M: M is a R^(C+P) -> R MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{M}$ in the paper.
        :param C: the dimensionality of the points to be convolved.
        :param P: the difference in dimensionality between the points to be convolved and the output points.
        """
        super().__init__(aggr="mean")
        self.sigma = sigma
        self.F = F
        self.W = W
        self.M = M
        self.C = C
        self.P = P
        self.B = torch.randn(C+P, requires_grad=True)

    def forward(self, feature_matrix, edge_index):
        # feature_matrix size = (N,C) where N=number points, C=points dimensionality
        # edge_index size = (2,E) where N=number points, E=edges
        return self.propagate(edge_index, feature_matrix=feature_matrix)

    def message(self, feature_matrix_i, feature_matrix_j):
        # feature_matrix_i size = (E, C) where E=edges, C=point dimensionality
        # feature_matrix_j size = (E, C) where E=edges, C=point dimensionality
        message = self.F(feature_matrix_j - feature_matrix_i)
        # message size = (E, M) where E=edges and M=C x (C+P)
        message = message.view(-1, self.C + self.P, self.C)
        # message size = (E, M, L) where E=edges, M=C+P, L=C
        feature_matrix_i_ = feature_matrix_i.unsqueeze(2)
        # feature_matrix_i_ size = (E,C,1) where E=edges, C=point dimensionality
        output = torch.bmm(message, feature_matrix_i_).squeeze()
        # output size = (E,M) where E=edges, M=C+P
        return output

    def update(self, aggr_out, feature_matrix):
        # aggr_out size = (N,L) where N=number points and L=C+P
        # feature_matrix size = (N,C) where N=number points, C=points dimensionality
        Weight = self.M(aggr_out)
        # Weight size = (N,1) where N=number points
        aggr_out = aggr_out * Weight
        # aggr_out size = (N,L) where N=number points and L=C+P
        transform = self.W(feature_matrix)
        # transform size = (N, M) where N=number points, M=(C x (C+P))
        transform = transform.view(-1, self.C + self.P, self.C)
        # transform size = (N, M, L) where N=number points, M=C+P, L=C
        feature_matrix = feature_matrix.unsqueeze(2)
        # feature_matrix size = (N,C,1) where N=number points, C=points dimensionality
        transformation = torch.bmm(transform, feature_matrix).squeeze()
        # transformation size = (N,L) where N=number points, L=C+P
        aggr_out = aggr_out + transformation
        # aggr_out size = (N,L) where N=number points, L=C+P
        output = aggr_out + self.B
        # output size = (N,L) where N=number points, L=C+P
        output = self.sigma(output)
        # output size = (N,L) where N=number points, L=C+P
        return output


class Aggregation(nn.Module):
    """
    This class implements the Aggregation module of the Shrinking unit.
    """
    def __init__(self, mlp1: nn.Module, mlp2: nn.Module):
        """
        Instantiate an aggregation module object.

        :param mlp1: a R^(C+P) -> R^(C+P) MLP, where C is the dimensionality of the points prior to the convolution
                    operation and P is the difference in dimensionality between the points prior to the convolution
                    operation and the output points of the convolution operation. It is denoted as $f_1$ in the paper.
        :param mlp2: a R^(C+P) -> R^(C+P) MLP, where C is the dimensionality of the points prior to the convolution
                    operation and P is the difference in dimensionality between the points prior to the convolution
                    operation and the output points of the convolution operation. It is denoted as $f_2$ in the paper.
        """
        super().__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.softmax = nn.Softmax(0)

    def forward(self, feature_matrix_batch: torch.Tensor, conv_feature_matrix_batch: torch.Tensor):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=cluster members, D=member dimensionality
        # conv_feature_matrix_batch size = (N,I,D') where N=batch number, I=cluster members, D'=C+P
        N, I, D = feature_matrix_batch.size()
        N_, I_, D_ = conv_feature_matrix_batch.size()
        augmentation = D_ - D
        if augmentation > 0:
            feature_matrix_batch = F.pad(feature_matrix_batch, (0, augmentation))
            # feature_matrix_batch size = (N,I,D') where N=batch number, I=members, D'=C+P
        S1 = torch.mean(feature_matrix_batch, 1)
        # S1 size = (N,D') where N=batch number, D'=C+P
        S2 = torch.mean(conv_feature_matrix_batch, 1)
        # S2 size = (N,D') where N=batch number, D'=C+P
        Z1 = self.mlp1(S1)
        # Z1 size = (N,D') where N=batch number, D'=C+P
        Z2 = self.mlp2(S2)
        # Z2 size = (N,D') where N=batch number, D'=C+P
        M = self.softmax(torch.stack((Z1, Z2), 0))
        # torch.stack((Z1, Z2), 0)) size = (2,N,D') where N=batch number, D'=C+P
        # M size = (2,N,D') where N=batch number, D'=C+P
        M1 = M[0]
        # M1 size = (N,D') where N=batch number, D'=C+P
        M2 = M[1]
        # M2 size = (N,D') where N=batch number, D'=C+P
        M1 = M1.unsqueeze(1).expand(-1, I, -1)
        M2 = M2.unsqueeze(1).expand(-1, I, -1)
        # M1 size = (N,I,D') where N=batch number, I=cluster members, D'=C+P
        # M2 size = (N,I,D') where N=batch number, I=cluster members, D'=C+P
        output = (M1 * feature_matrix_batch) + (M2 * conv_feature_matrix_batch)
        # output size = (N,I,D') where N=batch number, I=cluster members, D'=C+P
        return output


class MaxPool(nn.Module):
    """
    This class implements the Max-pool module of the Shrinking unit.
    """
    def __init__(self, k: int):
        """
        Instantiate a max-pool module object
        :param k: number of clusters previously used in the K-Means-Conv module
        """
        super().__init__()
        self.k = k

    def forward(self, feature_matrix_batch: torch.Tensor, cluster_index: torch.Tensor):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        # cluster_index size = (M) where M=N*I
        N, I, D = feature_matrix_batch.size()
        feature_matrix_batch = feature_matrix_batch.view(-1, D)
        # feature_matrix_batch size = (M,D) where M=N*I, D=member dimensionality
        output = scatter_max(feature_matrix_batch, cluster_index, dim=0)[0]
        # output size = (L,D) where L=k*N, D=member dimensionality
        output = output.view(N, self.k, -1)
        #output size = (N,K,D) where N=batch number, K=clusters, D=member dimensionality
        return output


class GraphConvPool3DPnet(nn.Module):
    """
    GraphConvPool3DPNet is a 3D point clouds artificial neural network classifier.
    It consists of two main sections:

        - Shrinking layers => a stacked sequence of layers (each layer is called Shrinking Layer).
                              Given a shrinking layer N, the shrinking layer N+1 receives the output of N.

        - Classifier => => a classic MLP which receives as input the output of the Shrinking Layers section
                           and outputs the probability distribution for categorisation purposes
    """
    def __init__(self, shrinkingLayers: [ShrinkingUnit], mlp: nn.Module):
        super().__init__()
        self.neuralNet = nn.Sequential(*shrinkingLayers, mlp)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # x should be None when starting with a point cloud with no features apart from the euclidean coordinates
        # pos size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = torch.cat((pos, x), 2) if x is not None else pos
        return self.neuralNet(feature_matrix_batch)


###################################################################################################################
###################################################################################################################

class ShrinkingUnitStack(nn.Module):
    """
    Utility for creating a vertical stack of Shrinking units
    """
    def __init__(self, input_stack: int, stack_fork: int, mlp: nn.Module, learning_rate: int, k: int, kmeansInit, n_init, sigma: nn.Module, F: nn.Module, W: nn.Module,
                 M: nn.Module, C, P, mlp1: nn.Module, mlp2: nn.Module):
        """
        Instantiate a stack of Shrinking units. The arguments of this function, except input_stack and stack_fork,
        are deep copied for each Shrinking unit. The weights of each Shrinking unit are initialised separately
        using the xavier initialisation procedure.
        :param input_stack: number of point cloud inputs
        :param stack_fork: number of different Shrinking units that process each point cloud input
        :param mlp: a R^C -> R^C MLP, where C is the dimensionality of the input points. It is used in the
                    Self-correlation module. It is denoted as $f$ in the paper.
        :param learning_rate: this is the initial value for the $\lambda$ parameter. It is used in the Self-correlation
                              module.
        :param k: the number of clusters. It is used in the K-Means-Conv module. It is denoted as $K$ in the paper.
        :param kmeansInit: method for initialization of the k-means algorithm. "k-means++" is the default method
                           proposed in the paper. It is used in the K-Means-Conv module.
        :param n_init: number of time the k-means algorithm will be run with different centroid seeds. It is used in the
                       K-Means-Conv module.
        :param sigma: activation function. It is denoted as $\sigma$ in the paper. The sigmoid function is the
                      default activation function proposed in the paper. It is used in the K-Means-Conv module.
        :param F: F is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{F}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons. It is used in the K-Means-Conv module.
        :param W: W is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{W}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons. It is used in the K-Means-Conv module.
        :param M: M is a R^(C+P) -> R MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{M}$ in the paper. It is used in the K-Means-Conv module.
        :param C: the dimensionality of the points to be convolved.
        :param P: the difference in dimensionality between the points to be convolved and the output points.
        :param mlp1: a R^(C+P) -> R^(C+P) MLP, where C is the dimensionality of the points prior to the convolution
                    operation and P is the difference in dimensionality between the points prior to the convolution
                    operation and the output points of the convolution operation. It is denoted as $f_1$ in the paper.
                    It is used in the Aggregation module.
        :param mlp2: a R^(C+P) -> R^(C+P) MLP, where C is the dimensionality of the points prior to the convolution
                    operation and P is the difference in dimensionality between the points prior to the convolution
                    operation and the output points of the convolution operation. It is denoted as $f_2$ in the paper.
                    It is used in the Aggregation module.
        """
        super().__init__()
        self.stack_fork = stack_fork
        stack_size = input_stack * stack_fork
        self.selfCorrStack = SelfCorrelationStack(stack_size, mlp, learning_rate)
        self.kmeansConvStack = KMeansConvStack(stack_size, k, kmeansInit, n_init, sigma, F, W, M, C, P)
        self.localAdaptFeaAggreStack = AggregationStack(stack_size, mlp1, mlp2)
        self.graphMaxPoolStack = MaxPoolStack(stack_size, k)

    def forward(self, feature_matrix_batch):
        # feature_matrix_batch size = (S,N,I,D) where S= input stack, N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = torch.repeat_interleave(feature_matrix_batch, self.stack_fork, dim=0)
        # feature_matrix_batch size = (S',N,I,D) where S'=stack_size, N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = self.selfCorrStack(feature_matrix_batch)
        # feature_matrix_batch size = (S',N,I,D) where S'=stack_size, N=batch number, I=members, D=member dimensionality
        feature_matrix_batch_, conv_feature_matrix_batch, cluster_index = self.kmeansConvStack(feature_matrix_batch)
        feature_matrix_batch = self.localAdaptFeaAggreStack(feature_matrix_batch, conv_feature_matrix_batch)
        output = self.graphMaxPoolStack(feature_matrix_batch, cluster_index)
        # output size = (S',N,K,D) where S'=stack_size, N=batch number, K=members, D=member dimensionality
        return output


class SelfCorrelationStack(nn.Module):
    """
    Utility for creating a vertical stack of Self-correlation modules
    """
    def __init__(self, stack_size: int, mlp: nn.Module, learning_rate: int = 1.0):
        """
        Instantiate a vertical stack of Self-correlation modules. The arguments of this function, except stack_size,
        are deep copied for each Self-correlation module. The weights of each Self-correlation module are initialised
        separately using the xavier initialisation procedure.
        :param stack_size: number of Self-correlation modules
        :param mlp: a R^C -> R^C MLP, where C is the dimensionality of the input points.
                    It is denoted as $f$ in the paper.
        :param learning_rate: this is the initial value for the $\lambda$ parameter.
        """
        super().__init__()
        self.selfCorrelationStack = nn.ModuleList([SelfCorrelation(copy.deepcopy(mlp), learning_rate) for i in range(stack_size)])
        self.apply(init_weights)

    def forward(self, feature_matrix_batch: torch.Tensor):
        # feature_matrix_batch size = (S,N,I,D) where S=stack_size, N=batch number, I=members, D=member dimensionality
        output = selfCorrThreader(self.selfCorrelationStack, feature_matrix_batch)
        # output size = (S,N,I,D) where where S=stack_size, N=batch number, I=members, D=member dimensionality
        return output


class KMeansConvStack(nn.Module):
    """
    Utility for creating a vertical stack of K-Means-Conv modules
    """
    def __init__(self, stack_size: int, k: int, kmeansInit, n_init: int, sigma: nn.Module, F: nn.Module, W: nn.Module,
                 M: nn.Module, C: int, P: int):
        """
        Instantiate a vertical stack of K-Means-Conv modules. The arguments of this function, except stack_size,
        are deep copied for each K-Means-Conv module. The weights of each K-Means-Conv module are initialised
        separately using the xavier initialisation procedure.
        :param stack_size: number of K-Means-Conv modules
        :param k: the number of clusters. It is denoted as $K$ in the paper.
        :param kmeansInit: method for initialization of the k-means algorithm. "k-means++" is the default method
                           proposed in the paper
        :param n_init: number of time the k-means algorithm will be run with different centroid seeds.
        :param sigma: activation function. It is denoted as $\sigma$ in the paper. The sigmoid function is the
                      default activation function proposed in the paper.
        :param F: F is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{F}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons.
        :param W: W is a R^C -> R^(C x (C+P)) MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{W}$ in the paper. The output dimensionality of this MLP differs from the
                  one in the paper due to implementation reasons.
        :param M: M is a R^(C+P) -> R MLP, where C is the dimensionality of the points to be convolved and P is
                  the difference in dimensionality between the points to be convolved and the output points.
                  It is denoted as $\mathcal{M}$ in the paper.
        :param C: the dimensionality of the points to be convolved.
        :param P: the difference in dimensionality between the points to be convolved and the output points.
        """
        super().__init__()
        self.kmeansConvStack = nn.ModuleList([
            KMeansConv(k, kmeansInit, n_init, copy.deepcopy(sigma), copy.deepcopy(F), copy.deepcopy(W),
                       copy.deepcopy(M), C, P) for i in range(stack_size)])
        self.apply(init_weights)

    def forward(self, feature_matrix_batch: torch.Tensor):
        # feature_matrix_batch size = (S,N,I,D) where S=stack size, N=batch number, I=members, D=member dimensionality
        feature_matrix_batch, conv_feature_matrix_batch, cluster_index = kmeansConvThreader(self.kmeansConvStack,
                                                                                            feature_matrix_batch)
        # feature_matrix_batch size = (S,N,I,D) where where S=stack_size, N=batch number, I=members, D=member dimensionality
        # conv_feature_matrix_batch size = (S,N,I,D) where where S=stack_size, N=batch number, I=members, D=member dimensionality
        # cluster_index size = (S,M) where S=stack_size, M=N*I
        return feature_matrix_batch, conv_feature_matrix_batch, cluster_index


class AggregationStack(nn.Module):
    """
    Utility for creating a vertical stack of Aggregation modules
    """
    def __init__(self, stack_size: int, mlp1: nn.Module, mlp2: nn.Module):
        """
        Instantiate a vertical stack of Aggregation modules. The arguments of this function, except stack_size,
        are deep copied for each Aggregation module. The weights of each Aggregation module are initialised
        separately using the xavier initialisation procedure.
        :param stack_size: number of Aggregation modules
        :param mlp1: a R^(C+P) -> R^(C+P) MLP, where C is the dimensionality of the points prior to the convolution
                    operation and P is the difference in dimensionality between the points prior to the convolution
                    operation and the output points of the convolution operation. It is denoted as $f_1$ in the paper.
        :param mlp2: a R^(C+P) -> R^(C+P) MLP, where C is the dimensionality of the points prior to the convolution
                    operation and P is the difference in dimensionality between the points prior to the convolution
                    operation and the output points of the convolution operation. It is denoted as $f_2$ in the paper.
        """
        super().__init__()
        self.localAdaptFeatAggreStack = nn.ModuleList([Aggregation(copy.deepcopy(mlp1), copy.deepcopy(mlp2)) for i
                                                       in range(stack_size)])
        self.apply(init_weights)

    def forward(self, feature_matrix_batch: torch.Tensor, conv_feature_matrix_batch: torch.Tensor):
        # feature_matrix_batch size = (S,N,I,D) where S = stack size, N=batch number, I=cluster members, D=member dimensionality
        # conv_feature_matrix_batch size = (S,N,I,D') where S= stack size, N=batch number, I=cluster members, D'=C+P
        output = threader(self.localAdaptFeatAggreStack, feature_matrix_batch, conv_feature_matrix_batch)
        # output size = (S,N,I,D') where S= stack size, N=batch number, I=cluster members, D'=C+P
        return output


class MaxPoolStack(nn.Module):
    """
    Utility for creating a vertical stack of Max-pool modules
    """
    def __init__(self, stack_size: int, k: int):
        """
        Instantiate a vertical stack of Max-pool modules.
        :param stack_size: number of Max-pool modules
        :param k: number of clusters previously used in the K-Means-Conv modules
        """
        super().__init__()
        self.graphMaxPoolStack = nn.ModuleList([MaxPool(k) for i in range(stack_size)])
        self.apply(init_weights)

    def forward(self, feature_matrix_batch: torch.Tensor, cluster_index: torch.Tensor):
        # feature_matrix_batch size = (S,N,I,D) where S=stack size, N=batch number, I=members, D=member dimensionality
        # cluster_index size = (S,M) where S=stack size, M=N*I
        output = threader(self.graphMaxPoolStack, feature_matrix_batch, cluster_index)
        # output size = (S,N,K,D) where S=stack size, N=batch number, K=clusters, D=member dimensionality
        return output


def selfCorrThreader(modules, input_tensor):
    list_append = []
    threads = []
    for i, t in enumerate(input_tensor):
        threads.append(Thread(target=selfCorrAppender, args=(modules[i], t, list_append, i)))
    [t.start() for t in threads]
    [t.join() for t in threads]
    list_append.sort()
    list_append = list(map(lambda x: x[1], list_append))
    return torch.stack(list_append)


def selfCorrAppender(module, tensor, list_append, index):
    list_append.append((index, module(tensor)))


def kmeansConvThreader(modules, input_tensor):
    list1_append = []
    list2_append = []
    list3_append = []
    threads = []
    for i, t in enumerate(input_tensor):
        threads.append(
            Thread(target=kmeansAppender, args=(modules[i], t, list1_append, list2_append, list3_append, i)))
    [t.start() for t in threads]
    [t.join() for t in threads]
    list1_append.sort()
    list2_append.sort()
    list3_append.sort()
    list1_append = list(map(lambda x: x[1], list1_append))
    list2_append = list(map(lambda x: x[1], list2_append))
    list3_append = list(map(lambda x: x[1], list3_append))
    return torch.stack(list1_append), torch.stack(list2_append), torch.stack(list3_append)


def kmeansAppender(module, input, list1_append, list2_append, list3_append, index):
    x, y, z = module(input)
    list1_append.append((index, x))
    list2_append.append((index, y))
    list3_append.append((index, z))


def threader(modules, input_tensor1, input_tensor2):
    list_append = []
    threads = []
    for i, t in enumerate(input_tensor1):
        threads.append(Thread(target=threaderAppender, args=(modules[i], t, input_tensor2[i], list_append, i)))
    [t.start() for t in threads]
    [t.join() for t in threads]
    list_append.sort()
    list_append = list(map(lambda x: x[1], list_append))
    return torch.stack(list_append)


def threaderAppender(module, t1, t2, list_append, index):
    list_append.append((index, module(t1, t2)))


class Classifier(nn.Module):
    """
    A point cloud classifier which uses vertical and horizontal stacks of Shrinking units for the feature detection
    process.
    """
    def __init__(self, shrinkingLayersStack: [ShrinkingUnitStack], mlp: nn.Module):
        """
        Instantiate a point cloud classifier object
        :param shrinkingLayersStack: list of vertical stacks of Shrinking units
        :param mlp: MLP used for label assignment
        """
        super().__init__()
        self.neuralNet = nn.Sequential(*shrinkingLayersStack)
        self.mlp = mlp

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # x should be None when starting with a point cloud with no features apart from the euclidean coordinates
        # pos size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = pos.unsqueeze(0)
        # feature_matrix_batch size = (1,N,I,D) where N=batch number, I=members, D=member dimensionality
        output = self.neuralNet(feature_matrix_batch)
        # output size = (S,N,D) where S= stack size, N=batch number, D'=member dimensionality
        output = torch.mean(output, dim=0)
        # output size = (N,D) where N=batch number, D'=member dimensionality
        return self.mlp(output)