import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, knn
from torch_scatter import scatter_mean

from .lovasz import StableBCELoss, lovasz_hinge, lovasz_softmax_flat

# Collection of Miscellaneous Loss Functions not yet implemented in Pytorch.


def unique_label_torch(label):
    _, label2, cts = torch.unique(label, return_inverse=True, return_counts=True)
    return label2, cts


def iou_batch(pred: torch.BoolTensor, labels: torch.BoolTensor, eps=0.0):
    """
    pred: N x C
    labels: N x C (one-hot)
    """
    intersection = (pred & labels).float().sum(0)
    union = (pred | labels).float().sum(0)
    iou = (intersection + eps) / (union + eps)  # We smooth our devision to avoid 0/0

    return iou.mean()


class LovaszHingeLoss(torch.nn.modules.loss._Loss):

    def __init__(self, reduction="none"):
        super(LovaszHingeLoss, self).__init__(reduction=reduction)

    def forward(self, logits, targets):
        num_clusters = targets.shape[1]
        return lovasz_hinge(
            logits.T.view(num_clusters, 1, -1), targets.T.view(num_clusters, 1, -1)
        )


class LovaszSoftmaxWithLogitsLoss(torch.nn.modules.loss._Loss):

    def __init__(self, reduction="none"):
        super(LovaszSoftmaxWithLogitsLoss, self).__init__(reduction=reduction)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, logits, targets):
        probs = self.softmax(logits)
        return lovasz_softmax_flat(probs, targets)


def find_cluster_means(features, labels):
    """
    For a given image, compute the centroids mu_c for each
    cluster label in the embedding space.
    Inputs:
        features (torch.Tensor): the pixel embeddings, shape=(N, d) where
        N is the number of pixels and d is the embedding space dimension.
        labels (torch.Tensor): ground-truth group labels, shape=(N, )
    Returns:
        cluster_means (torch.Tensor): (n_c, d) tensor where n_c is the number of
        distinct instances. Each row is a (1,d) vector corresponding to
        the coordinates of the i-th centroid.
    """
    centroids = scatter_mean(features, labels, dim=0)
    return centroids


def intra_cluster_loss(features, cluster_means, labels, margin=1.0):
    """
    Computes the intra-cluster loss between an embedding point cloud and
    a set of attractor points.

    <labels> must range between 0 to the number of <cluster_means>, otherwise
    the loss will be underestimated as <scatter_mean> zero value placeholders.
    """
    from torch_scatter import scatter_mean

    x = features[:, None, :]
    mu = cluster_means[None, :, :]
    l = torch.clamp(torch.norm(x - mu, dim=-1) - margin, min=0) ** 2
    l = torch.gather(l, 1, labels.view(-1, 1)).squeeze()

    if len(l.size()) and len(labels.size()):
        intra_loss = torch.mean(scatter_mean(l, labels))
        return intra_loss
    else:
        # print('intra_cluster_loss', l.size(), labels.size())
        return 0.0


def inter_cluster_loss(cluster_means, margin=0.2):
    inter_loss = 0.0
    n_clusters = len(cluster_means)
    if n_clusters < 2:
        # Inter-cluster loss is zero if there only one instance exists for
        # a semantic label.
        return 0.0
    else:
        indices = torch.triu_indices(cluster_means.shape[0], cluster_means.shape[0], 1)
        dist = squared_distances(cluster_means, cluster_means)
        return torch.pow(
            torch.clamp(2.0 * margin - dist[indices[0, :], indices[1, :]], min=0), 2
        ).mean()


def regularization_loss(cluster_means):
    return torch.mean(torch.norm(cluster_means, dim=1))


def margin_smoothing_loss(sigma, sigma_means, labels, margin=0):
    from torch_scatter import scatter_mean

    x = sigma[:, None]
    mu = sigma_means[None, :]
    l = torch.sqrt(torch.clamp(torch.abs(x - mu) - margin, min=0) ** 2 + 1e-6)
    l = torch.gather(l, 1, labels.view(-1, 1)).view(-1)
    loss = torch.mean(scatter_mean(l, labels))
    return loss


def get_probs(embeddings, margins, labels, eps=1e-6):
    from torch_scatter import scatter_mean

    device = embeddings.device
    n = labels.shape[0]
    centroids = find_cluster_means(embeddings, labels)
    sigma = scatter_mean(margins.squeeze(), labels)
    num_clusters = labels.unique().shape[0]

    # Compute spatial term
    em = embeddings[:, None, :]
    centroids = centroids[None, :, :]
    sqdists = ((em - centroids) ** 2).sum(-1)

    p = sqdists / (2.0 * sigma.view(1, -1) ** 2)
    p = torch.clamp(torch.exp(-p), min=eps, max=1 - eps)
    logits = torch.logit(p, eps=eps)
    eye = torch.eye(len(labels.unique()), dtype=torch.float32, device=device)
    targets = eye[labels]
    loss_tensor = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
    loss = loss_tensor.mean(dim=0).mean()
    with torch.no_grad():
        acc = iou_batch(logits > 0, targets.bool())
    smoothing_loss = margin_smoothing_loss(
        margins.squeeze(), sigma.detach(), labels, margin=0
    )
    p = torch.gather(p, 1, labels.view(-1, 1))
    return loss, smoothing_loss, p.squeeze(), acc


def multivariate_kernel(centroid, log_sigma, Lprime, eps=1e-8):
    def f(x):
        N = x.shape[0]
        L = torch.zeros(3, 3)
        tril_indices = torch.tril_indices(row=3, col=3, offset=-1)
        L[tril_indices[0], tril_indices[1]] = Lprime
        sigma = torch.exp(log_sigma) + eps
        L += torch.diag(sigma)
        cov = torch.matmul(L, L.T)
        dist = torch.matmul((x - centroid), torch.inverse(cov))
        dist = torch.bmm(dist.view(N, 1, -1), (x - centroid).view(N, -1, 1)).squeeze()
        probs = torch.exp(-dist)
        return probs

    return f


def bhattacharyya_distance_matrix(v1, v2, eps=1e-8):
    x1, s1 = v1[:, :3], v1[:, 3].view(-1)
    x2, s2 = v2[:, :3], v1[:, 3].view(-1)
    g1 = torch.ger(s1**2, 1.0 / (s2**2 + eps))
    g2 = g1.t()
    dist = squared_distances(x1.contiguous(), x2.contiguous())
    denom = 1.0 / (eps + s1.unsqueeze(1) ** 2 + s2**2)
    out = 0.25 * torch.log(0.25 * (g1 + g2 + 2)) + 0.25 * dist / denom
    return out


def squared_distances(v1, v2):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.pow(v2_2 - v1_2, 2).sum(2)


def bhattacharyya_coeff_matrix(v1, v2, eps=1e-6):
    x1, s1 = v1[:, :3], v1[:, 3].view(-1)
    x2, s2 = v2[:, :3], v1[:, 3].view(-1)
    g1 = torch.ger(s1**2, 1.0 / (s2**2 + eps))
    g2 = g1.t()
    dist = torch.cidst(x1.contiguous(), x2.contiguous())
    denom = 1.0 / (eps + s1.unsqueeze(1) ** 2 + s2**2)
    out = 0.25 * torch.log(0.25 * (g1 + g2 + 2)) + 0.25 * dist / denom
    out = torch.exp(-out)
    return out


def get_graphspice_logits(
    sp_emb,
    ft_emb,
    cov,
    groups,
    sp_centroids,
    ft_centroids,
    eps=0.001,
    compute_accuracy=True,
):

    device = sp_emb.device
    cov_means = find_cluster_means(cov, groups)

    # Compute spatial term
    sp_emb_tmp = sp_emb[:, None, :]
    sp_centroids_tmp = sp_centroids[None, :, :]
    sp_cov = torch.clamp(cov_means[:, 0][None, :], min=eps)
    sp_sqdists = ((sp_emb_tmp - sp_centroids_tmp) ** 2).sum(-1) / (sp_cov**2)

    # Compute feature term
    ft_emb_tmp = ft_emb[:, None, :]
    ft_centroids_tmp = ft_centroids[None, :, :]
    ft_cov = torch.clamp(cov_means[:, 0][None, :], min=eps)
    ft_sqdists = ((ft_emb_tmp - ft_centroids_tmp) ** 2).sum(-1) / (ft_cov**2)

    # Compute joint kernel score
    pvec = torch.exp(-sp_sqdists - ft_sqdists)
    # probs = (1-pvec).index_put((torch.arange(groups.shape[0]), groups),
    #     torch.gather(pvec, 1, groups.view(-1, 1)).squeeze())
    logits = torch.logit(pvec)

    acc = None
    eye = torch.eye(len(groups.unique()), dtype=torch.float32, device=device)
    targets = eye[groups]
    if compute_accuracy:
        acc = iou_batch(logits > 0, targets.bool())

    return logits, acc, targets
