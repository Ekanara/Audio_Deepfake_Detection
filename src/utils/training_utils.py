import torch.nn as nn
import math
import torch
import torch.nn.functional as F

def feature_extract(input, feature_extractor):
    with torch.no_grad():
      output = feature_extractor.forward_features(input)
    return output

def apply_reshape(input):
    flatten_input = torch.flatten(input, start_dim=2, end_dim=3)
    reshaped_input = torch.stack([flatten_input, flatten_input, flatten_input], dim=1) 
    return reshaped_input

class ASoftmaxLoss(nn.Module):
  def __init__(self, embed_dim, n_classes , m=2.5, s=30):
    super().__init__()
    self.n_classes = n_classes
    self.embed_dim = embed_dim
    self.m = m
    self.s = s

    self.weight = nn.Parameter(torch.randn(n_classes, embed_dim))
    nn.init.xavier_normal_(self.weight)

  def phi(self, theta):
    k = torch.floor(self.m * theta / math.pi)
    sign = (-1.0) ** k
    phi_theta = sign * torch.cos(self.m * theta) - 2*k
    return phi_theta

  def forward(self, x, y=None):
    # x: [bs, embed_dim]
    # y: [bs,] or [bs, n_classes]
    assert x.ndim == 2
    assert y.ndim in [1, 2]

    # convert to one hot
    if(y.ndim == 1):
      y = F.one_hot(y, num_classes=self.n_classes)

    if(y is None):
      loss = torch.sum(-y * F.log_softmax(x, dim=1), dim=1).mean()
      return x, loss
    
    # normalize weight
    with torch.no_grad():
      self.weight.data = F.normalize(self.weight.data, p=2, dim=1)

    # normalize x
    x = F.normalize(x, p=2, dim=1)

    # compute cos(theta)
    # [bs, embed_dim] @ [embed_dim, n_classes]
    # [bs, n_classes]
    cos_theta = x @ self.weight.T 

    # Let phi(theta) = cos(theta) + delta
    # delta = phi(theta) - cos(theta)
    with torch.no_grad():
      theta = torch.arccos(torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6))
      phi_theta = self.phi(theta)
      phi_theta = phi_theta * y + cos_theta * (1-y)
      delta = phi_theta - cos_theta
    
    # compute phi_theta
    phi_theta = cos_theta - delta
    logit = self.s * phi_theta  

    # loss
    loss = torch.sum(-y * F.log_softmax(logit, dim=1), dim=1).mean()
    return logit, loss
  

class SINCERE(nn.Module):
  # Supervised Contrastive Loss
  def __init__(self, temperature=0.1, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.temperature = temperature

  def forward(self, x, y):
    '''
    x: shape (bs, embed_dim) - embeddings
    y: shape (bs, num_classes) - one-hot labels
    '''

    # Normalize embeddings
    x = F.normalize(x, dim=1)
    bs = x.size(0)

    # Compute similarity matrix
    sim_matrix = x @ x.T
    
    sim_matrix /= self.temperature
    
    # Compute exponential similarity matrix
    exp_sim_matrix = torch.exp(sim_matrix)

    # Positive mask
    mask = y @ y.T  # bs x bs
    
    pos_mask = mask - torch.eye(bs, device=x.device)
    n_pos = torch.sum(pos_mask, dim=1, keepdim=True)  # Pos count per sample

    # Negative 
    neg_mask = 1 - mask

    # Denominator (sum over negatives)
    denom = torch.sum(neg_mask * exp_sim_matrix, dim=1, keepdim=True) + 1e-8
    denom = denom + pos_mask * exp_sim_matrix

    # Compute probabilities
    prob = (pos_mask * exp_sim_matrix) / denom

    # Compute log probabilities
    log_prob = pos_mask * torch.log(prob + 1e-8)

    # Compute loss
    loss = -torch.sum(log_prob, dim=1) / (n_pos)
    loss = torch.mean(loss)  # Average over batch

    # print(loss)
    # exit()
    return loss
  
class CenterLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.center = None
  
  def set_center(self, center):
    self.center = center.detach()

  def forward(self, x, y, id_):
    """
      x: [bs, embed_dim]
      y: [bs, n_classes] - one hot
      id_: index of the class you want to apply center loss
    """
    # filter applied embeddigns
    x = x[y[:, id_] == 1]

    if(x.shape[0] == 0):
      return torch.tensor(0.0, device=x.device)

    if(self.center is None):
      center = torch.mean(x.detach(), dim=0)
      loss = torch.norm(x-center, dim=1) ** 2
    else:
      loss = torch.norm(x - self.center, dim=1) ** 2

    loss = loss.mean()
    return loss