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
        if y is None:
            loss = torch.sum(-F.log_softmax(x, dim=1), dim=1).mean()
            return x, loss

        assert y.ndim in [1, 2]
        if y.ndim == 1:
            y = F.one_hot(y, num_classes=self.n_classes)
        y = y.float()

        # save ‖x‖ BEFORE normalizing
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # [bs, 1]

        with torch.no_grad():
            self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        cos_theta = x @ self.weight.T  # [bs, n_classes]

        with torch.no_grad():
            theta     = torch.arccos(torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6))
            phi_theta = self.phi(theta)
            phi_theta = phi_theta * y + cos_theta * (1 - y)
            delta     = phi_theta - cos_theta

        phi_theta = cos_theta + delta

        # clamp x_norm to minimum s so logits never collapse
        scale = torch.clamp(x_norm, min=self.s)
        logit = scale * phi_theta

        loss = torch.sum(-y * F.log_softmax(logit, dim=1), dim=1).mean()
        return logit, loss
    
class ArcFaceLoss(nn.Module):
    def __init__(self, embed_dim, n_classes, m=0.5, s=30):
        super().__init__()
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.randn(n_classes, embed_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, y):
        with torch.no_grad():
            self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        cos_theta = x @ self.weight.T
        theta     = torch.arccos(cos_theta.clamp(-1+1e-6, 1-1e-6))

        if y.ndim == 1:
            y = F.one_hot(y, num_classes=cos_theta.shape[1])
        y = y.float()

        # Add margin only to GT class
        phi       = torch.cos(theta + self.m)
        logit     = self.s * (phi * y + cos_theta * (1 - y))
        loss      = torch.sum(-y * F.log_softmax(logit, dim=1), dim=1).mean()
        return logit, loss
class CenterContrastiveLoss(nn.Module):
    """
    Center Contrastive Loss (CCL) — arxiv 2308.00458
    
    L = -log [ e^(s*(c_y^T x - m) + 2λ*c_y^T x) ]
            / [ e^(s*(c_y^T x - m)) + Σ_{j≠y} e^(s*c_j^T x) ]

    Args:
        embed_dim : dimension of input embeddings
        n_classes : number of classes
        s         : hypersphere radius / temperature scale  (default 64)
        m         : cosine margin for positive center       (default 0.2)
        lambda_c  : weight for center loss term             (default 1.0)
    """
    def __init__(self, embed_dim, n_classes, s=64, m=0.2, lambda_c=1.0, real_w=4.0):
        super().__init__()
        self.s = s
        self.m = m
        self.lambda_c = lambda_c
        self.n_classes = n_classes
        self.real_w = real_w

        # Learnable class centers — shape [n_classes, embed_dim]
        self.centers = nn.Parameter(torch.randn(n_classes, embed_dim))
        nn.init.xavier_normal_(self.centers)

    def forward(self, x, labels):
        if labels.dim() > 1:
            labels = labels.argmax(dim=1).long()
        else:
            labels = labels.long()

        B = x.size(0)
        x_norm = F.normalize(x, p=2, dim=1)
        c_norm = F.normalize(self.centers, p=2, dim=1)

        cos_sim = x_norm @ c_norm.T                              # [B, 2]
        cos_pos = cos_sim[torch.arange(B), labels]               # [B]

        # Numerator exponent (Eq. 5 top)
        num_exp = self.s * (cos_pos - self.m) + 2 * self.lambda_c * cos_pos

        # Denominator: positive gets s*(cos_pos - m), negatives get s*cos_sim
        denom_logits = self.s * cos_sim.clone()
        denom_logits[torch.arange(B), labels] = self.s * (cos_pos - self.m)

        log_denom = torch.logsumexp(denom_logits, dim=1)         # [B]
        per_sample_loss = -(num_exp - log_denom)                 # [B]

        # --- Class balancing: upweight real (label=1), downweight fake (label=0) ---
        class_weights = torch.where(labels == 1,
                                    torch.tensor(self.real_w, device=x.device),
                                    torch.tensor(1.0, device=x.device))
        # Normalize so total weight = B (keeps loss scale stable)
        class_weights = class_weights * B / class_weights.sum()

        loss = torch.mean(class_weights * per_sample_loss)

        return loss, cos_sim
      
# class ASoftmaxLoss(nn.Module):
#     def __init__(self, embed_dim, n_classes, m=2, s=30, eps=1e-6):
#         super().__init__()
#         self.n_classes = n_classes
#         self.embed_dim = embed_dim
#         self.m = m
#         self.s = s
#         self.eps = eps

#         self.weight = nn.Parameter(torch.randn(n_classes, embed_dim))
#         nn.init.xavier_normal_(self.weight)

#     def phi(self, theta):
#         k = torch.floor(self.m * theta / math.pi)
#         sign = torch.where(
#             (k.long() % 2) == 0,
#             torch.ones_like(theta),
#             -torch.ones_like(theta),
#         )
#         return sign * torch.cos(self.m * theta) - 2 * k

#     def forward(self, x, y):
#         assert x.ndim == 2

#         if y.ndim == 2:
#             y = y.argmax(dim=1)

#         y_onehot = F.one_hot(y, num_classes=self.n_classes).float()

#         x = F.normalize(x, p=2, dim=1)
#         W = F.normalize(self.weight, p=2, dim=1)

#         cos_theta = x @ W.T
#         cos_theta = torch.clamp(cos_theta, -1 + self.eps, 1 - self.eps)

#         theta = torch.acos(cos_theta)
#         phi_theta = self.phi(theta)

#         logits = self.s * (y_onehot * phi_theta + (1.0 - y_onehot) * cos_theta)
#         loss = F.cross_entropy(logits, y)

#         return logits, loss



# class ASoftmaxLoss(nn.Module):
#     """
#     Angular Softmax (A-Softmax / SphereFace) loss.

#     This follows the formulation used in:
#     "Angular Softmax Loss for End-to-end Speaker Verification"
#     and the original SphereFace definition.

#     Logit for non-target classes:
#         ||x|| * cos(theta_j)

#     Logit for target class:
#         ||x|| * phi(theta_y)

#     where
#         phi(theta) = (-1)^k * cos(m * theta) - 2k,
#         theta in [k*pi/m, (k+1)*pi/m], k in [0, m-1]

#     Args:
#         in_features: embedding dimension
#         out_features: number of training speakers/classes
#         m: angular margin multiplier, should be integer >= 2
#         eps: clamp epsilon for acos stability
#     """
#     def __init__(self, embed_dim: int, n_classes: int, m: int = 4, eps: float = 1e-6, s=30):
#         super().__init__()
#         if not isinstance(m, int) or m < 2:
#             raise ValueError("m must be an integer >= 2 for A-Softmax.")

#         self.in_features = embed_dim
#         self.out_features = n_classes
#         self.m = m
#         self.eps = eps

#         self.weight = nn.Parameter(torch.empty(n_classes, embed_dim))
#         nn.init.xavier_uniform_(self.weight)

#     def _phi_theta(self, cos_theta: torch.Tensor) -> torch.Tensor:
#         """
#         Compute phi(theta) from cos(theta) using the piecewise SphereFace rule.
#         """
#         cos_theta = cos_theta.clamp(-1.0 + self.eps, 1.0 - self.eps)
#         theta = torch.acos(cos_theta)

#         k = torch.floor(self.m * theta / math.pi)
#         sign = torch.where((k.long() % 2) == 0,
#                            torch.ones_like(cos_theta),
#                            -torch.ones_like(cos_theta))

#         phi_theta = sign * torch.cos(self.m * theta) - 2.0 * k
#         return phi_theta

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         """
#         Args:
#             x: [batch, in_features] speaker embeddings
#             target: [batch] class indices

#         Returns:
#             logits: [batch, out_features]
#             loss: scalar cross-entropy over A-Softmax logits
#         """
#         if x.ndim != 2:
#             raise ValueError(f"x must have shape [batch, in_features], got {x.shape}")
#         if y.ndim != 1:
#             raise ValueError(f"target must have shape [batch], got {y.shape}")
#         if x.size(1) != self.in_features:
#             raise ValueError(f"Expected x.size(1) == {self.in_features}, got {x.size(1)}")

#         # Normalize weights, as described in the paper.
#         W = F.normalize(self.weight, p=2, dim=1)   # [C, D]

#         # Keep feature norm ||x||.
#         x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp_min(self.eps)  # [B, 1]
#         x_unit = x / x_norm                                                    # [B, D]

#         # cos(theta_j) for every class.
#         cos_theta = torch.matmul(x_unit, W.t()).clamp(-1.0 + self.eps, 1.0 - self.eps)  # [B, C]

#         # phi(theta_j)
#         phi_theta = self._phi_theta(cos_theta)  # [B, C]

#         # One-hot target mask
#         one_hot = F.one_hot(y, num_classes=self.out_features).float()

#         # Replace only target-class cosine with phi(theta)
#         logits_cos = cos_theta * (1.0 - one_hot) + phi_theta * one_hot

#         # Multiply by ||x||, matching the paper's formula
#         logits = logits_cos * x_norm

#         loss = F.cross_entropy(logits, y)
#         return logits, loss
    
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