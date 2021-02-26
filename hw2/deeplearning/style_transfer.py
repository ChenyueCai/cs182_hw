import numpy as np

import torch
import torch.nn.functional as F


def content_loss(content_weight, content_current, content_target):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    _, C_l, H_l, W_l = content_current.shape
    content_current = content_current.view((-1, H_l*W_l))
    content_target = content_target.view((-1, H_l*W_l))
    return content_weight * ((content_current - content_target) ** 2).sum()
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Variable of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Variable of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    N, C, H, W = features.shape
    gram = torch.zeros((N, C, C))
    features = features.view((N, C,-1))
    for i in range (N):
        gram[i] = torch.mm(features[i], features[i].transpose(1,0))  
        
    if normalize == True:
        gram = gram / (H * W * C)
    return gram
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Variable giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Variable holding a scalar giving the style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    style_loss = torch.tensor(0, dtype=torch.float64)
    for i in range(len(style_layers)):
        index = style_layers[i]
        diff = ((gram_matrix(feats[index]) - style_targets[i]) ** 2).sum()
        style_loss += style_weights[i] * diff
    return style_loss
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    _,_, H, W = img.shape
    diff = ((img[:, :, 0:H, 1:W] - img[:, :, 0: H, 0: W-1]) ** 2).sum() + ((img[:, :, 1:H, 0:W] - img[:, :, 0: H-1, 0:W]) **2).sum()
    return diff * tv_weight
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
