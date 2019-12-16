import babelfish as bf
import torch as T
from tqdm import tqdm


def backprop_gradient_mask(model, batch, grad):
    """For one batch, backprop gradient mask.
    
    grad should be Z x H x W"""    
    (X,Y) = batch
    model.zero_grad()
    # TODO add general pattern to deal with DataParallel
    # model = conv_model.module
    xb = X['brain'].cuda(non_blocking=True)
    batch_size = len(xb)
    grad = grad[None].cuda(non_blocking=True).expand(batch_size,-1,-1,-1)
    xb.requires_grad=True
    output = model(xb)
    X_pred = output["prev"]
    Y_pred = output["pred"]
    mean = output["mean"]
    logvar = output["logvar"]
    # TODO is this what I want?? double check
    T.autograd.backward(Y_pred,grad)
    xb_grad = xb.grad.detach()
    return xb_grad

def avg_backprop_gradient_mask(model, dataloader, grad, progress=False):
    """For all batch, backprop gradient mask.
    
    grad should be Z x H x W"""    
    if progress:
        dataloader = tqdm(dataloader)
    else:
        dataloader = dataloader
    res = []
    for batch in dataloader:
        res.append(backprop_gradient_mask(model, batch, grad))
    return T.cat(res).max(0)[0]
