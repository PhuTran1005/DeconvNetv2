import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_loss(name):
    if name is None:
        name = 'dice_loss'
    return {
        'dice_loss': BCEWithLogitsLoss,
        'cross_entropy': CrossEntropyLoss,
        'soft_dice': SoftDiceLoss,
        'focal': FocalLoss,
        'log_cosh_dice': LogCoshDiceLoss,
        'crf': CRFLoss
    }[name]

# Implementation CRF loss function
def dense_crf(image, unary, num_classes, sigma_rgb, sigma_xy):
    """
    Compute the pairwise potentials using the dense CRF
    Args:
        image (torch.Tensor): The input image
        unary (torch.Tensor): The unary potentials
        num_classes (int): The number of classes
        sigma_rgb (float): The parameter for the RGB difference
        sigma_xy (float): The parameter for the spatial difference
    Returns:
        pairwise (torch.Tensor): The computed pairwise potentials
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian

    # Convert the image to numpy array
    image = image.squeeze().permute(1, 2, 0).numpy()
    h, w, c = image.shape

    # Create the dense CRF object
    d = dcrf.DenseCRF2D(w, h, num_classes)

    # Set the unary potentials
    U = unary.cpu().numpy()
    d.setUnaryEnergy(U)

    # Create the pairwise potentials
    # Spatial pairwise potential
    pairwise_energy = create_pairwise_gaussian(sdims=(sigma_xy, sigma_xy), shape=(h, w))
    d.addPairwiseEnergy(pairwise_energy, compat=10)

    # RGB pairwise potential
    pairwise_energy = create_pairwise_bilateral(sdims=(sigma_rgb, sigma_rgb), schan=(255, 255, 255), img=image, chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=10)

    # Inference to obtain the pairwise potentials
    Q = d.inference(5)
    pairwise = torch.from_numpy(Q).view(1, num_classes, h, w).to(unary.device)

    return pairwise

class CRFLoss(nn.Module):
    def __init__(self):
        super(CRFLoss, self).__init__()

    def forward(logits, labels, image, num_classes=21, sigma_rgb=3.0, sigma_xy=64.0):
        """
        Compute the CRF loss for semantic segmentation task using PyTorch
        Args:
            logits (torch.Tensor): The logits output of the segmentation model
            labels (torch.Tensor): The ground truth labels
            image (torch.Tensor): The input image
            num_classes (int): The number of classes
            sigma_rgb (float): The parameter for the RGB difference
            sigma_xy (float): The parameter for the spatial difference
        Returns:
            crf_loss (torch.Tensor): The computed CRF loss
        """
        # Compute the unary potentials from the logits
        unary = -F.log_softmax(logits, dim=1)

        # Compute the pairwise potentials from the CRF
        pairwise = dense_crf(image, unary, num_classes, sigma_rgb, sigma_xy)

        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)

        # Combine the unary and pairwise potentials
        crf_loss = ce_loss + pairwise

        return crf_loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, targets, inputs, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class SoftDiceLoss(nn.Module):
    def __init__(self, reduction='none', use_softmax=True):
        """
        Args:
            use_softmax: Set it to False when use the function for testing purpose
        """
        super(SoftDiceLoss, self).__init__()
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, output, target, epsilon=1e-6):
        """
        References:
        JeremyJordan's Implementation
        https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
        Paper related to this function:
        Formula for binary segmentation case - A survey of loss functions for semantic segmentation
        https://arxiv.org/pdf/2006.14822.pdf
        Formula for multiclass segmentation cases - Segmentation of Head and Neck Organs at Risk Using CNN with Batch
        Dice Loss
        https://arxiv.org/pdf/1812.02427.pdf
        Args:
            output: Tensor shape (N, N_Class, H, W), torch.float
            target: Tensor shape (N, H, W)
            epsilon: Use this term to avoid undefined edge case
        Returns:
        """
        num_classes = output.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == one_hot_target.shape
        if self.reduction == 'none':
            return 1.0 - soft_dice_loss(output, one_hot_target)
        elif self.reduction == 'mean':
            return 1.0 - torch.mean(soft_dice_loss(output, one_hot_target))
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, target, input):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class LogCoshDiceLoss(nn.Module):
    """
    L_{lc-dce} = log(cosh(DiceLoss)
    """
    def __init__(self, use_softmax=True):
        super(LogCoshDiceLoss, self).__init__()
        self.use_softmax = use_softmax

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == one_hot_target.shape
        numerator = 2. * torch.sum(output * one_hot_target, dim=(-2, -1))  # Shape [batch, n_classes]
        denominator = torch.sum(output + one_hot_target, dim=(-2, -1))
        return torch.log(torch.cosh(1 - torch.mean((numerator + epsilon) / (denominator + epsilon))))