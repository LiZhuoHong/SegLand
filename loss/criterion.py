from unicodedata import normalize
import torch.nn as nn
import torch
from torch.nn import functional as F

class CELoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, pred, target, aux_pred=None):
        '''
            pred      : [BxKxhxw]
            target    : [BxHxW]
        '''
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)
        main_loss = self.seg_criterion(scale_pred, target)
        if aux_pred is not None:
            scale_aux_pred = F.interpolate(input=aux_pred, size=(h, w), mode='bilinear', align_corners=True)
            aux_loss = self.seg_criterion(scale_aux_pred, target)
            total_loss = main_loss + 0.4 * aux_loss
            loss_dict = {'total_loss':total_loss, 'main_loss':main_loss, 'aux_loss':aux_loss}
        else:
            loss_dict = {'total_loss':main_loss}
        return loss_dict

class OrthLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(OrthLoss, self).__init__()
        self.ignore_index = ignore_index
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        # self.seg_criterion = MulticlassHybridLoss(ignore_index=ignore_index, reduction=reduction)
        self.w = 10.0

    def get_orth_loss(self, proto_sim, is_ft=False):
        '''
            protos:   : [K1xK2] K1 <= K2
        '''
        eye_sim = torch.triu(torch.ones_like(proto_sim), diagonal=1)
        loss_orth = torch.abs(proto_sim[eye_sim == 1]).mean()
        return loss_orth

    def forward(self, preds, target, is_ft=False, proto_sim=None, aux_preds=None):
        '''
            pred      : [BxKaxhxw]
            target    : [BxHxW]
            proto_sim:   : [K1xK2]
        '''
        scale_pre = F.interpolate(input=preds, size=target.shape[1:], mode='bilinear', align_corners=True)
        seg_loss = self.seg_criterion(scale_pre, target)

        orth_loss = self.get_orth_loss(proto_sim, is_ft=is_ft)

        if aux_preds is not None:
            scale_aux_pred = F.interpolate(input=aux_preds, size=target.shape[1:], mode='bilinear', align_corners=True)
            aux_loss = self.seg_criterion(scale_aux_pred, target)
            total_loss = seg_loss + orth_loss * self.w + 0.4 * aux_loss
            loss_dict = {'total_loss':total_loss, 'seg_loss':seg_loss, 'aux_loss':aux_loss, 'orth_loss':orth_loss}
        else:
            total_loss = seg_loss + orth_loss * self.w
            loss_dict = {'total_loss':total_loss, 'seg_loss':seg_loss, 'orth_loss':orth_loss}

        return loss_dict
 
# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
 
#     def	forward(self, input, target):

#         N = target.size(0) # batch size
#         smooth = 1.0
 
#         input_flat = input.view(N, -1)
#         target_flat = target.view(N, -1)
 
#         intersection = input_flat * target_flat
 
#         loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
#         loss = 1 - loss
 
#         return loss

# class MulticlassCELoss(nn.Module):
#     def __init__(self, weight=None, ignore_index=None, reduction='none'):
#         super(MulticlassCELoss, self).__init__()
#         if isinstance(ignore_index, int):
#             self.ignore_index = [ignore_index]
#         elif isinstance(ignore_index, list) or ignore_index is None:
#             self.ignore_index = ignore_index
#         else:
#             raise TypeError("Wrong type for ignore index, which should be int or list or None")
#         if isinstance(weight, list) or weight is None:
#             self.weight = weight
#         else:
#             raise TypeError("Wrong type for weight, which should be list or None")
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
#         self.reduction = reduction

#     def forward(self, input, target):
#         # labels.shape: [b,]
#         C = input.shape[1]
#         logpt = F.log_softmax(input, dim=1)
        
#         if len(target.size()) <= 3:
#             if self.ignore_index is not None:
#                 mask = torch.zeros_like(target)
#                 for idx in self.ignore_index:
#                     m = torch.where(target == idx, torch.zeros_like(target), torch.ones_like(target))
#                     mask += m
#                 mask = torch.where(mask > 0, torch.ones_like(target), torch.zeros_like(target))
#                 target *= mask
#             target = F.one_hot(target, C).permute(0, 3, 1, 2)
#             target *= mask.unsqueeze(1)
        
#         if self.weight is None:
#             weight = torch.ones(logpt.shape[1]).to(target.device) #uniform weights for all classes
#         else:
#             weight = torch.tensor(self.weight).to(target.device)
        
#         for i in range(len(logpt.shape)):
#             if i != 1:
#                 weight = torch.unsqueeze(weight, dim=i)
        
#         s_weight = weight * target
#         for i in range(target.shape[1]):
#             if self.ignore_index is not None and i in self.ignore_index:
#                 target[:,i] = - target[:,i]
#                 s_weight[:,i] = 0
#         s_weight = s_weight.sum(1)

#         loss = -1 * weight * logpt * target
#         loss = loss.sum(1)

#         if self.reduction == 'none':
#             return torch.where(loss > 0, loss, torch.zeros_like(loss).to(loss.device))
#         elif self.reduction == 'mean':
#             if s_weight.sum() == 0:
#                 return loss[torch.where(loss > 0)].sum()
#             else:
#                 return loss[torch.where(loss > 0)].sum() / s_weight[torch.where(loss > 0)].sum()
#         elif self.reduction == 'sum':
#             return loss[torch.where(loss > 0)].sum()
#         else:
#             raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
 
# class MulticlassDiceLoss(nn.Module):
#     def __init__(self, weight=None, ignore_index=None, reduction='mean'):
#         # ignore index can be int or list
#         super(MulticlassDiceLoss, self).__init__()
#         if isinstance(weight, list) or weight is None:
#             self.weight = weight
#         else:
#             raise TypeError("Wrong type for weight, which should be list or None")
#         if isinstance(ignore_index, int):
#             self.ignore_index = [ignore_index]
#         elif isinstance(ignore_index, list) or ignore_index is None:
#             self.ignore_index = ignore_index
#         else:
#             raise TypeError("Wrong type for ignore index, which should be int or list or None")
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
#         self.reduction = reduction

#     def forward(self, input, target):
#         C = input.shape[1]
#         max = torch.max(input, dim=1).values.unsqueeze(dim=1)
#         input_s = F.softmax(input - max, dim=1)

#         if len(target.size()) <= 3:
#             if self.ignore_index is not None:
#                 mask = torch.zeros_like(target)
#                 for idx in self.ignore_index:
#                     m = torch.where(target == idx, torch.zeros_like(target), torch.ones_like(target))
#                     mask += m
#                 mask = torch.where(mask > 0, torch.ones_like(target), torch.zeros_like(target))
#                 target *= mask
#             target = F.one_hot(target, C).permute(0, 3, 1, 2)
#             target *= mask.unsqueeze(1)
 
#         dice = DiceLoss()
#         totalLoss = 0

#         if self.weight is None:
#             self.weight = torch.ones(C).to(target.device) #uniform weights for all classes
#         else:
#             self.weight = torch.tensor(self.weight).to(target.device)
 
#         for i in range(C):
#             if self.ignore_index is None or i not in self.ignore_index:
#                 diceLoss = dice(input_s[:,i], target[:,i])
#                 if self.weight is not None:
#                     diceLoss *= self.weight[i]
#                 totalLoss += diceLoss
        
#         if self.reduction == 'none':
#             return totalLoss
#         elif self.reduction == 'mean':
#             return totalLoss.mean()
#         elif self.reduction == 'sum':
#             return totalLoss.sum()
#         else:
#             raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")

# class MulticlassHybridLoss(nn.Module):
#     def __init__(self, weight=None, ignore_index=None, reduction='mean'):
#         # ignore index can be int or list
#         super(MulticlassHybridLoss, self).__init__()
#         self.ce = MulticlassCELoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
#         self.dice = MulticlassDiceLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    
#     def forward(self, input, target):
#         l1 = self.ce(input, target)
#         l2 = self.dice(input, target)
#         return l1 + l2