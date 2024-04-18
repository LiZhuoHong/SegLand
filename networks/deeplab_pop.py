import torch
import torch.nn as nn
from torch.nn import functional as F
import random

from .backbones import get_backbone

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))
        concat_ch = out_ch * (len(rates) + 2)
        self.add_module("fc", _ConvBnReLU(concat_ch, out_ch, 1, 1, 0, 1))

    def forward(self, x):
        m = torch.cat([stage(x) for stage in self.stages.children()], dim=1)
        out = self.fc(m)
        return out

class GFSS_Model(nn.Module):
    '''
        Segmenter for Generalized Few-shot Semantic Segmentation
    '''
    def __init__(self, n_base, criterion=None, norm_layer=nn.BatchNorm2d, use_base=True, is_ft=False, n_novel=0, **kwargs):
        super(GFSS_Model, self).__init__()
        d_model = 256
        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)
        self.decoder = _ASPP(2048, d_model, rates=[6, 12, 18])
        self.classifier = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, 1, kernel_size=1, bias=False)
        )

        if is_ft:
            self.base_emb = nn.Parameter(torch.zeros(n_base, d_model), requires_grad=False)
            self.novel_emb = nn.Parameter(torch.zeros(n_novel, d_model), requires_grad=True)
            self.classifier_n = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_model, 1, kernel_size=1, bias=False)
            )
            nn.init.orthogonal_(self.novel_emb)
            self.ft_freeze()
        else:
            self.base_emb = nn.Parameter(torch.zeros(n_base, d_model), requires_grad=True)
            nn.init.orthogonal_(self.base_emb)
            self.novel_emb = None
        self.n_novel = n_novel
        self.use_base = use_base
        self.is_ft = is_ft
        self.criterion = criterion
        self.n_base = n_base

    def init_cls_n(self):
        for param_q, param_k in zip(self.classifier.parameters(), self.classifier_n.parameters()):
            param_k.data.copy_(param_q.data)  # initialize

    def train_mode(self):
        self.train()
        # to prevent BN from learning data statistics with exponential averaging
        self.backbone.eval()
        self.decoder.eval()
        # self.classifier.eval()

    def ft_freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    @torch.cuda.amp.autocast(enabled=False)
    def orthogonal_decompose(self, feats, bases_b, bases_n=None):
        '''
            feats: [BxCxN]
            bases_b: [1xKxC]
            bases_n: [1xKxC]
            ---
            out_fg:   [BxKxCxN]
            out_bg:   [Bx1xCxN]
        '''
        q = feats.to(torch.float) # [BxCxN]
        s1 = F.normalize(bases_b.to(torch.float), p=2, dim=-1) # [1xKxC]
        # q = feats # [BxCxN]
        # s1 = F.normalize(bases_b, p=2, dim=-1) # [1xKxC]

        proj1 = torch.matmul(s1, q) # [BxKxN]
        out_fg_b = proj1.unsqueeze(2) * s1.unsqueeze(-1) # [BxKxCxN]
        out_bg = q - out_fg_b.sum(1) # [BxCxN]
        if bases_n is not None:
            s2 = F.normalize(bases_n, p=2, dim=-1) # [1xKxC]
            proj2 = torch.matmul(s2, q) # [BxKxN]
            out_fg_n = proj2.unsqueeze(2) * s2.unsqueeze(-1) # [BxKxCxN]
            out_bg = out_bg - out_fg_n.sum(1)# [BxCxN]
            return out_fg_b, out_fg_n, out_bg.unsqueeze(1)
        else:
            out_fg = out_fg_b
            return out_fg, out_bg.unsqueeze(1)

    def forward(self, img, mask=None, img_b=None, mask_b=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        if self.is_ft:
            if self.training:
                return self.forward_novel(img, mask, img_b, mask_b)
            else:
                return self.forward_all(img, mask)
        else:
            return self.forward_base(img, mask)

    def forward_all(self, img, mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        features = self.backbone.base_forward(img)
        features = self.decoder(features) # [BxCxhxw]
        B, C, h, w = features.shape

        base_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]
        novel_emb = self.novel_emb.unsqueeze(0) # [1xKnovelxC]

        out_fg_b, out_fg_n, feats_bg = self.orthogonal_decompose(features.flatten(2), base_emb, novel_emb)

        out_fg_b = out_fg_b.contiguous().view(B*self.n_base, C, h, w) # [(BxKb)xCxhxw]
        preds1 = self.classifier(out_fg_b) # [(BxKb)x1xhxw]
        preds1 = preds1.view(B, self.n_base, h, w) # [BxKbxhxw]

        feats_n = torch.cat([feats_bg, out_fg_n], dim=1) # [Bx(1+Kn)xCxN]
        feats_n = feats_n.contiguous().view(B*(self.n_novel+1), C, h, w) # [(Bx(1+Kn))xCxhxw]
        preds2 = self.classifier_n(feats_n) # [(Bx(1+Kn))x1xhxw]
        preds2 = preds2.view(B, self.n_novel+1, h, w) # [Bx(1+Kn)xhxw]

        preds = torch.cat([preds2[:,0].unsqueeze(1), preds1, preds2[:,1:]], dim=1)
        return preds

    def forward_base(self, img, mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        # x4, x3, _  = self.backbone.base_forward(img, return_list=True)
        features = self.backbone.base_forward(img)
        features = self.decoder(features) # [BxCxhxw]

        B, C, h, w = features.shape
        cls_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]

        n_class = 1 + cls_emb.shape[1]
        features = features.flatten(2) # [BxCxN]
        feats_fg, feats_bg = self.orthogonal_decompose(features, cls_emb)

        feats_all = torch.cat([feats_bg, feats_fg], dim=1) # [Bx(1+K)xCxN]
        feats_all = feats_all.contiguous().view(B*n_class, C, h, w) # [(Bx(1+K))xCxhxw]

        preds = self.classifier(feats_all) # [(Bx(1+K))x1xhxw]
        preds = preds.view(B, n_class, h, w) # [Bx(1+K)xhxw]

        if self.criterion is not None and mask is not None:
            cls_emb = F.normalize(cls_emb, p=2, dim=-1).squeeze(0) # [KbasexC]
            proto_sim = torch.matmul(cls_emb, cls_emb.t()) # [KbasexKbase]
            return self.criterion(preds, mask, proto_sim=proto_sim)
        else:
            return preds

    def forward_novel(self, img, mask, img_b, mask_b):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        # with torch.no_grad():
        img_full = torch.cat([img, img_b], dim=0)
        features_full = self.backbone.base_forward(img_full)
        features_full = self.decoder(features_full) # [BxCxhxw]

        B, C, h, w = features_full.shape

        base_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]
        novel_emb = self.novel_emb.unsqueeze(0) # [1xKnovelxC]

        n_class = 1 + base_emb.shape[1] + novel_emb.shape[1]
        features_full = features_full.flatten(2) # [BxCxN]
        out_fg_b, out_fg_n, feats_bg = self.orthogonal_decompose(features_full, base_emb, novel_emb)

        out_fg_b = out_fg_b.reshape(B*self.n_base, C, h, w) # [(BxKb)xCxhxw]
        preds1 = self.classifier(out_fg_b) # [(BxKb)x1xhxw]
        preds1 = preds1.view(B, self.n_base, h, w) # [BxKbxhxw]

        feats_n = torch.cat([feats_bg, out_fg_n], dim=1) # [Bx(1+Kn)xCxN]
        feats_n = feats_n.reshape(B*(1+self.n_novel), C, h, w) # [(Bx(1+Kn))xCxhxw]
        preds2 = self.classifier_n(feats_n) # [(Bx(1+Kn))x1xhxw]
        preds2 = preds2.view(B, 1+self.n_novel, h, w) # [Bx(1+Kn)xhxw]
        
        preds = torch.cat([preds2[:,0].unsqueeze(1), preds1, preds2[:,1:]], dim=1)

        mask_new = []
        for b in range(B//2):
            bg_mask = mask_b[b] == 0
            bg_out = preds2[B//2+b] # [(1+Kn)xhxw]
            bg_out = F.interpolate(input=bg_out.unsqueeze(0), size=mask_b[b].shape, mode='bilinear', align_corners=True)

            bg_idx = torch.argmax(bg_out.squeeze(0), dim=0) # [hxw]
            bg_idx[bg_idx>0] += self.n_base
            mask_b[b][bg_mask] = bg_idx[bg_mask]
            mask_new.append(mask_b[b])
        mask_new = torch.stack(mask_new, dim=0)

        if self.criterion is not None and mask is not None:
            with torch.cuda.amp.autocast(enabled=False):
                mask_all = torch.cat([mask, mask_new], dim=0)
                novel_emb = F.normalize(novel_emb.to(torch.float), p=2, dim=-1) # [1xKnovelxC]
                novel_emb = novel_emb.reshape(-1, C) # [KnxC]
                all_emb = torch.cat([novel_emb, F.normalize(base_emb.to(torch.float).squeeze(0), p=2, dim=-1)], dim=0) # [((Kn+Kb)xC]
                proto_sim = torch.matmul(novel_emb, all_emb.t()) # [Knx(Kn+Kb)]

                return self.criterion(preds.to(torch.float), mask_all, is_ft=True, proto_sim=proto_sim)
        else:
            return preds
