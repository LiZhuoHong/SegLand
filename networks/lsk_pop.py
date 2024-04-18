import torch
import torch.nn as nn
from torch.nn import functional as F

from .backbones import get_backbone

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer, act_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, norm_layer, act_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        relu = act_layer()
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class UperNet_Decoder(nn.Module):
    def __init__(self, filters, dim=512, ppm_size=(1, 2, 3, 6)):
        super(UperNet_Decoder, self).__init__()
        self.dim = dim
        self.psp = PSPModule(filters[-1], self.dim, sizes=ppm_size)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in filters[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.dim, 3, padding=1),
                nn.BatchNorm2d(self.dim),
                nn.ReLU()
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(self.dim, self.dim, 3, padding=1),
                nn.BatchNorm2d(self.dim),
                nn.ReLU()
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        self.fpn_bottleneck = nn.Sequential(
                nn.Conv2d(len(filters) * self.dim, self.dim, 3, padding=1),
                nn.BatchNorm2d(self.dim),
                nn.ReLU()
            )
        
    def forward(self, x):

        # build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp(x[-1]))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=True)
        
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=True)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        return output

class SimpleDecoder(nn.Module):
    def __init__(self, filters, dim=512):
        super(SimpleDecoder, self).__init__()
        self.dim = dim
        self.semantic_embedding = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                                nn.Conv2d(filters[-1], self.dim, kernel_size=1, bias=False),
                                                nn.Sigmoid())

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(filters)):
            l_conv = nn.Sequential(
                nn.Conv2d(filters[i], self.dim, 3, padding=1),
                nn.BatchNorm2d(self.dim),
                nn.ReLU()
            )
            head_length = max(
                1,
                int(torch.log2(torch.tensor(filters[i])) - torch.log2(torch.tensor(filters[0]))))
            scale_head = []
            for _ in range(head_length):
                scale_head.append(
                    nn.Sequential(
                        nn.Conv2d(self.dim, self.dim, 3, padding=1),
                        nn.BatchNorm2d(self.dim),
                        nn.ReLU(inplace=True)
                    )
                )
                if filters[i] != filters[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=True))
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(nn.Sequential(*scale_head))
    
    def forward(self, x):

        # build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        semantic_embedding = self.semantic_embedding(x[-1])
        laterals = [l * semantic_embedding + l for l in laterals]

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(len(x))
        ]

        fpn_outs = [
            F.interpolate(f, size=x[0].shape[-2:], mode='bilinear', align_corners=True) if f.shape[-2:] != x[0].shape[-2:] else f 
            for f in fpn_outs
        ]
        output = torch.stack(fpn_outs, dim=-1).sum(-1)
        return output


class GFSS_Model(nn.Module):
    '''
        Segmenter for Generalized Few-shot Semantic Segmentation
    '''
    def __init__(self, n_base, criterion=None, norm_layer=nn.BatchNorm2d, use_base=True, is_ft=False, n_novel=0, **kwargs):
        super(GFSS_Model, self).__init__()
        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)
        d_model = 192
        self.decoder = SimpleDecoder(self.backbone.get_filters(), d_model)
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
        features = self.backbone(img)
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
        features = self.backbone(img)
        features = self.decoder(features) # [BxCxhxw]

        B, C, h, w = features.shape
        cls_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]
        # cls_emb = torch.linalg.qr(self.base_emb.transpose(-2, -1))[0].transpose(-2, -1).unsqueeze(0) # [1xKbasexC]

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
        features_full = self.backbone(img_full)
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
