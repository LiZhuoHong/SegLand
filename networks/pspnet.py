import torch
import torch.nn as nn
from torch.nn import functional as F

from .backbones import get_backbone

def masked_average_pooling(feature, mask):
    '''
        downsample mask to the shape of features and conduct MAP.
        feature : [BxCxhxw]
        mask    : [Bx1xHxW]
    '''
    mask = F.interpolate(mask, size=feature.shape[-2:], mode="bilinear", align_corners=True)
    masked_feature = torch.sum(feature*mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5) # [BxC]
    return masked_feature.mean(0, keepdim=True).unsqueeze(1) # [1x1xC]

class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=256, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=1)
            )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn, nn.ReLU(inplace=True))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class GFSS_Model(nn.Module):
    '''
        Segmenter for Generalized Few-shot Semantic Segmentation
    '''
    def __init__(self, n_base, criterion=None, norm_layer=nn.BatchNorm2d, use_base=True, is_ft=False, n_novel=0, **kwargs):
        super(GFSS_Model, self).__init__()
        self.use_base = use_base
        self.d_model = 512

        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)
        self.decoder = PSPModule(2048, out_features=self.d_model, norm_layer=norm_layer)

        if is_ft:
            assert n_novel > 0
            self.classifier = nn.Conv2d(self.d_model, 1+n_base, kernel_size=1, bias=False)
            self.classifier_n = nn.Conv2d(self.d_model, n_novel, kernel_size=1, bias=False)
            self.ft_freeze()
        else:
            self.classifier = nn.Conv2d(self.d_model, 1+n_base, kernel_size=1, bias=False)
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1+n_base, kernel_size=1)
            )
            
        self.is_ft = is_ft
        self.n_base = n_base
        self.criterion = criterion
        self.T = 0.1

    def train_mode(self):
        self.train()
        self.backbone.eval()
        # self.decoder.eval()

    def ft_freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        # for param in self.decoder.parameters():
        #     param.requires_grad = False

    def forward(self, img, mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        if self.is_ft:
            with torch.no_grad():
                features = self.backbone.base_forward(img)
            features = self.decoder(features) # [BxCxhxw]
            out_b = self.classifier(features)
            out_n = self.classifier_n(features)
            out = torch.cat([out_b, out_n], dim=1)
        else:
            x4, x3, _  = self.backbone.base_forward(img, return_list=True)
            features = self.decoder(x4) # [BxCxhxw]
            out = self.classifier(features)
            aux_out = self.aux_classifier(x3)

        if self.criterion is not None and mask is not None:
            return self.criterion(out, mask, aux_pred=aux_out)
        else:
            return out
