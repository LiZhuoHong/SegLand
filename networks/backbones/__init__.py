from .resnet import ResNet, ResNetv2, Bottleneck
from .swintransformer import SwinTransformer
from .hrnet import HighResolutionNet
from .convnext import convnext_tiny
from .lsknet import lsknet_tiny
from utils.pyt_utils import load_model

def get_backbone(norm_layer, pretrained_model=None, backbone='resnet101', relu_l3=True, relu_l4=True, **kwargs):
    if backbone == 'resnet101':
        model = ResNet(Bottleneck,[3, 4, 23, 3], norm_layer=norm_layer, relu_l3=relu_l3, relu_l4=relu_l4, **kwargs)
        print('Backbone:resnet101')
    elif backbone == 'resnet50':
        model = ResNet(Bottleneck,[3, 4, 6, 3], norm_layer=norm_layer, relu_l3=relu_l3, relu_l4=relu_l4, **kwargs)
        print('Backbone:resnet50')
    elif backbone == 'resnet50v2':
        model = ResNetv2(Bottleneck,[3, 4, 6, 3], norm_layer=norm_layer, relu_l3=relu_l3, relu_l4=relu_l4, **kwargs)
        print('Backbone:resnet50v2')
    elif backbone == 'resnet101v2':
        model = ResNetv2(Bottleneck,[3, 4, 23, 3], norm_layer=norm_layer, relu_l3=relu_l3, relu_l4=relu_l4, **kwargs)
        print('Backbone:resnet101v2')
    elif backbone == 'swin-t':
        model = SwinTransformer(pretrain_img_size=224, window_size=7, backbone='swin-t')
        print('Backbone:swin-t')
    elif backbone == 'swin-s':
        model = SwinTransformer(pretrain_img_size=224, window_size=7, backbone='swin-s')
        print('Backbone:swin-s')
    elif backbone == 'swin-b':
        model = SwinTransformer(pretrain_img_size=224, window_size=7, backbone='swin-b')
        print('Backbone:swin-b')
    elif backbone == 'swin-l':
        model = SwinTransformer(pretrain_img_size=224, window_size=7, backbone='swin-l')
        print('Backbone:swin-l')
    elif backbone == 'hr-w18' or backbone == 'hr-w32' or backbone == 'hr-w48':
        model = HighResolutionNet(backbone=backbone)
    elif backbone == 'convnext-t':
        model = convnext_tiny()
    elif backbone == 'lsk-t':
        model = lsknet_tiny()
    else:
        raise RuntimeError('unknown backbone: {}'.format(backbone))
    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model