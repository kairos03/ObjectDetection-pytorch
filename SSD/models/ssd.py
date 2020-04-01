import torch
import torch.nn as nn
import torch.nn.functional as F

# from .vgg import vgg16
from torchvision.models import vgg

__all__ = [
    'SSD',
    'vgg_ssd'
]

class SSD(nn.Module):
    def __init__(self, size, num_classes, base, extras, conf_headers, loc_headers):
        super(SSD, self).__init__()
        self.size = size
        self.num_classes = num_classes
        self.base = nn.Sequential(*base)
        self.extras = nn.ModuleList(extras)
        self.conf_headers = nn.ModuleList(conf_headers)
        self.loc_headers = nn.ModuleList(loc_headers)
        self._init_weights()

    def _init_weights(self):
        layers = [ 
            *[self.base[i] for i in range(23, len(self.base))],
            *self.extras, 
            *self.conf_headers, 
            *self.loc_headers
        ]
        for l in layers:
            for param in l.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        ### extract source ###
        # conv4_3 relu
        for i in range(23):
            x = self.base[i](x)
        sources.append(x)
        # base
        for i in range(23, len(self.base)):
            x = self.base[i](x)
        sources.append(x)
        # extras
        for i, l in enumerate(self.extras):
            x = F.relu(l(x), inplace=True)
            if i % 2 == 1:
                sources.append(x)

        for x in sources:
            print(x.shape)

        ### header output ###
        for x, l, c in zip(sources, self.loc_headers, self.conf_headers):
            loc.append(l(x).view(x.size(0), 4, -1))
            conf.append(c(x).view(x.size(0), self.num_classes, -1))
        
        loc, conf = torch.cat(loc, 2).contiguous(), torch.cat(conf, 2).contiguous()

        # return output

    def check_base(self):
        print(self.base)
        print(self.extras)
        print(self.conf_headers)
        print(self.loc_headers)

extras_cfg = {
    300: [256, 'S2', 128, 'S2', 128, 'S1', 128, 'S1']
}
headers_cfg = {
    300: [4, 6, 6, 6, 4, 4]
}


def get_extras(cfg, in_channels):
    layers = []
    for v in cfg:
        if isinstance(v, str) and v[0] == 'S':
            out_channels = in_channels * 2
            if v[1] == '2':
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)]
            elif v[1] == '1':
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)]
        else:
            out_channels = v
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)]
        in_channels = out_channels
    return layers


def get_headers(cfg, base, base_idx, extras, num_classes):
    conf_headers = []
    loc_headers = []
    count = 0
    for i in base_idx:
        loc_headers += [nn.Conv2d(base[i].out_channels, cfg[count] * 4, kernel_size=3, padding=1)]
        conf_headers += [nn.Conv2d(base[i].out_channels, cfg[count] * num_classes, kernel_size=3, padding=1)]
        count += 1
    for i in range(1, len(extras), 2):
        loc_headers += [nn.Conv2d(extras[i].out_channels, cfg[count] * 4, kernel_size=3, stride=1, padding=1)]
        conf_headers += [nn.Conv2d(extras[i].out_channels, cfg[count] * num_classes, kernel_size=3, stride=1, padding=1)]
        count += 1
    return conf_headers, loc_headers


def vgg_for_ssd(vgg):
    features = vgg.features[:-1]
    features[16] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=6, dilation=6)
    fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
    relu = nn.ReLU(True)
    return [*features, pool5, fc6, relu, fc7, relu]


def vgg_ssd():
    size = 300
    num_classes = 21
    base = vgg_for_ssd(vgg.vgg16(pretrained=True))
    extras = get_extras(extras_cfg[size], 1024)
    cls_headers, reg_headers = get_headers(headers_cfg[size], base, [21, 33], extras, num_classes)
    return SSD(size, num_classes, base, extras, cls_headers, reg_headers)


class Loss(nn.Module):
    """
    Objective Loss:
        L(x, c, l, g) = (Lconf(x, c) + αLloc(x, l, g)) / N
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, x, c, l, g):
        # localization loss
        pass
