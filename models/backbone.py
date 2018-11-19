import torch as t
from torch import nn
from torchvision.models import vgg16

def init_from_vgg16():
    model = vgg16()
    features = model.features
    classifiers = model.classifier
    not_use_dropout = True
    if not_use_dropout:
        del classifiers[2]
        del classifiers[5]
    
    for layer in features[:10]:
        for p in layer.parameters():
            p.require_grad = False
    features = nn.Sequential(*features)
    classifiers = nn.Sequential(*classifiers)
    return features, classifiers 
    


if __name__ == '__main__':
    m, c = init_from_vgg16()
    print(c)

