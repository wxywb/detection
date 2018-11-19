import torch
import numpy as np
from torch import nn
from roi.roi_module import RoIPooling2D
class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
    
    def forward(self,img):
        feature = self.extracor(img)


class RoIHead(nn.Module):
    #scale use form estimated the position in feature map which given in original image coordination.
    def __init__(self, n_class, roisize, classifier, scale):
    #because classifier got form vgg 16 ,the output dimension is 4096.
        self.fc_cls = nn.Linear(4096, n_class)
        self.fc_loc = nn.Linear(4096, n_class * 4)
    # define the pooled size after roi pooling.
        self.roisize = roisize 
        self.classifer = classifier
        self.roi = RoIPooling2D(roisize, roisize, scale)

    #rois_boxes is matrix of (n,5)
    def forward(self, input, roi_boxes):
        # TODO: there is more thing to do convert the roi_boxes to tensor, for now just implement the whole system.
        roi_feature = self.roi.forward(input, roi_boxes)
        #for now the feature is (n , c, 7, 7) seems it will can be 4096,1
        roi_feature = roi_feature.view(roi_feature.shape[0], -1)
        fc = self.classifer(roi_feature)
        score = self.fc_cls(fc)
        loc = self.fc_loc(fc)
        return score, loc 
        