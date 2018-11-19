import torch
import numpy as numpy
from torch import nn
import numpy as np
import torch.nn.functional as F
from helper import loc2bbox
from nms import non_maximum_suppression
#R len(ratios) * len(anchors_scales) will be genreated by this funciton
#size of the result is (y_{min}, x_{min}, y_{max}, x_{max})
#ratios : this is the ratio width to height


class ProposalCreater:
    def __init__(self,):
        self.min_size = 16
        self.pre_nms = 6000
        self.post_nms = 300
    def __call___(self,loc, scores, anchor, img_size, scale = 1.):
        loc = loc.detach().numpy()
        scores = scores.detach().numpy()
        decodedbbox = loc2bbox(anchor,loc)
        h, w = img_size
        #decodebbox (y_{min}, x_{min}, y_{max}, x_{max})
        decodedbbox[:,0] = np.clip(decodedbbox, 0, h)
        decodedbbox[:,2] = np.clip(decodedbbox, 0, h)
        decodedbbox[:,1] = np.clip(decodedbbox, 0 ,w)
        decodedbbox[:,3] = np.clip(decodedbbox, 0 ,w)
        h = decodedbbox[:,2] - decodedbbox[:,0]
        w = decodedbbox[:,3] - decodedbbox[:,1]
        min_size = self.min_size
        keep = np.where((h > min_size) & (w > min_size))

        roi = decodedbbox[keep,:]
        score = scores[keep,:]

        ind = np.argsort(score)
        roi = roi[ind,:]
        roi = roi[np.arange(min(roi.shape[0], self.pre_nms)),:]



def generate_anchor(orisize, ratios, anchors_scales):
    anchors = list()
    cy = cx = orisize / 2
    R = len(ratios) * len(anchors_scales)
    res = numpy.ndarray(shape = (R,4), dtype = np.float32)
    idx = 0
    for ratio in ratios:
        for anchor in anchors_scales:
            h = orisize * anchor * np.sqrt(ratio)
            w = orisize * anchor * np.sqrt(1 / ratio)
            ymin = cy - h / 2
            ymax = cy + h / 2
            xmin = cx - w / 2
            xmax = cx + w / 2
            vec = np.array([ymin, xmin, ymax, xmax]) 
            res[idx,:] = vec
            idx = idx + 1
    return res


# (y_{min}, x_{min}, y_{max}, x_{max})
def shift_the_anchor(anchor_base, feat_stride, height, width):
    h_ = np.arange(0,feat_stride * height, feat_stride)
    w_ = np.arange(0,feat_stride * width, feat_stride)
    yy,xx = np.meshgrid(h_,w_)
    yy = yy.reshape((yy.size,))
    xx = xx.reshape((xx.size,))
    anchors = np.stack([yy,xx,yy,xx], axis = 1)
    res = anchors.reshape(anchors.shape[0],1 ,4) + anchor_base.reshape(1, anchor_base.shape[0],4)
    return res.reshape(res.shape[0] * res.shape[1],4)


class RPN(nn.Module):
    def __init__(self, in_channels, midchannels):
        self.anchorbase = generate_anchor(16, [0.5, 1, 2],[8,16,32]).shape[0]
        n_anchors = self.anchorbase.shape[0]
        self.middlelayer = nn.Conv2d(in_channels, midchannels, 3, 1, 1)
        self.convloc = nn.Conv2d(midchannels, 4 * n_anchors, 3, 1, 1)
        self.convscore = nn.Conv2d(midchannels, 2 * n_anchors, 1, 1, 1)



    def forward(self, input):
        n,c,h,w = input.shape
        hidden = F.relu(h)
        rpn_locs = self.convloc(hidden)
        rpn_scores= self.convscore(hidden)

        anchor = shift_the_anchor(self.anchor_base,16,h,w)
        # should be (n,4 * n_anchors, hh, ww)
        rpn_locs = rpn_locs.permute(0,2,3,1).contiguous().view(n,-1,4)
        # should be (n,2 * n_anchors, hh, ww)
        rpn_scores = rpn_scores.permute(0,2,3,1).contiguous().view(n,c,h,w,2)
        scores = F.softmax(rpn_scores, dim =4)
        scores = scores.contiguous().view(n,-1,2)
        



def _test():
    ans = generate_anchor(16, [0.5,1,2,],[8,16,32])
    shifted_anchors = shift_the_anchor(ans,16,32,15)
    print(shifted_anchors.shape)

    s = np.ndarray((1,3,10,10,2))
    st = torch.from_numpy(s)
    print(F.softmax(st,dim = 4))

if __name__ == '__main__':
    _test()