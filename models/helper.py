import numpy as np
#anchor  (y_{min}, x_{min}, y_{max}, x_{max})
# loc is `t_y, t_x, t_h, t_w`.
# return bbox is  (y_{min}, x_{min}, y_{max}, x_{max})
def loc2bbox(anchor, loc):
    px = (anchor[:,1] + anchor[:,3]) * 0.5
    py = (anchor[:,0] + anchor[:,2]) * 0.5
    ah = (anchor[:,2] - anchor[:,0])
    aw = (anchor[:,3] - anchor[:,1]) 

    ty = loc[:,0]
    tx = loc[:,1]
    th = loc[:,2]
    tw = loc[:,3]

    hy = ah * ty + py
    hx = aw * tx + px
    hh = ah * np.exp(th)
    hw = aw * np.exp(tw)

    y_min = hy - hh * 0.5
    y_max = hy + hh * 0.5
    x_min = hx - hw * 0.5
    x_max = hx + hw * 0.5
    return np.stack([y_min,x_min,y_max, x_max], axis=1) 

def _test():
    a = np.ndarray((10,4))
    b = np.ndarray((10,4))
    t = loc2bbox(a,b)
 
if __name__ == '__main__':
    _test()