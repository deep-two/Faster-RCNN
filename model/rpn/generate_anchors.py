import numpy as np

def generate_anchors(base_size=16,  ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
    pass

def get_anchor_info(anchor):
    w = anchor[2] - anchor[0]
    h = anchor[3] - anchor[1]
    x_cen = (anchor[2] + anchor[0]) / 2
    y_cen = (anchor[3] + anchor[1]) / 2

    return w, h, x_cen, y_cen

def get_anchors_points(ws, hs, x_cen, y_cen):
    pass

def make_anchors_by_ratio(base_anchor, ratios):
    pass

def make_anchors_by_scale(base_anchor, scales):
    pass


if __name__ == "__main__":
    print(get_anchor_info(np.array([300,200,600,400])))