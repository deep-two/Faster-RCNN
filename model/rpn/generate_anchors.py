import numpy as np

def generate_anchors(base_size=16,  ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
    base = np.array([0, 0, base_size, base_size])
    anchors = make_anchors_by_ratio(base, ratios)
    anchors = np.vstack([make_anchors_by_scale(anchor, scales) for anchor in anchors])

    return anchors

def get_anchor_info(anchor):
    w = anchor[2] - anchor[0]
    h = anchor[3] - anchor[1]
    x_cen = (anchor[2] + anchor[0]) / 2
    y_cen = (anchor[3] + anchor[1]) / 2

    return w, h, x_cen, y_cen

def get_anchors_points(ws, hs, x_cen, y_cen):
    assert len(ws) == len(hs)

    anchors = np.stack([x_cen - ws / 2, y_cen - hs /2, x_cen + ws / 2, y_cen + hs / 2], axis=-1)

    return anchors

def make_anchors_by_ratio(base_anchor, ratios, sus_size = True):
    w, h, x, y = get_anchor_info(base_anchor)
    if not isinstance(ratios, np.ndarray):
        ratios = np.array(ratios)

    if sus_size:
        ws = np.sqrt(w * h / ratios)
        hs = ws / ratios
    else:
        ws = np.array([w for _ in range(len(ratios))])
        hs = w / ratios

    anchors = get_anchors_points(ws, hs, x, y)

    return anchors


def make_anchors_by_scale(base_anchor, scales):
    w, h, x, y = get_anchor_info(base_anchor)

    if not isinstance(scales, np.ndarray):
        scales = np.array(scales)

    ws = scales * w
    hs = scales * h

    anchors = get_anchors_points(ws, hs, x, y)

    return anchors


if __name__ == "__main__":
    print(generate_anchors())