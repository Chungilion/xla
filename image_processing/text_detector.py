import cv2
import torch
import numpy as np

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CRAFT.craft import CRAFT
from CRAFT import craft_utils
from CRAFT import imgproc

_craft_net = None

def copyStateDict(state_dict):
    from collections import OrderedDict
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def detect_text_craft(image, trained_model, cuda=False, text_threshold=0.4, link_threshold=0.3, low_text=0.2, canvas_size=1280, mag_ratio=1.5, poly=False):
    """Detect text regions in the image using CRAFT"""
    global _craft_net

    if _craft_net is None:
        print("Loading CRAFT model for the first time...")
        _craft_net = CRAFT()
        _craft_net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))
        _craft_net.eval()

    # Increase mag_ratio to 1.5 for better handling of small text
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # Enhance preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = torch.autograd.Variable(x.unsqueeze(0))

    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = _craft_net(x)

    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Adjust thresholds for better text detection
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    # Filter out very small boxes that might be noise
    boxes = [box for box in boxes if (box[:,0].max() - box[:,0].min()) > 5 and (box[:,1].max() - box[:,1].min()) > 5]

    return boxes, polys