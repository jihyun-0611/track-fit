import math
import numpy as np
import cv2
import torch


def normalize(img, img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    img = np.array(img, dtype=np.float32)
    return (img - img_mean) * img_scale


def pad_width(img, stride, min_dims, pad_value=(0, 0, 0)):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def infer_frame(net, device, frame, net_input_height, stride=8, upsample_ratio=4):
    h, w, _ = frame.shape
    scale = net_input_height / h
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img = normalize(img)
    min_dims = [net_input_height, max(img.shape[1], net_input_height)]
    img, pad = pad_width(img, stride, min_dims)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img = img.to(device)

    stages_output = net(img)
    heatmaps = np.transpose(stages_output[-2].squeeze().detach().cpu().numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio)
    pafs = np.transpose(stages_output[-1].squeeze().detach().cpu().numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio)
    return heatmaps, pafs, scale, pad
