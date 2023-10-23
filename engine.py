import os

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from u2net import utils, model
from isnet.model import ISNetDIS
from skimage import io


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([utils.RescaleT(320), utils.ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


# def remove_bg(image, resize=False):
#     sample = preprocess(np.array(image))
#
#     with torch.no_grad():
#         inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())
#
#         d1, _, _, _, _, _, _ = model_pred(inputs_test)
#         pred = d1[:, 0, :, :]
#         predict = norm_pred(pred).squeeze().cpu().detach().numpy()
#         img_out = Image.fromarray(predict * 255).convert("RGB")
#         img_out = img_out.resize((image.size), resample=Image.BILINEAR)
#         empty_img = Image.new("RGBA", (image.size), 0)
#         img_out = Image.composite(image, empty_img, img_out.convert("L"))
#         del d1, pred, predict, inputs_test, sample
#
#         return img_out


def _remove(image):
    model_path = './ckpt/u2net.pth'
    model_pred = model.U2NET(3, 1)
    model_pred.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_pred.eval()

    sample = preprocess(np.array(image))
    with torch.no_grad():
        inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())
        d1, _, _, _, _, _, _ = model_pred(inputs_test)
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred).squeeze().cpu().detach().numpy()
        img_out = Image.fromarray(predict * 255).convert("RGB")
        image = image.resize((img_out.size), resample=Image.BILINEAR)
        empty_img = Image.new("RGBA", (image.size), 0)
        img_out = Image.composite(image, empty_img, img_out.convert("L"))

    return img_out


def remove_bg_mult(image, mode_type="u2net"):
    img_out = image.copy()
    if mode_type == "u2net":
        for _ in range(4):
            img_out = _remove(img_out)
    else:
        img_out = is_net_remove(img_out)

    img_out = img_out.resize((image.size), resample=Image.BILINEAR)
    empty_img = Image.new("RGBA", (image.size), 0)
    img_out = Image.composite(image, empty_img, img_out)
    return img_out


def change_background(image, background):
    background = background.resize((image.size), resample=Image.BILINEAR)
    img_out = Image.alpha_composite(background, image)
    return img_out


# ++++++++++++++++++++++++ IS NET +++++++++++++++++++++++++++++++++++++
def is_net_remove(image):
    model_path = 'ckpt/isnet-general-use.pth'
    net = ISNetDIS()
    input_size = [1024, 1024]

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    with torch.no_grad():
        im_array = np.array(image)
        im = im_array[:, :, 0:3]

        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp = im.shape[0:2]

        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        if torch.cuda.is_available():
            image = image.cuda()
        # 归一化后的图像输入到 net 模型
        result = net(image)
        result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)

        result = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        image_array = result.squeeze()
        img_out = Image.fromarray(image_array, mode='L')

    return img_out
