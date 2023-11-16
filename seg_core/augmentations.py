"""
Dual-Teacher
Copyright (c) 2023-present NAVER Cloud Corp.
distributed under NVIDIA Source Code License for SegFormer
--------------------------------------------------------
References:
SegFormer: https://github.com/NVlabs/SegFormer
--------------------------------------------------------
"""
import torch
import random
from PIL import ImageFilter
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as transforms_f


def compute_cutmix(h, w, imgs, labels, criterion, model, ema_model, image_u, threshold):
    with torch.no_grad():
        pred = ema_model(image_u)
        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        pred = F.softmax(pred, dim=1)
        pred_logit, pred_label = torch.max(pred, dim=1)

    image_aug, label_aug = cut_mixer(image_u, pred_label.clone())

    image_aug, label_aug, pred_logit = \
        batch_transform(image_aug, label_aug, pred_logit,
                        crop_size=(pred_logit.shape[1], pred_logit.shape[2]), scale_size=(1.0, 1.0), apply_augmentation=True)

    num_labeled = len(imgs)
    outputs = model(torch.cat([imgs, image_aug]))
    outputs, outputs_u = outputs[:num_labeled], outputs[num_labeled:]
    pred_large = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
    sup_loss = criterion(pred_large, labels.type(torch.long).clone())

    pred_u = F.interpolate(outputs_u, (h, w), mode="bilinear", align_corners=False)

    cutmix_loss = compute_unsupervised_loss(pred_u, label_aug.clone(), pred_logit, threshold)
    return sup_loss + cutmix_loss


def tensor_to_pil(im, label, logits):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return im, label, logits


def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2


def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, interpolation=transforms_f.InterpolationMode("bilinear"))  # Image.BILINEAR
    label = transforms_f.resize(label, resized_size, interpolation=transforms_f.InterpolationMode("nearest"))  # Image.NEAREST
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, interpolation=transforms_f.InterpolationMode("nearest"))  # Image.NEAREST

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label


def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).cuda(), torch.cat(label_list).cuda(), torch.cat(logits_list).to(device)
    return data_trans, label_trans, logits_trans


def compute_unsupervised_loss(predict, target, logits, strong_threshold):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()  # only count valid pixels
    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    loss = F.cross_entropy(predict, target, reduction='none', ignore_index=255)
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    return weighted_loss


def rand_bbox_1(size, lam=None):
    # past implementation
    W = size[2]
    H = size[3]
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mixer(data, target):
    target = target.unsqueeze(dim=1)
    mix_data = data.clone()
    mix_target = target.clone()
    u_rand_index = torch.randperm(data.size()[0])[:data.size()[0]].cuda()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_1(data.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_data.shape[0]):
        mix_data[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            data[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            target[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del data, target
    torch.cuda.empty_cache()
    return mix_data, mix_target.squeeze(dim=1)


def get_bin_mask(b, argmax_occluder):
    for image_i in range(b):
        classes = torch.unique(argmax_occluder[image_i])

        classes = classes[classes != 255]
        nclasses = classes.shape[0]
        classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()
        if image_i == 0:
            binary_mask = generate_class_mask(argmax_occluder[image_i], classes).unsqueeze(0).cuda()
        else:
            binary_mask = torch.cat((binary_mask, generate_class_mask(argmax_occluder[image_i], classes).unsqueeze(0).cuda()))
    return binary_mask


def compute_classmix(b, h, w, criterion, cm_loss_fn, model, ema_model, imgs, labels, unsup_imgs, image_u_strong, threshold):
    # Unlabeled Process
    with torch.no_grad():
        logits_occluder = ema_model(unsup_imgs)  # 129
        logits_occluder = F.interpolate(logits_occluder, (h, w), mode="bilinear", align_corners=False)  # 513
        softmax_occluder = torch.softmax(logits_occluder, dim=1)
        max_prob_occluder, argmax_occluder = torch.max(softmax_occluder, dim=1)

    binary_mask = get_bin_mask(b, argmax_occluder)
    binary_mask = binary_mask.squeeze(dim=1)
    if b == 2:
        shuffle_index = torch.tensor([1, 0])
    else:
        shuffle_index = torch.randperm(b).cuda()
    class_mixed_img = class_mix(occluder_mask=binary_mask, occluder=image_u_strong, occludee=image_u_strong[shuffle_index])

    num_labeled = len(imgs)
    outputs = model(torch.cat([imgs, class_mixed_img]))
    outputs, outputs_u = outputs[:num_labeled], outputs[num_labeled:]

    pred_large = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
    sup_loss = criterion(pred_large, labels.type(torch.long).clone())
    del outputs, pred_large
    torch.cuda.empty_cache()
    logits_class_mixed = F.interpolate(outputs_u, (h, w), mode="bilinear", align_corners=False)

    class_mixed_softmax = class_mix(occluder_mask=binary_mask, occluder=softmax_occluder, occludee=softmax_occluder[shuffle_index])
    max_prob_occluder, pseudo_label = torch.max(class_mixed_softmax, dim=1)

    unlabeled_weight = torch.sum(max_prob_occluder.ge(threshold).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
    pixel_weight = unlabeled_weight * torch.ones(max_prob_occluder.shape).cuda()

    class_mix_loss = cm_loss_fn(logits_class_mixed, pseudo_label, pixel_weight)
    loss = sup_loss + class_mix_loss
    return loss


def class_mix(occluder_mask, occluder, occludee):
    if occluder.dim() == 4 and occluder.shape[1] == 3:  # Image
        occluder_mask = occluder_mask.unsqueeze(dim=1).repeat(1, 3, 1, 1)
    elif occluder.dim() == 4 and occluder.shape[1] == 150:  # Image
        occluder_mask = occluder_mask.unsqueeze(dim=1).repeat(1, 150, 1, 1)

    masked_data = occluder_mask.float() * occluder + (1 - occluder_mask.float()) * occludee
    del occluder_mask, occluder, occludee
    torch.cuda.empty_cache()
    return masked_data


def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    binary_mask = pred.eq(classes).sum(0)
    return binary_mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def compute_ic(model, ema_model, image_u, image_u_strong, criterion_u, label_u, h, w, threshold):
    with torch.no_grad():
        logits = ema_model(image_u)  # 129
        logits = F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)  # 513
        softmax_out = torch.softmax(logits, dim=1)
        max_probs, argmax_label = torch.max(softmax_out, dim=1)
    pred_dc = model(image_u_strong)
    pred_dc = F.interpolate(pred_dc, (h, w), mode="bilinear", align_corners=False)  # 513
    loss_dc = criterion_u(pred_dc, argmax_label)
    loss_dc = loss_dc * ((max_probs >= threshold) & (label_u != 255))
    loss_dc = loss_dc.sum() / (label_u != 255).sum().item()
    return loss_dc.clone()


class ClassMixLoss(nn.Module):
    def __init__(self, weight=None, reduction=None, ignore_index=None):
        super(ClassMixLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target, pixel_weight):
        loss = self.CE(output, target)
        loss = torch.mean(loss * pixel_weight)
        return loss
