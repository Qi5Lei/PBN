import math
import os
import json
import random
import sys
import copy

import torch
from PIL import Image
import yaml
import shutil
import cv2
import time
import numpy as np
from loguru import logger
from pathlib import Path

from matplotlib import pyplot as plt
from tqdm import tqdm

dir_list = lambda x: list_dir(x, with_dir_path=True)
exist = lambda target_path: os.path.exists(target_path)

info = lambda msg: logger.info(msg)
error = lambda msg: logger.error(msg)
debug = lambda msg: logger.debug(msg)
warn = lambda msg: logger.warning(msg)


def configure_logging(logger_dir, level="INFO"):
    """配置日志"""
    create_not_exist(logger_dir)
    t = time.strftime("%Y_%m_%d")
    logger.add(f"{logger_dir}/log_{t}.log", rotation="500MB", encoding="utf-8", enqueue=True,
               retention="10 days", level=level)


def list_dir(dir_path, with_dir_path=False):
    file_list = os.listdir(dir_path)
    if with_dir_path:
        file_list = [connect_path(dir_path, file_path) for file_path in file_list]
    return file_list


def get_filename(file_path, need_suffix=False):
    if need_suffix:
        return os.path.basename(file_path)
    else:
        return os.path.basename(file_path).split(".")[0]


def copy2dir(source_path_list, target_dir_path):
    create_not_exist(target_dir_path)
    for idx, pth in enumerate(source_path_list):
        if not os.path.exists(pth):
            print("file: {}  not exist")
        # print("{}/{}".format(idx + 1, len(source_path_list)))
        target_path = connect_path(target_dir_path, os.path.basename(pth))
        shutil.copy(pth, target_path)


def add_suffix(path, suffix, filename=None):
    if filename is None:
        res = path + "." + suffix
    else:
        res = os.path.join(path, get_filename(filename)) + "." + suffix
    return res


def create_not_exist(*target_path_list):
    for target_path in target_path_list:
        if not os.path.exists(target_path):
            os.makedirs(target_path)


def connect_path(*paths):
    return os.path.join(*paths)


def del_file_list(file_list):
    for file in file_list:
        os.remove(file)


def reset_dir(target_dir_path):
    delete_dir_exist(target_dir_path)
    create_not_exist(target_dir_path)


def delete_dir_exist(target_dir_path):
    if os.path.exists(target_dir_path) and os.path.isdir(target_dir_path):
        shutil.rmtree(target_dir_path)


def write_txt_file(txt, txt_path):
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(txt)


def load_config(config_path):
    assert os.path.exists(config_path), "Config file: {} is not exists".format(config_path)
    if config_path.endswith(".yaml"):
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    elif config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.loads(config_file.read())
    else:
        raise ValueError("the config type is not support")
    return config


def cal_iou(box1, box2):
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou


def cal_center(box):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.
    y_center = (y_min + y_max) / 2.
    return x_center, y_center


def cal_area(box):
    x_min, y_min, x_max, y_max = box

    return abs((x_max - x_min) * (y_max - y_min))


def out_img(name, img, root="."):
    count = 0
    file_name = f"{root}/{name}_{count}.jpg"
    if root != ".":
        create_not_exist(root)

    while exist(file_name):
        count += 1
        file_name = f"{root}/{name}_{count}.jpg"

    # plt.imshow(img)
    # plt.savefig(file_name)

    cv2.imwrite(file_name, img)


def load_img(img_path):
    try:
        img: Image = Image.open(img_path)
        img.load()
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_name = get_filename(img_path)
    except Exception as e:
        error(img_path)
        return None, None
    return img_name, img


def cache_images(valid_img_list, batch_idx, batch_size, thread_pool, batch_num):
    # load valid images
    image_cache_list = []
    image_name_list = []
    start_idx = batch_idx * batch_size
    end_idx = min(len(valid_img_list), start_idx + batch_size)
    cache_list = valid_img_list[start_idx:end_idx]
    result = thread_pool.imap(lambda x: load_img(x), cache_list)
    gb = 0  # Gigabytes of cached images
    pbar = tqdm(result, total=len(cache_list))
    for img_name, img in pbar:
        if img is None:
            continue
        gb += img.nbytes
        pbar.desc = f'Caching (batch: {batch_idx + 1}/{batch_num}/{gb / 1E9:.1f}GB) '
        image_cache_list.append(img)
        image_name_list.append(img_name)
    return image_cache_list, image_name_list


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_mean_std(module, input, output):
    # input是一个元组，包含输入张量。
    # output是输出张量。
    mean = output.mean(dim=[0,2,3]).detach().cpu().numpy()
    std = output.std(dim=[0,2,3]).detach().cpu().numpy()
    print(f"Layer {module.__class__.__name__}: Mean={mean}, Std={std}")