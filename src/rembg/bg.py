import functools
import io
import psutil
import os

import numpy as np
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion
from time import time
import csv
from .u2net import detect



def csv_writer(file):
    return open(file, 'a',)
    # return csv.writer(csv_file)


csv_file = csv_writer("./times.csv")

pid = os.getpid()
py = psutil.Process(pid)

id_count = 1

field_names = ["id", "time", "cpu", "memory", "memory_process"]


def timing_decorator(func):

    def inner(*args, **kwargs):
        global id_count

        t1 = time()
        res = func(*args, **kwargs)
        t2 = time()
        writer = csv.DictWriter(csv_file, field_names)

        data = {
            "id": id_count,
            "time": t2-t1,
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory()[2],
            "memory_process": py.memory_info()[0]/2.**30,
        }

        id_count = id_count + 1
        writer.writerow(data)
        csv_file.flush()

        return res

    return inner


def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    size = img.size

    img.thumbnail((base_size, base_size), Image.LANCZOS)
    mask = mask.resize(img.size, Image.LANCZOS)

    img = np.asarray(img)
    mask = np.asarray(mask)

    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)
    cutout = cutout.resize(size, Image.LANCZOS)

    return cutout


def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
    return cutout


@functools.lru_cache(maxsize=None)
def get_model(model_name):
    if model_name == "u2netp":
        return detect.load_model(model_name="u2netp")
    if model_name == "u2net_human_seg":
        return detect.load_model(model_name="u2net_human_seg")
    else:
        return detect.load_model(model_name="u2net")


@timing_decorator
def remove(
    data,
    model_name="u2net",
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_structure_size=10,
    alpha_matting_base_size=1000,
    file_name=None,
    s3=None
):
    model = get_model(model_name)
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # img
    if file_name and s3:
        fn = "./inputs/"+file_name+".png"
        img.save(fn)
        print(fn)
        s3.upload_file(fn, "yogupta",  "input/"+file_name+".png")

    mask = detect.predict(model, np.array(img)).convert("L")

    if alpha_matting:
        cutout = alpha_matting_cutout(
            img,
            mask,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_structure_size,
            alpha_matting_base_size,
        )
    else:
        cutout = naive_cutout(img, mask)

    bio = io.BytesIO()
    # cutout.save(bio, "PNG")
    cutout.save(bio, "PNG", optimize=True)

    if file_name and s3:
        fn = "./outputs/" + file_name + ".png"
        img.save(fn)
        s3.upload_file(fn, "yogupta",  "output/"+file_name+".png")

    return bio.getbuffer()
