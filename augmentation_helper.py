import os
from PIL import ImageOps, ImageEnhance
from PIL.Image import Image


def setup_folder(path) -> None:
    if os.path.isdir(path) == False:
        os.mkdir(path)


def file_exists(path) -> bool:
    return os.path.isfile(path)


def flip_img(img: Image, dir: str = "h") -> Image:
    if dir == "v":
        return ImageOps.flip(img)
    return ImageOps.mirror(img)


def rotate_img(img: Image, deg: float) -> Image:
    return img.rotate(deg)


def adjust_brightness(img: Image, amount: float) -> Image:
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(amount)


def zoom(img: Image, factor: float) -> Image:
    width, height = img.size
    return img.resize((int(width * factor), int(height * factor)))


def crop_quadrant(img: Image, quad: int = 0, side: int = 256):
    width, height = img.size
    if quad == 1:
        return img.crop((0, 0, side, side))
    elif quad == 2:
        return img.crop((width - side, 0, width, side))
    elif quad == 3:
        return img.crop((0, height-side, side, height))
    elif quad == 4:
        return img.crop((width - side, height-side, width, height))

    w_pad = (width - side)/2
    h_pad = (height - side)/2

    x1 = w_pad
    x2 = width - w_pad
    while x2 - x1 != side:
        x2 -= 1

    y1 = h_pad
    y2 = height - h_pad
    while y2 - y1 != side:
        y2 -= 1
    return img.crop((x1, y1, x2, y2))


def zoomed_area(img: Image, factor: float, quad: int = 0):
    zoomed = zoom(img, factor)
    return crop_quadrant(zoomed, quad)


def stretch(img: Image, width_factor: float = 1, height_factor: float = 1):
    width, height = img.size
    new_width = int(width * width_factor)
    new_height = int(height * height_factor)
    stretched = img.resize((new_width, new_height))
    if new_width > width or new_height > height:
        sqr_stretched = crop_quadrant(stretched, side=new_width if new_width > new_height else new_height)
        return sqr_stretched.resize((256, 256))
    return crop_quadrant(stretched)
