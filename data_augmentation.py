from Pokemon import Pokedex
from file_n_image_fn import setup_folder, file_exists, rotate_img, flip_img, adjust_brightness, zoomed_area, stretch
from tqdm import tqdm

dex = Pokedex("pokemon.csv")
TRANSFORMS = ["h_flip", "v_flip", "rot_45", "rot_90",
              "rot_135", "rot_180", "rot_225", "rot_270",              
              "zoom_0", "zoom_1", "zoom_2", "zoom_3", "zoom_4",
              "thin", "short"]
ILUMINATION = ["darken_10", "darken_25", "darken_50",
              "lighten_10", "lighten_25", "lighten_50"]
ZOOM_FACTOR = 1.5
TRANSFORM_FN = {
    "h_flip": lambda img: flip_img(img),
    "v_flip": lambda img: flip_img(img, "v"),
    "rot_45": lambda img: rotate_img(img, 45),
    "rot_90": lambda img: rotate_img(img, 90),
    "rot_135": lambda img: rotate_img(img, 135),
    "rot_180": lambda img: rotate_img(img, 180),
    "rot_225": lambda img: rotate_img(img, 225),
    "rot_270": lambda img: rotate_img(img, 270),
    "darken_10": lambda img: adjust_brightness(img, 0.9),
    "darken_25": lambda img: adjust_brightness(img, 0.75),
    "darken_50": lambda img: adjust_brightness(img, 0.5),
    "lighten_10": lambda img: adjust_brightness(img, 1.1),
    "lighten_25": lambda img: adjust_brightness(img, 1.25),
    "lighten_50": lambda img: adjust_brightness(img, 1.5),
    "zoom_0": lambda img: zoomed_area(img, ZOOM_FACTOR),
    "zoom_1": lambda img: zoomed_area(img, ZOOM_FACTOR, 1),
    "zoom_2": lambda img: zoomed_area(img, ZOOM_FACTOR, 2),
    "zoom_3": lambda img: zoomed_area(img, ZOOM_FACTOR, 3),
    "zoom_4": lambda img: zoomed_area(img, ZOOM_FACTOR, 4),
    "thin": lambda img: stretch(img, width_factor=ZOOM_FACTOR-1),
    "short": lambda img: stretch(img, height_factor=ZOOM_FACTOR-1),
}

base_path = "data_augmentation"
setup_folder(base_path)

for transform in tqdm(TRANSFORMS, desc="Transforms"):
    transform_folder_path = f"{base_path}/{transform}"
    setup_folder(transform_folder_path)

    for pkm in tqdm(dex.pokemons, leave=False, desc="Pokemon"):
        transformed_path = f"{transform_folder_path}/{pkm.pokedex_number}.png"
        if file_exists(transformed_path) == False:
            transformed = TRANSFORM_FN[transform](pkm.image())
            transformed.save(transformed_path)

for transform in tqdm([None] + TRANSFORMS, desc="Transforms"):
    for light in ILUMINATION:
        transform_folder_path = f"{base_path}/{'' if transform == None else (transform + '_')}{light}"
        setup_folder(transform_folder_path)

        for pkm in tqdm(dex.pokemons, leave=False, desc="Pokemon"):
            transformed_path = f"{transform_folder_path}/{pkm.pokedex_number}.png"
            pkm.img_variation = transform
            if file_exists(transformed_path) == False:
                transformed = TRANSFORM_FN[light](pkm.image())
                transformed.save(transformed_path)



