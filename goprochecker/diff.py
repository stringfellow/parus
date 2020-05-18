"""Iterate over images comparing to a base image to check for changes.

1. Doesn't cope well with movement caused by wind.
2. Shadows moving with the sun are not so bad as the base images tracks along
with the comparison (e.g base is updated to last image matching within
threshold).
3. Cloud cover changes are not dealt with well in the same way as wind movement.

Perhaps need to consider instead of overall percent change, how localised the
change is; which might help 1 and 3 a bit.

"""
from pathlib import Path
import numpy
from PIL import Image, ImageChops


def _compute_manhattan_distance(diff_image):
    """
    Computes a percentage of similarity of the difference image given.

    :param PIL.Image diff_image:
        An image in RGB mode computed from ImageChops.difference

    SOURCE:

    https://github.com/ESSS/pytest-regressions/blob/2.0.0/src/pytest_regressions
    /image_regression.py#L41

    """
    number_of_pixels = diff_image.size[0] * diff_image.size[1]
    return (
        # To obtain a number in 0.0 -> 100.0
        100.0
        * (
            # Compute the sum of differences
            numpy.sum(diff_image)
            /
            # Divide by the number of channel differences RGB * Pixels
            float(3 * number_of_pixels)
        )
        # Normalize between 0.0 -> 1.0
        / 255.0
    )


def find_images(
        dir, diff_threshold,
        copy=False, base_image_name=None, success_limit=5
):
    """Find images in dir that are significantly different from base_image."""
    base_image_name = base_image_name or 'BASE*.JPG'
    results_dir = '__results__'
    data_folder = Path(dir)
    if copy:
        data_folder.joinpath(results_dir).mkdir(exist_ok=True)

    image_list = sorted(list(data_folder.rglob('*.JPG')))

    base_names = data_folder.rglob(base_image_name)
    bases = [Image.open(base_name) for base_name in base_names]

    success_count = 0
    for image_path in image_list:
        if results_dir in str(image_path):
            continue
        comp = Image.open(image_path)
        diffs = [
            ImageChops.difference(base, comp)
            for base in bases
        ]
        pcts = [_compute_manhattan_distance(diff) for diff in diffs]
        sig = [pct > diff_threshold for pct in pcts]
        if all(sig) and success_count <= success_limit:
            print(image_path, pcts)
            success_count += 1
            if copy:
                comp.save(
                    data_folder.joinpath(results_dir, Path(image_path).name)
                )
        else:
            # either the comparison showed no match, or we've matched up to the
            # limit; either way we reset our comparison to the current image,
            # which helps for shifting shadows etc.
            del(bases[0])
            bases.append(comp)
            success_count = 0
