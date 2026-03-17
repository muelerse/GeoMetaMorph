"""
framework/transforms.py
=======================
Built-in geometric transformation functions for 2D and 3D numpy arrays.

All functions follow the same signature contract:
    func(image: np.ndarray, *params, background: float = 0) -> np.ndarray

They return a new array and never modify the input in-place, so a single loaded image can be passed to multiple
transforms safely.

The background parameter controls the fill value used wherever new pixels are introduced (e.g., after a translation or
rotation). Mirror operations do not introduce new pixels and ignore this parameter.

2D transforms
-------------
translate_2d(image, dx, dy, background=0)
rotate_2d(image, angle, background=0)
mirror_2d(image, axis, background=0)       (axis: 0=rows (vertical flip), 1=cols (horizontal flip))
scale_2d(image, factor, background=0)

3D transforms
-------------
translate_3d(image, dx, dy, dz, background=0)
rotate_3d(image, angle_x, angle_y, angle_z, background=0)
mirror_3d(image, axis, background=0)       (axis: 0, 1, or 2)
scale_3d(image, factor, background=0)

Default registry
----------------
DEFAULT_TRANSFORMS maps the canonical transform names used in PARAMS dicts to the corresponding functions above.
AbstractRunner.apply_transform() dispatches through this registry by default. Subclasses may extend or replace it by
overriding the transform_functions property.
"""

import numpy as np
import scipy.ndimage as ndimage

try:
    import skimage.transform as _skimage
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


####2D transforms#######################################################################################################

def translate_2d(image: np.ndarray, dx: float, dy: float, background: float = 0) -> np.ndarray:
    """
    Translate a 2D image by dx pixels along axis-1 (columns / horizontal) and dy pixels along axis-0 (rows / vertical).
    """
    return ndimage.shift(image, shift=(dy, dx), cval=background, order=1)


def rotate_2d(image: np.ndarray, angle: float, background: float = 0) -> np.ndarray:
    """
    Rotate a 2D image counter-clockwise by angle degrees around its center.
    """
    return ndimage.rotate(image, angle, reshape=False, cval=background)


def mirror_2d(image: np.ndarray, axis: int, background: float = 0) -> np.ndarray:
    """
    Mirror a 2D image along axis.

    axis=0: vertical flip   (rows reversed)
    axis=1: horizontal flip (columns reversed)
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (vertical flip) or 1 (horizontal flip)")
    return np.flip(image, axis=axis).copy()


def scale_2d(image: np.ndarray, factor: float, background: float = 0) -> np.ndarray:
    """
    Scale a 2D image uniformly by factor. Uses scikit-image rescale when available (better quality), otherwise
    falls back to scipy.ndimage.zoom.
    """
    if factor <= 0:
        raise ValueError("scale factor must be positive")
    if _HAS_SKIMAGE:
        scaled = _skimage.rescale(
            image,
            factor,
            mode="constant",
            cval=background,
            preserve_range=True,
            anti_aliasing=(factor < 1),
        )
        return scaled.astype(image.dtype)
    return ndimage.zoom(image, factor, cval=background).astype(image.dtype)


####3D transforms#######################################################################################################

def translate_3d(
    image: np.ndarray, dx: float, dy: float, dz: float, background: float = 0
) -> np.ndarray:
    """
    Translate a 3D image by (dx, dy, dz) voxels along axes (0, 1, 2).
    """
    return ndimage.shift(image, shift=(dx, dy, dz), cval=background, order=1)


def rotate_3d(
    image: np.ndarray,
    angle_x: float,
    angle_y: float,
    angle_z: float,
    background: float = 0,
) -> np.ndarray:
    """
    Rotate a 3D image sequentially around the z, y, and x axes.

    Rotations are applied in z, y, x order, matching the PS3D reference implementation. For a volume stored as
    [depth, height, width]:

    * z-rotation: axes (0, 2)  - depth-width plane
    * y-rotation: axes (1, 2)  - height-width plane
    * x-rotation: axes (0, 1)  - depth-height plane
    """
    result = image
    if angle_z % 360 != 0:
        result = ndimage.rotate(result, -angle_z, axes=(0, 2), reshape=False, cval=background)
    if angle_y % 360 != 0:
        result = ndimage.rotate(result, -angle_y, axes=(1, 2), reshape=False, cval=background)
    if angle_x % 360 != 0:
        result = ndimage.rotate(result, -angle_x, axes=(0, 1), reshape=False, cval=background)
    return result


def mirror_3d(image: np.ndarray, axis: int, background: float = 0) -> np.ndarray:
    """
    Mirror a 3D image along axis (0, 1, or 2).
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    return np.flip(image, axis=axis).copy()


def scale_3d(image: np.ndarray, factor: float, background: float = 0) -> np.ndarray:
    """
    Scale a 3D image uniformly by factor. Uses scikit-image rescale when available, otherwise scipy.ndimage.zoom.
    """
    if factor <= 0:
        raise ValueError("scale factor must be positive")
    if _HAS_SKIMAGE:
        scaled = _skimage.rescale(
            image,
            factor,
            mode="constant",
            cval=background,
            preserve_range=True,
            anti_aliasing=(factor < 1),
        )
        return scaled.astype(image.dtype)
    return ndimage.zoom(image, factor, cval=background).astype(image.dtype)


####Default transform registry##########################################################################################
# Maps canonical transform names (as used in PARAMS dicts) to the corresponding transform function.
# AbstractRunner.apply_transform() dispatches through this dict by default.
# Subclasses may extend it by overriding the transform_functions property:
#     @property
#     def transform_functions(self):
#         return {**DEFAULT_TRANSFORMS, "MyTransform": my_fn}

DEFAULT_TRANSFORMS: dict = {
    ####2D##########################################
    "Translate2D": translate_2d,    # params: image, dx, dy, background
    "Rotate2D":    rotate_2d,       # params: image, angle, background
    "Mirror2D":    mirror_2d,       # params: image, axis, background
    "Scale2D":     scale_2d,        # params: image, factor, background
    ####3D##########################################
    "Translate3D": translate_3d,
    "Rotate3D":    rotate_3d,
    "Mirror3D":    mirror_3d, # params: [axis]
    "Scale3D":     scale_3d,
}


####Default parameter registry#########################################################################################
# Maps each transform name to a list of parameter vectors containing only empirically stable values.
#
# Selection criteria:
#   2D transforms (Mirror2D, Scale2D, Translate2D):
#       Parameter values rated Stable in PolarityJaM.
#   3D transforms (Mirror3D, Scale3D, Rotate3D):
#       Intersection of parameter values rated Stable in both PS3D and GPAW.
#       (PolarityJaM is a 2D domain; 3D transforms are N/A there.)
#
# Transforms omitted entirely:
#   Rotate2D:   Rotate z is Unstable in PolarityJaM; no stable values found.
#   Translate3D:Uses PS3D stable values only: GPAW translations are Unsafe due to that domain's geometry optimizer, not
#               because translation is generally unsafe.

DEFAULT_PARAMS: dict = {

    ####2D (PolarityJaM stable values)##################################################################################

    # Mirror x (axis=0): Stable in PolarityJaM. Mirror y (axis=1) is only Semi-Stable and thus excluded.
    "Mirror2D": [
        [0],
    ],

    # Scale factors rated Stable in PolarityJaM.
    "Scale2D": [
        [0.8], [0.92], [0.95], [1.03], [1.1],
    ],

    # Translate x/y tested independently (one axis non-zero at a time).
    # x-stable in PolarityJaM: -13, 5  ;  y-stable in PolarityJaM: -7
    "Translate2D": [
        [-13, 0], [5, 0],   # x-axis translations
        [0, -7],            # y-axis translations
    ],

    ####3D (intersection of PS3D & GPAW stable values)#################################################################

    # All three mirror axes are Stable in both PS3D and GPAW.
    "Mirror3D": [
        [0], [1], [2],
    ],

    # Scale factors stable in both PS3D and GPAW.
    # PS3D: {0.9, 0.92, 0.95, 0.99, 1.01, 1.03, 1.05, 1.1, 1.2}
    # GPAW: {0.95, 0.99, 1.01, 1.03, 1.05}
    # Intersection: {0.95, 0.99, 1.01, 1.03, 1.05}
    "Scale3D": [
        [0.95], [0.99], [1.01], [1.03], [1.05],
    ],

    # Translations tested one axis at a time (remaining axes set to 0).
    # Values from PS3D stable set as described above.

    # PS3D Translate x: {-10,-7,-5,-3,-2,-1,1,2,3,5,7,10}
    # PS3D Translate y: {-13,-10,-7,-5,-3,-2,-1,1,2,3,5,7,10}
    # PS3D Translate z: {-20,-18,-15,-13,-10,-7,-5,-3,-2,-1,1,2,3,5,7,10,13,15,18,20}
    "Translate3D": [
        # x-axis
        [-10, 0, 0], [-7, 0, 0], [-5, 0, 0], [-3, 0, 0], [-2, 0, 0], [-1, 0, 0],
        [1, 0, 0], [2, 0, 0], [3, 0, 0], [5, 0, 0], [7, 0, 0], [10, 0, 0],
        # y-axis
        [0, -13, 0], [0, -10, 0], [0, -7, 0], [0, -5, 0], [0, -3, 0], [0, -2, 0], [0, -1, 0],
        [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 5, 0], [0, 7, 0], [0, 10, 0],
        # z-axis
        [0, 0, -20], [0, 0, -18], [0, 0, -15], [0, 0, -13], [0, 0, -10], [0, 0, -7],
        [0, 0, -5], [0, 0, -3], [0, 0, -2], [0, 0, -1],
        [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 5], [0, 0, 7], [0, 0, 10],
        [0, 0, 13], [0, 0, 15], [0, 0, 18], [0, 0, 20],
    ],

    # Rotations tested one axis at a time (remaining axes set to 0).
    # Intersection of stable angle values in PS3D and GPAW per axis:
    #   x: PS3D {0.5,1,2,3,5,180}
    #      GPAW {0.5,1,2,3,5,7,10,...,270}
    #      =    {0.5,1,2,3,5,180}
    #   y: PS3D {0,0.5,1,2,3,5,90,180,270}
    #      GPAW {0.5,1,2,3,5,7,10,...,270}
    #      =    {0.5,1,2,3,5,90,180,270}
    #   z: PS3D {0.5,1,2,3,5,180}
    #      GPAW {0.5,1,2,3,5,7,10,...,270}
    #      =    {0.5,1,2,3,5,180}
    "Rotate3D": [
        # x-axis
        [0.5, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [5, 0, 0], [180, 0, 0],
        # y-axis
        [0, 0.5, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 5, 0], [0, 90, 0], [0, 180, 0], [0, 270, 0],
        # z-axis
        [0, 0, 0.5], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 5], [0, 0, 180],
    ],
}