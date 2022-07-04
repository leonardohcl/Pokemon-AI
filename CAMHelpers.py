import numpy as np
from PIL import Image, ImageChops
from matplotlib import cm


def createImage(cam, colormap: str = None):
    """ Converts a class activation map into a PIL Image
    Args:
        cam(array): cam to be converted
        colormap(string): name for a colormap to be applied on the output image
    """
    input_data = np.array(cam)
    
    # Convert pixel values from [0..1] to [0..255]
    if colormap:
        colormap = cm.get_cmap(colormap)
        cam_8bit = colormap(input_data[0]) * 255
        cam_8bit = cam_8bit.astype(np.uint8)

        # Create a 3xMxN array from map (remove alpha channel)
        cam_array = cam_8bit[:, :, :3]

    else:
        cam_8bit = input_data[0] * 255
        cam_8bit = cam_8bit.astype(np.uint8)

        # Create a 3xMxN array from map
        cam_array = np.array([cam_8bit, cam_8bit, cam_8bit])

        # Shift array shape
        # PIL.Image needs it to be MxNx3 to convert from array to image
        cam_array = np.rollaxis(cam_array, 0, 3)

    # Create image from array
    return Image.fromarray(cam_array)


def overlayCAM(image: Image, cam_image: Image, overlay_opacity: float = 0.5):
    """Create an image that's an overlay of a CAM to it's original image (or any two images really)
    
    Args:
        image(Image): original image (or the background image)
        cam_image(Image): image of the CAM (or the overlay image)
        overlay_opacity(float): opacity to the overlay between 0 and 1
    """
    background = image.convert("RGBA")
    overlay = cam_image.convert("RGBA")

    return Image.blend(background, overlay, overlay_opacity)


def multiplyCAM(image: Image, cam_image: Image):
    """ Create an image that's a pixel by pixel multiplication of a CAM to it's original image (or any two images really)
    
    Args:
        image(Image): original image (or the background image)
        cam_image(Image): image of the CAM (or the overlay image)
    
    Remarks:
        This kind of operation works better when the CAM is a greyscale image, so the multiply will result in an image that will darken the less active areas and keep the values closest to the original values on the most active ones

    """
    return ImageChops.multiply(image, cam_image)
