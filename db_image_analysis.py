import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# =======================================================================================
# import and export of images
def load_image_from_file(files):
    """_summary_

    Args:
        files (_type_): _description_

    Returns:
        _type_: _description_
    """
    images = dict()
    for file in files:
        file_name = file.split('/')[-1].split('.')[0]
        
        # load image from file
        image = cv2.imread(file)
        
        # convert BGR to RGB as OpenCV loads images in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # store in dictionary
        images[file_name] = image
    return images


# =======================================================================================
# processing of images
def split_image_into_channels(images):
    """_summary_

    Args:
        images (_type_): _description_

    Returns:
        _type_: _description_
    """
    image_channels = dict()
    for k, i in images.items():
        
        # split into channels
        red, green, blue = cv2.split(i)
        
        # store in dictionary
        image_channels[k] = (red, green, blue)
        
    return image_channels


def load_and_stack(image_list):
    """_summary_

    Args:
        image_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    stacked_images = []
    for img_path in image_list:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image
        if img is not None:
            stacked_images.append(img)
    return np.stack(stacked_images, axis=0) if stacked_images else None


def calculate_channel_ratio(stack_channel1, stack_channel2):
    """_summary_
    Calculating the ratio between two image channels.
    Args:
        stack_channel1 (_type_): numerator channel
        stack_channel2 (_type_): denominator channel - by which channel is to be divided

    Returns:
        _type_: either an array of the ration in each pixel or None
    """
    if stack_channel1 is not None:
        print("Images in Stack_R:")
        for idx, img in enumerate(stack_channel1):
            print(f"{idx}: {img}")
        print(f"Stack_R shape: {stack_channel1.shape}")
    else:
        print("No images found for _R")

    if stack_channel2 is not None:
        print("Images in Stack_G1:")
        for idx, img in enumerate(stack_channel2):
            print(f"{idx}: {img}")
        print(f"Stack_G1 shape: {stack_channel2.shape}")
    else:
        print("No images found for _G1")
   
    if stack_channel1 is not None and stack_channel2 is not None and stack_channel1.shape == stack_channel2.shape:
        ratio_stack = stack_channel1.astype(np.float32) / (stack_channel2.astype(np.float32) + 1e-6)
        print(f"\n >> Ratio stack computed with shape: {ratio_stack.shape}")
        return ratio_stack
    else:
        print("\n >> Ratio stack could not be computed due to mismatched shapes or missing data")
        return None


# =======================================================================================
# calibration
def sternvolmer_simple(x, f, k):
    """
    fitting function according to the common two site model. In general, x represents the pO2 or pCO2 content, whereas
    m, k and f are the common fitting parameters
    :param x:   list
    :param k:   np.float
    :param f:   np.float
    :return: int0/int
    """
    # Int0/Int 
    return 1/(f / (1. + k*x) + (1.-f))


def reverse_sternvolmer(int_norm: np.ndarray, f: float, k: float):
    """_summary_
    Reverse stern-volmer equation calculating the analyte concentration based on given fitting parameters and a 
    normalized intenstiy ratio for a given image. 
    Args:
        int_norm (np.ndarray): normalized intensity ratio given as ndarray. The normalization is done as I/I0 with I0 
                               being the luminescence intensity in the absence of the target analyte.
        f (float): fitting parameter f for stern volmer equation
        k (float): fitting parameter k for stern volmer equation

    Returns:
        _type_: optional either None if no concentration can be calulated (when either k is 0 or I/I0 equals 1-f) or an 
                ndarray presenting the analyte concentration
    """
    denom = (int_norm + f - 1)*k
    nom = 1- int_norm
    if denom is not None and (denom !=0).any():
        return nom/denom
    else:
        return None


# =======================================================================================
# display/plot images
def display_channels_for_image(red_channel, green_channel, blue_channel, figsize=(12, 4)):
    """_summary_

    Args:
        red_channel (_type_): _description_
        green_channel (_type_): _description_
        blue_channel (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12, 4).

    Returns:
        _type_: _description_
    """
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].set_title("Red Channel")
    ax[1].set_title("Green Channel")
    ax[2].set_title("Blue Channel")

    if isinstance(red_channel, np.ndarray):
        ax[0].imshow(red_channel, cmap="Reds")
    if isinstance(green_channel, np.ndarray):
        ax[1].imshow(green_channel, cmap="Greens")
    if isinstance(blue_channel, np.ndarray):
        ax[2].imshow(blue_channel, cmap="Blues")

    for axi in fig.axes:
        axi.axis("off")

    plt.tight_layout()
    return fig


def display_calibration_points(data, column_plot, arg, figsize=(5, 3.5), fs=10):
    """_summary_

    Args:
        data (_type_): _description_
        column_plot (_type_): _description_
        arg (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (5, 3.5).
        fs (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'title' in arg.keys():
        ax.set_title(arg['title'], loc='left', fontsize=fs)
    else:
        ax.set_title('Simplified Stern-Volmer for Calibration', loc='left', fontsize=fs)
    ax.set_xlabel(arg['xlabel'], fontsize=fs)
    ax.set_ylabel(arg['ylabel'], fontsize=fs)


    ax.scatter(data.index, data[column_plot]/data[column_plot].min(), marker='*')

    sns.despine()
    plt.tight_layout()
    plt.show()
    
    return fig, ax