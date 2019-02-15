import os
import random
from PIL import Image
from sklearn.utils import shuffle


def shuffle_sets(set_A, set_B):
    """
        Shuffle two set in a consistent manner.
    """
    return shuffle(set_A, set_B)


def save_image(image, image_path):
    """
        Save an image to disk.

        Args:
            image: A numpy array representing an image
            image_path: The path the image is to be stored at
    """
    img = Image.fromarray(image)
    img.save(os.path.expanduser(image_path))


def get_image_paths(dir):
    """
        Get the list of image paths in a certain directory.

        Args:
            dir: Directory in which an image set resides
    """
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                      '.png', '.PNG', '.bmp', '.BMP']
    image_paths = []

    # traverse directory to obtain only paths to images
    for paths in sorted(os.walk(dir)):
        if any(paths.endswith(extensions) for extension in IMG_EXTENSIONS):
            image_paths.append(os.path.expanduser(paths))

    return image_paths


def augment(opt, image, grayscale=False, normalize=True):
    """
        Perform data augmentation on an image.

        Args:
            opt: holds info about what type of augmentation to perform on images
            image: A numpy array representing the image
    """
    augmented_image = image

    # convert the image to grayscale
    if grayscale:
        augmented_image.convert('L')

    if opt.preprocess is 'resize':
        augmented_image = __resize(augmented_image, opt.load_size)
    elif opt.preprocess is 'scale_width':
        augmented_image = __scale_width(augmented_image, opt.load_size)
    elif opt.preprocess is 'crop':
        augmented_image = __crop(augmented_image, opt.crop_size)
    elif opt.preprocess is 'resize_and_crop':
        augmented_image = __resize(augmented_image, opt.load_size)
        augmented_image = __crop(augmented_image, opt.crop_size)
    elif opt.preprocess is 'scale_width_and_crop':
        augmented_image = __scale_width(augmented_image, opt.load_size)
        augmented_image = __crop(augmented_image, opt.crop_size)

    if opt.preprocess is None:
        augmented_image = __make_power_2(augmented_image, base=4)

    if opt.flip:
        augmented_image = __flip(augmented_image)

    if opt.normalize:
        augmented_image = __normalize(augmented_image)

    return augmented_image


def __resize(image, new_size):
    """
        Resize an image to a specific height and width.
    """
    size = [new_size, new_size]
    image.resize(new_size, Image.BICUBIC)

    return image


def __scale_width(image, new_width):
    """
        Resize height and width based of a desired width.
    """
    old_width, old_height = image.size

    if old_width != new_width:
        new_height = int(new_width * (old_height / old_width))
        image.resize([new_width, new_height], Image.BICUBIC)

    return image


def __crop(image, new_size):
    """
        Crop an image to a specific height and width.
    """
    old_width, old_height = image.size
    start_height = random.randint(0, old_height - new_size)
    start_width = random.randint(0, old_width - new_size)

    # randomly crop an image
    image.crop((start_width, start_height,
                new_size + (start_width - 1), new_size + (start_height - 1)))

    return image


def __make_power_2(image, base):
    """
        Resize an image to be a power of 2.
    """
    old_width, old_height = image.size

    new_width = int((old_width / base) * base)
    new_height = int((old_height / base) * base)

    if (new_width == old_width) and (new_height == old_height)
        return image

    image.resize([new_width, new_height], Image.BICUBIC)

    return image


def __flip(image):
    """
        Randomly flip an image along the y axis.
    """
    flip = random.random() > 0.5

    if flip:
        image.transpose(Image.FLIP_LEFT_RIGHT)

    return image


def __normalize(image):
    """
        Normalize each channel by subtracting a mean of 0.5 and dividing by a
        standard deviation of 0.5
    """
    return (image - 0.5) / 0.5
