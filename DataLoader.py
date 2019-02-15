import os
import sys
import utils
import numpy as np
from PIL import Image

class DataLoader():
    """
        Load image paths and the actual images for two datasets.
    """
    def __init__(self, opt):
        self.opt = opt

        # path to data
        self.path_A = os.path.expanduser(self.opt.data_A)
        self.path_B = os.path.expanduser(self.opt.data_B)

        # get image path sets
        self.image_paths_A = sorted(utils.get_image_paths(self.path_A))
        self.image_paths_B = sorted(utils.get_image_paths(self.path_B))

        # get the size of the dataset
        self.size_A = len(self.images_paths_A)
        self.size_B = len(self.images_paths_B)

    def __getitem__(self, index):
        """
            Get image at index from image set A and image set B.

            Args:
                index: Index of image in the image path list
        """
        # make sure index doesn't exceed number of images
        image_path_A = self.image_paths_A[index % self.size_A]
        image_path_B = self.image_paths_B[index % self.size_B]

        # get image as numpy array
        image_A = Image.open(image_path_A).convert('RGB')
        image_B = Image.open(image_path_B).convert('RGB')

        # set input and output channels properly based on direction of mapping
        AtoB = self.opt.direction == 'AtoB'
        self.in_channels = self.opt.in_channels if AtoB else self.opt.out_channels
        self.out_channels = self.opt.out_channels if AtoB else self.opt.in_channels

        # perform data augmentation on images
        A = utils.augment(self.opt, image_A, grayscale=(self.in_channels == 1))
        B = utils.augment(self.opt, image_B, grayscale=(self.out_channels == 1))

        return {'A': A, 'A_path': image_path_A, 'B': B, 'B_path': image_path_B}

    def __len__(self):
        """
            Return the length of a the larger dataset
        """
        return max(self.size_A, self.size_B)

    def shuffle(self):
        """
            Shuffle both sets of images in a consistent manner.
        """
        self.image_paths_A, self.image_paths_B = utils.shuffle_sets(self.image_paths_A, self.image_paths_B)