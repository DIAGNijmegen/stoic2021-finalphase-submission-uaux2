#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.preprocessing.image import Iterator
import numpy as np
# AUCMEDI libraries
from aucmedi import DataGenerator
from aucmedi.data_processing.io_loader import image_loader

#-----------------------------------------------------#
#          Custom Data Generator for Metadata         #
#-----------------------------------------------------#
""" Custom AUCMEDI DataGenerator to pass also Metadata.
"""
class DataGenerator_MetaData(Iterator):
    #-----------------------------------------------------#
    #                    Initialization                   #
    #-----------------------------------------------------#
    """Initialization function of the Data Generator which acts as a configuraiton hub.

        If using for prediction, the 'labels' parameter have to be None.
        Data augmentation is applied even for prediction if a DataAugmentation object is provided!
        Applying 'None' to resize will result into no image resizing. Default (224, 224)

        Arguments:
            samples (List of Strings):      List of sample/index encoded as Strings.
            path_imagedir (String):         Path to the directory containing the images.
            metadata (NumPy Aarray):        NumPy Array with additional metadata. Have to be shape (n_samples, meta_variables).
            labels (NumPy Array):           Classification list with One-Hot Encoding.
            image_format (String):          Image format to add at the end of the sample index for image loading.
            batch_size (Integer):           Number of samples inside a single batch.
            resize (Tuple of Integers):     Resizing shape consisting of a X and Y size. (optional Z size for Volumes)
            subfunctions (List of Subfunctions):
                                            List of Subfunctions class instances which will be SEQUENTIALLY executed on the data set.
            img_aug (ImageAugmentation):    Image Augmentation class instance which performs diverse data augmentation techniques.
            shuffle (Boolean):              Boolean, whether dataset should be shuffled.
            grayscale (Boolean):            Boolean, whether images are grayscale or RGB.
            standardize_mode (String):      Standardization modus in which image intensity values are scaled.
            sample_weights (List of Floats):List of weights for samples.
            workers (Integer):              Number of workers. If n_workers > 1 = use multi-threading for image preprocessing.
            prepare_images (Boolean):       Boolean, whether all images should be prepared and backup to disk before training.
                                            Recommended for large images or volumes to reduce CPU computing time.
            loader (Function):              Function for loading samples/images from disk.
            seed (Integer):                 Seed to ensure reproducibility for random function.
            kwargs (Dictionary):            Additional parameters for the sample loader.
    """
    def __init__(self, samples, path_imagedir, metadata, labels=None,
                 image_format=None, batch_size=32, resize=(224, 224),
                 subfunctions=[], img_aug=None, shuffle=False, grayscale=False,
                 standardize_mode="z-score", sample_weights=None, workers=1,
                 prepare_images=False, loader=image_loader, seed=None,
                 **kwargs):
        # Cache class variables
        self.image_generator = DataGenerator(samples=samples,
                     path_imagedir=path_imagedir, labels=labels,
                     image_format=image_format, batch_size=batch_size,
                     resize=resize, subfunctions=subfunctions,
                     img_aug=img_aug, shuffle=shuffle, grayscale=grayscale,
                     standardize_mode=standardize_mode,
                     sample_weights=sample_weights, workers=workers,
                     prepare_images=prepare_images, loader=loader, seed=seed,
                     **kwargs)
        self.metadata = metadata
        # Pass initialization parameters to parent Iterator class
        size = len(samples)
        super(DataGenerator_MetaData, self).__init__(size, batch_size, shuffle, seed)

    #-----------------------------------------------------#
    #              Batch Generation Function              #
    #-----------------------------------------------------#
    """Function for batch generation given a list of random selected samples."""
    def _get_batches_of_transformed_samples(self, index_array):
        # Obtain image batch
        batch_image = self.image_generator._get_batches_of_transformed_samples(index_array)
        # Obtain meta batch
        batch_meta = self.metadata[index_array]
        # Combine batch types
        batch = []
        for i, package in enumerate(batch_image):
            if i == 0:
                batch.append([package, batch_meta])
            else : batch.append(package)
        # Return generated Batch
        return batch
