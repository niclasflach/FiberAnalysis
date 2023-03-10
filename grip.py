import cv2
import numpy
import math
from enum import Enum

class GripPipeline:
    """
    An OpenCV pipeline generated by GRIP.
    """
    
    def __init__(self):
        """initializes all values to presets or None if need to be set
        """


        self.desaturate_output = None

        self.__cv_threshold_0_src = self.desaturate_output
        self.__cv_threshold_0_thresh = 225.0
        self.__cv_threshold_0_maxval = 263.0
        self.__cv_threshold_0_type = cv2.THRESH_BINARY

        self.cv_threshold_0_output = None

        self.__cv_dilate_0_src = self.cv_threshold_0_output
        self.__cv_dilate_0_kernel = None
        self.__cv_dilate_0_anchor = (-1, -1)
        self.__cv_dilate_0_iterations = 1.0
        self.__cv_dilate_0_bordertype = cv2.BORDER_CONSTANT
        self.__cv_dilate_0_bordervalue = (-1)

        self.cv_dilate_0_output = None

        self.__distance_transform_input = self.cv_dilate_0_output
        self.__distance_transform_type = cv2.DIST_L1
        self.__distance_transform_mask_size = 0

        self.distance_transform_output = None

        self.__find_min_and_max_image = self.distance_transform_output
        self.__find_min_and_max_mask = None

        self.find_min_and_max_min_val = None

        self.find_min_and_max_max_val = None

        self.find_min_and_max_min_loc = None

        self.find_min_and_max_max_loc = None

        self.__cv_threshold_1_src = self.distance_transform_output
        self.__cv_threshold_1_thresh = 6.0
        self.__cv_threshold_1_maxval = 8.0
        self.__cv_threshold_1_type = cv2.THRESH_BINARY

        self.cv_threshold_1_output = None

        self.__normalize_input = self.cv_threshold_1_output
        self.__normalize_type = cv2.NORM_MINMAX
        self.__normalize_alpha = 0.0
        self.__normalize_beta = 255

        self.normalize_output = None

        self.__cv_dilate_1_src = self.normalize_output
        self.__cv_dilate_1_kernel = None
        self.__cv_dilate_1_anchor = (-1, -1)
        self.__cv_dilate_1_iterations = 3.0
        self.__cv_dilate_1_bordertype = cv2.BORDER_CONSTANT
        self.__cv_dilate_1_bordervalue = (-1)

        self.cv_dilate_1_output = None


    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step Desaturate0:
        self.__desaturate_input = source0
        (self.desaturate_output) = self.__desaturate(self.__desaturate_input)

        # Step CV_Threshold0:
        self.__cv_threshold_0_src = self.desaturate_output
        (self.cv_threshold_0_output) = self.__cv_threshold(self.__cv_threshold_0_src, self.__cv_threshold_0_thresh, self.__cv_threshold_0_maxval, self.__cv_threshold_0_type)

        # Step CV_dilate0:
        self.__cv_dilate_0_src = self.cv_threshold_0_output
        (self.cv_dilate_0_output) = self.__cv_dilate(self.__cv_dilate_0_src, self.__cv_dilate_0_kernel, self.__cv_dilate_0_anchor, self.__cv_dilate_0_iterations, self.__cv_dilate_0_bordertype, self.__cv_dilate_0_bordervalue)

        # Step Distance_Transform0:
        self.__distance_transform_input = self.cv_dilate_0_output
        (self.distance_transform_output) = self.__distance_transform(self.__distance_transform_input, self.__distance_transform_type, self.__distance_transform_mask_size)

        # Step Find_Min_and_Max0:
        self.__find_min_and_max_image = self.distance_transform_output
        (self.find_min_and_max_min_val,self.find_min_and_max_max_val,self.find_min_and_max_min_loc,self.find_min_and_max_max_loc) = self.__find_min_and_max(self.__find_min_and_max_image, self.__find_min_and_max_mask)

        # Step CV_Threshold1:
        self.__cv_threshold_1_src = self.distance_transform_output
        (self.cv_threshold_1_output) = self.__cv_threshold(self.__cv_threshold_1_src, self.__cv_threshold_1_thresh, self.__cv_threshold_1_maxval, self.__cv_threshold_1_type)

        # Step Normalize0:
        self.__normalize_input = self.cv_threshold_1_output
        (self.normalize_output) = self.__normalize(self.__normalize_input, self.__normalize_type, self.__normalize_alpha, self.__normalize_beta)

        # Step CV_dilate1:
        self.__cv_dilate_1_src = self.normalize_output
        (self.cv_dilate_1_output) = self.__cv_dilate(self.__cv_dilate_1_src, self.__cv_dilate_1_kernel, self.__cv_dilate_1_anchor, self.__cv_dilate_1_iterations, self.__cv_dilate_1_bordertype, self.__cv_dilate_1_bordervalue)


    @staticmethod
    def __desaturate(src):
        """Converts a color image into shades of gray.
        Args:
            src: A color numpy.ndarray.
        Returns:
            A gray scale numpy.ndarray.
        """
        (a, b, channels) = src.shape
        if(channels == 1):
            return numpy.copy(src)
        elif(channels == 3):
            return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        elif(channels == 4):
        	return cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
        else:
            raise Exception("Input to desaturate must have 1, 3 or 4 channels") 

    @staticmethod
    def __distance_transform(input, type, mask_size):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.array.
            type: Opencv enum.
            mask_size: The size of the mask. Either 0, 3, or 5.
        Returns:
            A black and white numpy.ndarray.
        """
        h, w = input.shape[:2]
        dst = numpy.zeros((h, w), numpy.float32)
        cv2.distanceTransform(input, type, mask_size, dst = dst)
        return numpy.uint8(dst)

    @staticmethod
    def __find_min_and_max(src, mask):
        """Finds the minimum and maximum values of the Mat as well as the associated Points.
        Args:
            src: A numpy.ndarray.
            mask: A black and white numpy.ndarray.
        Returns:
            The minimum value.
            The maximimum value.
            The point where the minimum value is located.
            The point where the maximum value is located/
        """
        return cv2.minMaxLoc(src, mask)

    @staticmethod
    def __cv_threshold(src, thresh, max_val, type):
        """Apply a fixed-level threshold to each array element in an image
        Args:
            src: A numpy.ndarray.
            thresh: Threshold value.
            max_val: Maximum value for THRES_BINARY and THRES_BINARY_INV.
            type: Opencv enum.
        Returns:
            A black and white numpy.ndarray.
        """
        return cv2.threshold(src, thresh, max_val, type)[1]

    @staticmethod
    def __normalize(input, type, a, b):
        """Normalizes or remaps the values of pixels in an image.
        Args:
            input: A numpy.ndarray.
            type: Opencv enum.
            a: The minimum value.
            b: The maximum value.
        Returns:
            A numpy.ndarray of the same type as the input.
        """
        return cv2.normalize(input, None, a, b, type)

    @staticmethod
    def __cv_dilate(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of higher value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for dilation. A numpy.ndarray.
           iterations: the number of times to dilate.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after dilation.
        """
        return cv2.dilate(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)



