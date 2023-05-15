import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image[start_row:start_row + num_rows, start_col:start_col + num_cols, :]
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = [(i**2)/2 for i in image]
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    row_scale_factor = output_rows/input_rows
    col_scale_factor = output_cols/input_cols
    for i in range(0, output_rows):
        for j in range(0, output_cols):
            input_i = int(i / row_scale_factor)
            input_j = int(j / col_scale_factor)
            output_image[i,j,:] = input_image[input_i, input_j, :]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    rotate_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = np.matmul(rotate_mat, point)
    ### END YOUR CODE
    return out


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    ## YOUR CODE HERE
    image_center_i = input_rows/2
    image_center_j = input_cols/2
    temp = np.array([image_center_i, image_center_j])
    rotate_mat = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    for i in range(0, input_rows):
        for j in range(0, input_cols):
            point = np.array([i,j])
            # print(point)
            temp_point = np.subtract(point, temp)
            # print(temp_point)
            rotated_point = np.matmul(rotate_mat, temp_point)
            # print(rotated_point)
            temp_rotated_point = np.add(temp, rotated_point)
            # print(temp_rotated_point)
            out_i = int(temp_rotated_point[0])
            out_j = int(temp_rotated_point[1])
            # int() cause details loss, so try to implement surrounding pixels while calculate one pixel
            if (out_i in range(0,input_rows) and out_j in range(input_cols)):
                output_image[out_i, out_j, :] = input_image[i,j,:]
            if (out_i in range(0,input_rows) and out_j-1 in range(input_cols)):
                output_image[out_i, out_j-1, :] = input_image[i,j,:]
            if (out_i in range(0,input_rows) and out_j+1 in range(input_cols)):
                output_image[out_i, out_j+1, :] = input_image[i,j,:]
            if (out_i-1 in range(0,input_rows) and out_j-1 in range(input_cols)):
                output_image[out_i-1, out_j-1, :] = input_image[i,j,:]
            if (out_i-1 in range(0,input_rows) and out_j in range(input_cols)):
                output_image[out_i-1, out_j, :] = input_image[i,j,:]
            if (out_i-1 in range(0,input_rows) and out_j+1 in range(input_cols)):
                output_image[out_i-1, out_j+1, :] = input_image[i,j,:]
            if (out_i+1 in range(0,input_rows) and out_j-1 in range(input_cols)):
                output_image[out_i+1, out_j-1, :] = input_image[i,j,:]
            if (out_i+1 in range(0,input_rows) and out_j in range(input_cols)):
                output_image[out_i+1, out_j, :] = input_image[i,j,:]
            if (out_i+1 in range(0,input_rows) and out_j+1 in range(input_cols)):
                output_image[out_i+1, out_j+1, :] = input_image[i,j,:]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image
