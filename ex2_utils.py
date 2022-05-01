import cv2
import numpy as np
import math


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 209088046


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    # make the padding the new signal
    signal_with_zero = np.zeros(in_signal.size + 2 * (k_size.size - 1)).astype(float)
    for i in range(in_signal.size):
        signal_with_zero[i + k_size.size - 1] = float(in_signal[i])
    conv_1d = np.zeros(in_signal.size + k_size.size - 1).astype(float)
    for i in range((in_signal.size + k_size.size - 1)):
        # multiply entries correctly
        conv_1d[i] = float((k_size.astype(float) * signal_with_zero[i:k_size.size + i]).sum().astype(float))

    return conv_1d.astype(float)




def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    kernel = np.flip(kernel)
    # setting the sizes of the kernel
    size_x_ker = kernel.shape[0]
    size_y_ker = kernel.shape[1]
    # setting the padding size
    part_x_ker = int((size_x_ker / 2))
    part_y_ker = int((size_y_ker / 2))
    size_x_img = in_image.shape[0]
    size_y_img = in_image.shape[1]
    conv_img = np.zeros(in_image.shape)
    # make the padded image
    padded_img = cv2.copyMakeBorder(
        in_image, part_y_ker, part_y_ker, part_x_ker, part_x_ker, cv2.BORDER_REPLICATE, None, 0)
    for i in range(size_y_img):
        for j in range(size_x_img):
            # multiply the correct pixels in the padded image with the kernel
            conv_img[j][i] = (padded_img[j:size_x_ker + j, i:size_y_ker + i] * kernel).sum()
            if conv_img[j][i] < 0:
                conv_img[j][i] = 0
    return np.round(conv_img)


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale image
    :return: (directions, magnitude,x_der,y_der)
    """
    kernel_x = np.array([[1, 0, -1]])
    kernel_y = kernel_x.reshape(3, 1)
    im_derive_x = cv2.filter2D(inImage, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    im_derive_y = cv2.filter2D(inImage, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    # MagG = ||G|| = (Ix^2 + Iy^2)^(0.5)
    mag = np.sqrt(np.square(im_derive_x) + np.square(im_derive_y))
    # DirectionG = tan^(-1) (Iy/ Ix)
    Direction = np.arctan2(im_derive_y, im_derive_x)
    return mag, Direction

def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    g_ker1d = np.array([1, 1])
    while len(g_ker1d) != k_size:
        g_ker1d = conv1D(g_ker1d, g_ker1d)
    g_ker1d = g_ker1d / g_ker1d.sum()
    g_ker1d = g_ker1d.reshape((1, len(g_ker1d)))
    g_ker2d = g_ker1d.T @ g_ker1d
    img = conv2D(in_image, g_ker2d)

    return img


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel = cv2.getGaussianKernel(k_size, 0)
    blur = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blur



def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    img = conv2D(img, laplacian_ker)
    zero_crossing = np.zeros(img.shape)
    for i in range(img.shape[0] - (laplacian_ker.shape[0] - 1)):
        for j in range(img.shape[1] - (laplacian_ker.shape[1] - 1)):
            if img[i][j] == 0:
                if (img[i][j - 1] < 0 and img[i][j + 1] > 0) or \
                        (img[i][j - 1] < 0 and img[i][j + 1] < 0) or \
                        (img[i - 1][j] < 0 and img[i + 1][j] > 0) or \
                        (img[i - 1][j] > 0 and img[i + 1][j] < 0):  # All his neighbors
                    zero_crossing[i][j] = 255
            if img[i][j] < 0:
                if (img[i][j - 1] > 0) or (img[i][j + 1] > 0) or (img[i - 1][j] > 0) or (img[i + 1][j] > 0):
                    zero_crossing[i][j] = 255
    return zero_crossing



laplacian_ker = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    # smoothing with 2D Gaussian filter
    smooth = cv2.GaussianBlur(img, (5, 5), 1)
    # convolve the smoothed image with the Laplacian filter
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    lap_img = cv2.filter2D(smooth, -1, laplacian, borderType=cv2.BORDER_REPLICATE)
    ans = zeroCrossing(lap_img)  # a binary image (0,1) that representing the edges
    return ans


# function that should find edges in the given image (second derivative)
# looking for {+,-} or {+,0,-}

def zeroCrossing(img: np.ndarray) -> np.ndarray:
    ans = np.zeros(img.shape)
    row = col = 1  # starting the loop from (1,1) pixel
    pairs_list = np.zeros(8)  # list of all the couples of the current pixel (those around it, in the 8 directions)
    while row < img.shape[0] - 1:
        while col < img.shape[1] - 1:
            pairs_list[0] = img[row - 1][col]  # up
            pairs_list[1] = img[row - 1][col + 1]  # top right diagonal            7  0  1
            pairs_list[2] = img[row][col + 1]  # right                              \ | /
            pairs_list[3] = img[row + 1][col + 1]  # lower right diagonal         6 - * - 2
            pairs_list[4] = img[row + 1][col]  # down                               / | \
            pairs_list[5] = img[row + 1][col - 1]  # lower left diagonal           5  4   3
            pairs_list[6] = img[row][col - 1]  # left
            pairs_list[7] = img[row - 1][col - 1]  # top left diagonal
            ans = find_edges(img, ans, pairs_list, row, col)  # update ans
            col += 1
        row += 1
    return ans


def find_edges(img: np.ndarray, ans: np.ndarray, pairs_list: np.ndarray, row: int, col: int) -> np.ndarray:
    pixel = img[row][col]
    posIndx = np.where(pairs_list > 0)[0]  # array representing where there are positive elements
    zerosIndx = np.where(pairs_list == 0)[0]  # all the indexes that there are zeros
    numNeg = pairs_list.size - posIndx.size - zerosIndx.size
    if pixel < 0:  # {+,-}
        if posIndx.size > 0:  # there is at least one positive number around
            ans[row][col] = 1.0
            print("{+,-}")
    elif pixel > 0:  # {-,+}
        if numNeg > 0:  # there is at least one negative number around
            ans[row][col] = 1.0
            print("{-,+}")
    else:  # pixel == 0, {+,0,-}
        comp_list = [pairs_list[0] < 0 and pairs_list[4] > 0, pairs_list[0] > 0 and pairs_list[4] < 0,
            pairs_list[1] < 0 and pairs_list[5] > 0, pairs_list[1] > 0 and pairs_list[5] < 0,
            pairs_list[2] < 0 and pairs_list[6] > 0, pairs_list[2] > 0 and pairs_list[6] < 0,
            pairs_list[3] < 0 and pairs_list[7] > 0, pairs_list[3] > 0 and pairs_list[7] < 0]
        if any(comp_list):
            ans[row][col] = 1.0
            print("{+,0,-}")
    return ans



def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    parts = 2
    if max_radius - min_radius < 30:
        threshold = 90
        min_radius = 16
    else:
        threshold = 120
    increase_rate = 5
    close_neighbors = 9
    num_of_angels = 180
    rows = img.shape[0]
    cols = img.shape[1]
    img_edges = cv2.Canny((img * 255).astype(np.uint8), img.shape[0], img.shape[1])
    circles = []
    # iterate over the every radius
    for radius in range(min_radius, max_radius):
        voting = np.zeros(img_edges.shape)
        # iterate over every second pixel
        for i in range(int(rows / parts)):
            for j in range(int(cols / parts)):
                # after detecting the edges they will appear white (255)
                if img_edges[i * parts, j * parts] == 255:
                    for angel in range(num_of_angels):
                        # iterate over every second angel from 0 to 360 in order to decrease run time
                        a = int(j * parts - math.cos(angel * math.pi / (num_of_angels / parts)) * radius)
                        b = int(i * parts - math.sin(angel * math.pi / (num_of_angels / parts)) * radius)
                        if 0 <= a < cols and 0 <= b < rows:
                            voting[b, a] += increase_rate
        if np.amax(voting) > threshold:
            voting[voting < threshold] = 0
            # iterate over all the pixels that can actually be circle and check the sum that they got voted
            for i in range(1, int(rows / parts) - 1):
                for j in range(1, int(cols / parts) - 1):
                    if voting[i * parts, j * parts] >= threshold:
                        avg_sum = voting[i * parts - 1:i * parts + 2,
                                  j * parts - 1: j * parts + 2].sum() / close_neighbors
                        if avg_sum >= threshold / close_neighbors:
                            ans = 0
                            # check that the size of the circle doesnt exists already
                            for x, y, r in circles:
                                if math.pow((j * parts - x), 2) + math.pow((i * parts - y), 2) < math.pow(r, 2):
                                    ans = 1
                                    break
                            if ans == 0:
                                circles.append((j * parts, i * parts, radius))
                                voting[i * parts - radius:i * parts + radius,
                                j * parts - radius: j * parts + radius] = 0
    return circles




def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    img_filter = np.zeros_like(in_image)
    width = int(np.floor(k_size / 2))  # width for padding
    img_pad = np.pad(in_image, ((width,), (width,)), 'constant', constant_values=0)  # zero padding the image
    if k_size % 2 != 0:  # k_size must be odd number
        Gauss_kernel = cv2.getGaussianKernel(abs(k_size), 1)
    else:
        Gauss_kernel = cv2.getGaussianKernel(abs(k_size) + 1, 1)

    for x in range(in_image.shape[0]):
        for y in range(in_image.shape[1]):
            pivot_v = in_image[x, y]
            neighbor_hood = img_pad[x:x + k_size,
                            y:y + k_size]
            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            combo = Gauss_kernel * diff_gau
            result = (combo * neighbor_hood / combo.sum()).sum()
            img_filter[x][y] = result

    #  Bilateral of cv2
    cv2_image = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    return cv2_image, img_filter
