import os
import numpy as np
import cv2

from matplotlib import pyplot as plt
from numpy.typing import NDArray


WAIT_MS: int = 5 * 10 ** 3


def display_and_save_image(output_path: str, output_files_dct: dict[str, NDArray[np.uint8]]) -> None:
    """Display and save multiple images in directory output_path."""
    os.makedirs(output_path, exist_ok=True)

    for name, image in output_files_dct.items():
        cv2.imshow(name.upper(), image)
        cv2.waitKey(WAIT_MS)
        cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(output_path, name + ".jpg"), image)


def display_image(input_path: str) -> None:
    """
    Loads an image from the given path and displays it in a window for 5 seconds.

    :param input_path: Path to image file.
    :return: None.
    """
    image: NDArray[np.uint8] | None = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    cv2.imshow("Original image", image)
    cv2.waitKey(WAIT_MS)
    cv2.destroyAllWindows()


def convert_and_histogram(input_path: str, output_path: str) -> None:
    """
    Convert an image from the given path to grayscale and HSV, display and save histograms.

    :param input_path: Path to image file.
    :param output_path: Directory to save resulting images.
    :return: None.
    """
    image: NDArray[np.uint8] | None = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    # From RGB to Grayscale
    grayscale_image: NDArray[np.uint8] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # From RGB to HSV
    hsv_image: NDArray[np.uint8] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Brightness histograms for original and grayscale images
    original_hist: NDArray[np.float32] = cv2.calcHist(
        [image], [0], None, [256], [0, 256]
    )
    grayscale_hist: NDArray[np.float32] = cv2.calcHist(
        [grayscale_image], [0], None, [256], [0, 256]
    )

    # Dict for filename: image
    output_files_dct: dict[str, NDArray[np.uint8]] = {
        "grayscale": grayscale_image,
        "hsv": hsv_image,
    }

    # Display and save images
    display_and_save_image(output_path, output_files_dct)

    # Plot and save histograms
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Brightness Histogram Original Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(original_hist, color='black')
    plt.xlim([0, 256])
    # plt.savefig(os.path.join(output_path, "brightness_histogram_original.png"))

    plt.subplot(1, 2, 2)
    plt.title("Brightness Histogram Grayscale Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(grayscale_hist, color='black')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "brightness_histograms.png"))
    plt.close()
    # plt.show()


def apply_filters(input_path: str,
          output_path: str,
          kernel_size_large: int = 7,
          kernel_size_small: int = 3,
          sigma_large: float = 5.0,
          sigma_small: float = 1e-3
          ) -> None:
    """
    Apply several image filters (Median, Gaussian, Laplacian) to the image from the given path,
    display the filtered results, and save each processed image to disk.

    :param input_path: Path to the image file.
    :param output_path: Directory to save resulting images.
    :param kernel_size_large: Kernel size for the larger median filter.
    :param kernel_size_small: Kernel size for the smaller median filter.
    :param sigma_large: Sigma value for Gaussian smoothing (larger sigma).
    :param sigma_small: Sigma value for Gaussian smoothing (smaller sigma).
    :return: None.
    """
    # Load image (Grayscale image is assumed)
    image: NDArray[np.uint8] | None = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    # Apply median blur with 2 different kernel sizes
    median_blur_large: NDArray[np.uint8] = cv2.medianBlur(image, kernel_size_large)
    median_blur_small: NDArray[np.uint8] = cv2.medianBlur(image, kernel_size_small)

    # Apply Gaussian smoothing with small and large sigma values
    gaussian_grayscale_sigma_small: NDArray[np.uint8] = cv2.GaussianBlur(
        image,
        (kernel_size_large, kernel_size_large),
        sigma_small
    )
    gaussian_grayscale_sigma_large: NDArray[np.uint8] = cv2.GaussianBlur(
        image,
        (kernel_size_large, kernel_size_large),
        sigma_large
    )

    # Apply Laplacian filter to image and convert result to displayable 8-bit image
    init_laplacian: NDArray[np.float64] = cv2.Laplacian(image, cv2.CV_64F, ksize=1)

    # Take absolute values and convert image to int
    laplacian_grayscale: NDArray[np.uint8] = np.uint8(np.absolute(init_laplacian))

    # Dict for filename: image
    output_files_dct: dict[str, NDArray[np.uint8]] = {
        f"median_blur_kernel_{kernel_size_large}": median_blur_large,
        f"median_blur_kernel_{kernel_size_small}": median_blur_small,
        f"gaussian_blur_sigma_{sigma_small}": gaussian_grayscale_sigma_small,
        f"gaussian_blur_sigma_{sigma_large}": gaussian_grayscale_sigma_large,
        "laplacian_filter": laplacian_grayscale,
    }

    # Display and save images
    display_and_save_image(output_path, output_files_dct)


def detect_edges_and_corners(input_path: str,
          output_path: str,
          *,
          kernel_size: int = 3,
          threshold_canny_1: int = 100,
          threshold_canny_2: int = 200,
          block_size_harris: int = 2) -> None:
    """
    Apply edge and corner detection algorithm to image.

    :param input_path: Path to the image file.
    :param output_path: Directory to save resulting images.
    :param kernel_size: Kernel size for Sobel operator and Harris detector.
    :param threshold_canny_1: First (lower) threshold for Canny edge detection algo.
    :param threshold_canny_2: Second (upper) threshold for Canny edge detection algo.
    :param block_size_harris: Block size for Harris corner detection.
    :return: None.
    """
    image: NDArray[np.uint8] | None = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    # Apply Sobel operator to image to compute horizontal and vertical gradients
    sobel_x_f: NDArray[np.float64] = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)  # Horizontal gradients
    sobel_y_f: NDArray[np.float64] = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)  # Vertical gradients

    # Convert float64 gradients to uint8 to display
    sobel_x: NDArray[np.uint8] = cv2.convertScaleAbs(sobel_x_f)
    sobel_y: NDArray[np.uint8] = cv2.convertScaleAbs(sobel_y_f)

    # Apply the Canny edge detection algorithm
    edges: NDArray[np.uint8] = cv2.Canny(image, threshold_canny_1, threshold_canny_2)

    # Apply the Harris corner detector to identify corner points on Grayscale image
    # grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert image to float 32
    grayscale_float_image: NDArray[np.float32] = np.float32(image)

    # Detect corners using Harris algorithm
    corners: NDArray[np.float32] = cv2.cornerHarris(grayscale_float_image, block_size_harris, kernel_size, 0.04)

    # Dilate corners points to make more visible
    corners_dilated: NDArray[np.float32] = cv2.dilate(corners, None)

    # Copy of original image to mark corners
    corners_original: NDArray[np.uint8] = image.copy()

    # Mark detected corners
    corners_original[corners_dilated > 0.01 * corners_dilated.max()] = 255

    # Dict for filename: image
    output_files_dct: dict[str, NDArray[np.uint8]] = {
        "sobel_horizontal_gradients": sobel_x,
        "sobel_vertical_gradients": sobel_y,
        "harris_detected_corners": corners_original,
        "canny_detected_edges": edges
    }

    # Display and save images
    display_and_save_image(output_path, output_files_dct)


def erode_and_dilate(input_path: str,
          output_path: str,
          *,
          thresh: int = 200,
          kernel_size: int = 3,
          erosion_num_iter: int = 10,
          dilation_num_iter: int = 5) -> None:
    """
    Apply binary thresholding and dilation and erosion to the grayscale image.

    :param input_path: Path to the image file.
    :param output_path: Directory to save resulting images.
    :param thresh: Threshold for binary segmentation.
    :param kernel_size: Kernel size for erosion and dilation.
    :param erosion_num_iter: Number of erosion iterations.
    :param dilation_num_iter: Number of dilation iterations.
    :return: None.
    """
    grayscale_image: NDArray[np.uint8] | None = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if grayscale_image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    # Binarize grayscale image using threshold segmentation
    _, binary_grayscale = cv2.threshold(
        grayscale_image, thresh, 255, cv2.THRESH_BINARY
    )

    kernel: NDArray[np.uint8] = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion operation to binary grayscale image
    eroded_binary_grayscale: NDArray[np.uint8] = cv2.erode(binary_grayscale, kernel, iterations=erosion_num_iter)

    # Apply dilation operation to binary grayscale image
    dilated_binary_grayscale: NDArray[np.uint8] = cv2.dilate(binary_grayscale, kernel, iterations=dilation_num_iter)

    # Dict for filename: image
    output_files_dct: dict[str, NDArray[np.uint8]] = {
        "binary_grayscale": binary_grayscale,
        "eroded_binary_grayscale": eroded_binary_grayscale,
        "dilated_binary_grayscale": dilated_binary_grayscale
    }

    # Display and save images
    display_and_save_image(output_path, output_files_dct)

