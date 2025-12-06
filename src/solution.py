from functions import (
    display_image,
    convert_and_histogram,
    apply_filters,
    detect_edges_and_corners,
    erode_and_dilate
)


if __name__ == "__main__":
    original_image_path = "../examples/example.jpg"  # path with chosen original image
    output_path = "../results"  # path to save resulting images for hw tasks

    display_image(original_image_path)  # task 1

    convert_and_histogram(original_image_path, output_path)  # task 2

    apply_filters(original_image_path, output_path)  # task 3

    detect_edges_and_corners(original_image_path, output_path)  # task 4

    erode_and_dilate(original_image_path, output_path)  # task 5
