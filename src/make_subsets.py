"""
Choose fixed Subset from BSDS500 and copy them in folder

Run (from Projektfolder):
    python -m src.make_subsets --source data/BSR/BSDS500/data/images --count 25

Result:
    data/BSDS500_val/   (25 Photo from val-Split)
    data/BSDS500_test/  (25 Photo from test-Split)
    In jedem Ordner liegt zusätzlich selected_images.txt mit der Bildliste
    (diese Liste gehört in die README).

The selection process is deterministic: sort the images by filename, then extract a fixed number of items distributed evenly. 
There is no element of chance, so the resulting list is the same every time it is run.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Map each dataset split to its output folder
SPLIT_TO_OUTPUT_FOLDER = {
    "val": "BSDS500_val",
    "test": "BSDS500_test",
}


def list_images_sorted(folder):
    """
    Get all images in a folder and sort them by filename
    """
    image_paths = sorted(
        path for path in Path(folder).iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    # Stop if the folder contains no image
    if not image_paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return image_paths


def select_evenly_spaced(all_image_paths, number_of_images):
    """Select a fixed number of images evenly from the sorted list."""

    # Check if enough images are available
    if number_of_images > len(all_image_paths):
        raise ValueError(
            f"Requested {number_of_images} images, but only "
            f"{len(all_image_paths)} are available"
        )

    # Choose the positions evenly across the list
    positions = np.linspace(0, len(all_image_paths) - 1, number_of_images)
    indices = positions.round().astype(int)

    # Get the images at the selected positions
    selected_image_paths = [all_image_paths[index] for index in indices]
    assert len(set(selected_image_paths)) == number_of_images, "Duplicated images selected"
    return selected_image_paths


def main():
    # Read command-line arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--source", required=True,
        help="Path to the folder containing train, val and test"
             "(bei BSDS500: .../BSR/BSDS500/data/images)")
    argument_parser.add_argument(
        "--output", default="data",
        help="Folder will be save in BSDS500_val und BSDS500_test")
    argument_parser.add_argument(
        "--count", type=int, default=25,
        help="Number of imagé to select from each split")
    arguments = argument_parser.parse_args()

    # Process the validation and test splits
    for split_name, output_folder_name in SPLIT_TO_OUTPUT_FOLDER.items():

        source_folder = Path(arguments.source) / split_name
        output_folder = Path(arguments.output) / output_folder_name
        output_folder.mkdir(parents=True, exist_ok=True)

        all_image_paths = list_images_sorted(source_folder)
        selected_image_paths = select_evenly_spaced(all_image_paths, arguments.count)

        for image_path in selected_image_paths:
            shutil.copy2(image_path, output_folder / image_path.name)

        image_list_text = "\n".join(path.name for path in selected_image_paths)
        (output_folder / "selected_images.txt").write_text(image_list_text + "\n")

        print(f"{split_name}: {len(selected_image_paths)} of {len(all_image_paths)} "
              f"and copied them to {output_folder}")


if __name__ == "__main__":
    main()