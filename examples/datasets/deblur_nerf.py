import glob
import os
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from .colmap import Dataset
from .colmap_dataparser import ColmapParser


def _find_files(directory: Path, exts: List[str]) -> List[Path]:
    """Find all files in a directory that have a certain file extension.

    Args:
        directory : The directory to search for files.
        exts :  A list of file extensions to search for. Each file extension should be in the form '*.ext'.

    Returns:
        A list of file paths for all the files that were found. The list is sorted alphabetically.
    """
    assert directory.exists()
    if os.path.isdir(directory):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(directory, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(list(set(files_grabbed)))
        files_grabbed = [Path(f) for f in files_grabbed]
        return files_grabbed
    return []


class DeblurNerfDataset(Dataset):
    """DeblurNerf dataset class."""

    def __init__(
            self,
            parser: ColmapParser,
            split: str = "train",
            patch_size: Optional[int] = None,
            load_depths: bool = False,
    ):
        super().__init__(parser, split, patch_size, load_depths)
        self.parser = deepcopy(parser)
        if split == "test":
            gt_dir = parser.data_dir / "images_test"
            gt_image_paths = _find_files(gt_dir, ["*.png", "*.jpg", "*.JPG", "*.PNG"])
            num_gt_images = len(gt_image_paths)
            indices = np.arange(len(self.parser.image_names))
            self.indices = indices if parser.test_every < 1 else indices[
                indices % self.parser.test_every != 0
                ]
            assert num_gt_images == 0 or num_gt_images == len(self.parser.image_names)
            self.parser.image_paths = gt_image_paths
