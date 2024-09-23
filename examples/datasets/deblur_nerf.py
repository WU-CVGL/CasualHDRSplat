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
    if directory.exists() and os.path.isdir(directory):
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
        if self.parser.nvs_on_contiguous_images:
            print(f"[INFO] Using contiguous images for NVS eval")
            parser.test_every = 0
        else:
            # find the file named `hold=n` , n is the eval_interval to be recognized
            hold_file = [f for f in os.listdir(parser.data_dir) if f.startswith('hold=')]
            if len(hold_file) == 0:
                print(f"[INFO] defaulting hold={parser.test_every}")
            else:
                parser.test_every = int(hold_file[0].split('=')[-1])
                print(f"[INFO] found hold={parser.test_every}")
        if split == "train" and parser.test_every < 1:
            split = "all"

        super().__init__(parser, split, patch_size, load_depths)

        # "test" for deblur, "val" for novel-view
        if split == "val" and parser.test_every < 1:
            self.indices = []
            return

        if split == "val" or split == "test":
            self.parser = deepcopy(parser)
            if self.parser.factor > 1:
                image_dir_suffix = f"_{self.parser.factor}"
            else:
                image_dir_suffix = ""
            gt_dir = parser.data_dir / ("images_test" + image_dir_suffix)

            if gt_dir.exists():
                # Will find both deblurring & NVS eval image files in `images_test` folder.
                gt_image_paths = _find_files(gt_dir, ["*.png", "*.jpg", "*.JPG", "*.PNG"])
                num_gt_images = len(gt_image_paths)
                indices = np.arange(num_gt_images)
                self.indices = indices if parser.test_every < 1 else indices[
                    indices % self.parser.test_every != 0 if split == "test" else
                        indices % self.parser.test_every == 0
                    ]
                assert num_gt_images == 0 or num_gt_images == len(self.parser.image_names)
                self.parser.image_paths = gt_image_paths
                self.parser.image_names = [image_path.stem for image_path in gt_image_paths]
                if self.parser.nvs_on_contiguous_images:
                    if split == 'val':
                        start_indices = self.indices[:self.parser.valstart]
                        end_indices = self.indices[-self.parser.valend:]
                        self.indices = np.concatenate((start_indices, end_indices))
                    else:
                        self.indices = self.indices[self.parser.valstart:-self.parser.valend]
            else:
                if split == "test":
                    # No deblurring eval images found
                    self.indices = []
                    return
                elif split == "val":
                    # Fallback to original Dataset split. Will find NVS eval image files in `images` folder.
                    self.parser = parser
