from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from .deblur_nerf import DeblurNerfDataset
from .colmap_dataparser import ColmapParser


class HdrDeblurNerfDataset(DeblurNerfDataset):
    """HDR DeblurNerf dataset class."""

    def __init__(
            self,
            parser: ColmapParser,
            split: str = "train",
            patch_size: Optional[int] = None,
            load_depths: bool = False,
            frame_rate: Optional[float] = 30,
    ):
        super().__init__(parser, split, patch_size, load_depths)

        # Path to the exposure times file relative to the data path.
        self.exposure_times_path = parser.data_dir / "exposure_times.txt"
        self.timestamps_file_path = parser.data_dir / "timestamp_to_filename.txt"

        timestamps = None
        timestamps_dict = None
        if self.timestamps_file_path.exists():
            timestamps_dict = self._get_timestamps_from_file(self.timestamps_file_path)
            timestamps = [timestamps_dict[filename] for filename in parser.image_names]
            assert len(timestamps) == len(parser.image_paths)
        else:
            try:
                timestamps = self._get_timestamps_from_filenames(parser.image_paths)
            except ValueError as e:
                print(f"[WARN] {e}")
                print(f"[WARN] Using framerate {frame_rate} to generate timestamps.")
                timestamps = np.arange(len(parser.image_paths)) / frame_rate

        timestamps = np.array([float(t) for t in timestamps])
        self.parser.timestamps_begin = np.floor(timestamps[0])
        timestamps = timestamps - self.parser.timestamps_begin

        try:
            exposure_times_dict = self._read_exposure_times()
        except FileNotFoundError:
            print("[WARN] Exposure times file not found.")
            print("[WARN] Using default exposure time: 0.5 / frame_rate)")
            exposure_times = torch.tensor([0.5 / frame_rate] * len(timestamps))
        else:
            exposure_times_dict_new = {}
            if exposure_times_dict is not None:
                for k, v in exposure_times_dict.items():
                    new_key = float(k) - self.parser.timestamps_begin
                    exposure_times_dict_new[new_key] = v
            exposure_times_dict = exposure_times_dict_new
            exposure_times = torch.tensor([float(exposure_times_dict[timestamp]) for timestamp in timestamps])
            if len(exposure_times) != len(timestamps):
                raise ValueError("[ERROR] Exposure times must be equal to the number of images.")
            torch.set_printoptions(sci_mode=False, precision=15)
            # timestamps = torch.tensor([float(timestamp) for timestamp in timestamps], dtype=torch.float64)

        timestamps = torch.tensor(timestamps, dtype=torch.float32)
        self._check_timestamps(timestamps)
        # if len(timestamps) == 1:
        #     timestamps = timestamps.expand(len(outputs.image_filenames), 1)
        parser.exposure_times = exposure_times
        parser.timestamps = timestamps

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = super().__getitem__(item)
        index = self.indices[item]
        data["exposure_time"] = self.parser.exposure_times[index]
        data["timestamp"] = self.parser.timestamps[index]
        return data

    def _get_timestamps_from_file(self, filename: Path) -> Dict[str, str]:
        """
        Get the timestamps from a mapping file.
        """
        ret = {}
        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            # line format: timestamp filename
            pair = line.strip().split(" ")
            assert len(pair) == 2
            timestamp, filename = pair
            ret[filename] = timestamp
        return ret

    def _read_exposure_times(self) -> Dict[str, str]:
        """
        Read timestamp, exposure_time from the file. Format: timestamp, exposure_time.
        Returns:
            exposure_times: a dictionary of timestamps and exposure times.
        """
        with open(self.exposure_times_path, "r") as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            timestamp, exposure_time = line.strip().split(" ")
            ret[timestamp] = exposure_time
        return ret

    @staticmethod
    def _get_timestamps_from_filenames(filenames) -> List[str]:
        """
        Get the timestamps from the filenames (e.g. 4089.6637.png).
        """
        timestamps = []
        for filename in filenames:
            timestamp = Path(filename).stem
            timestamps.append(timestamp)
        # check if the timestamps can be converted to float
        for timestamp in timestamps:
            try:
                float(timestamp)
            except ValueError:
                raise ValueError(f"Invalid timestamp: {timestamp}")
        return timestamps

    @staticmethod
    def _check_timestamps(timestamps: Tensor) -> None:
        """
        Check if the timestamps are in increasing order, also with constant duration.
        """
        if len(timestamps) < 2:
            return None
        assert isinstance(timestamps, Tensor)

        # check if the time grid is strictly increasing
        if not torch.diff(timestamps).all():
            raise ValueError("Timestamps must be strictly increasing.")

    @staticmethod
    def _check_suffixes(filenames) -> List[Path]:
        """
        Check if the file path exists. if not, check if the file path with the correct suffix exists.
        """
        for i, filename in enumerate(filenames):
            if not filename.exists():
                flag_found = False
                exts = [".png", ".PNG", ".jpg", ".JPG"]
                for ext in exts:
                    new_filename = filename.with_suffix(ext)
                    if new_filename.exists():
                        filenames[i] = new_filename
                        flag_found = True
                        break
                if not flag_found:
                    print(f"[WARN] {filename} not found in the images directory.")

        return filenames

    # def _get_timesstamps_from_timestamps_to_filename(self, image_filenames):
    #     mapping = {}
    #     mapping_file_path = self.config.data / self.config.timestamps_to_filename_path
    #     with open(mapping_file_path, 'r') as file:
    #         for line in file:
    #             timestamp, filename = line.strip().split()
    #             mapping[filename] = timestamp
    #     timestamps = []
    #     for filename in image_filenames:
    #         filename = os.path.basename(filename)
    #         timestamp = mapping.get(filename)
    #         timestamps.append(timestamp if timestamp else None)  # 如果找不到对应的时间戳，则添加 None

    #     return timestamps
