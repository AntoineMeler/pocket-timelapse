import json
import os
import glob
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

from datetime import datetime
from pysolar.solar import get_azimuth, get_altitude
from dateutil import tz


def sun_angle(time: datetime, lat: float = 42.4440, lon: float = -76.5019):
    """
    Returns the sun angle given a location and time
    """
    tzinfo = tz.gettz("America/New York")
    time = time.replace(tzinfo=tzinfo)

    azimuth = get_azimuth(lat, lon, time)
    altitude = get_altitude(lat, lon, time)
    return azimuth, altitude


def datetime_to_relative(
    date: datetime, start_date: datetime, end_date: datetime
) -> float:
    delta = date - start_date
    delta = delta.days + delta.seconds / (24 * 60 * 60)

    total = end_date - start_date
    total = total.days + total.seconds / (24 * 60 * 60)

    return delta / total  # normalize to [0, 1]


class TimeLapseDataset:
    def __init__(self, path, split: str = "train", data_factor: int = 1):

        self.data_factor = data_factor

        if split == "train":
            self.image_paths = sorted(glob.glob(os.path.join(path, "*.png")))
        elif split == "test":
            self.image_paths = sorted(glob.glob(os.path.join(path + "_test", "*.png")))
        elif split == "val":
            self.image_paths = sorted(glob.glob(os.path.join(path, "*.png")))[::10]
        self.indices = np.arange(len(self.image_paths))

        self.dates, self.sun_angles = self.process_times()

    def __len__(self):
        return len(self.indices)

    def process_times(self):
        """
        Process the times in the image filenames to get a list of datetime objects.
        """
        dates = []
        sun_angles = []
        for image_path in self.image_paths:
            date = os.path.basename(image_path)
            date = date.split(".")[0]
            if "" in date:
                date = datetime.strptime(date, "%Y-%m-%dT%H%M%S")
            elif ":" in date:
                date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
            else:
                date = datetime.strptime(date, "%Y-%m-%d-%H-%M-%S")
            dates.append(date)

            azimuth, altitude = sun_angle(date)
            azimuth, altitude = azimuth / 360, altitude / 90  # normalize to [0, 1]
            angle = [azimuth, altitude]
            sun_angles.append(angle)

        start_date: datetime = min(dates)
        end_date: datetime = max(dates)

        self.start_date = start_date
        self.end_date = end_date

        start_day: datetime = start_date.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        for i in range(len(dates)):
            delta = dates[i] - start_day
            dates[i] = delta.days  # + delta.seconds / (24 * 60 * 60)

        # Evenly space the unique dates to [0, 1] range
        unique_days = sorted(list(set(dates)))
        days_linspace = np.linspace(0, 1, len(unique_days))

        for i, date in enumerate(dates):
            date_index = unique_days.index(date)
            dates[i] = days_linspace[date_index]

        sun_angles = np.array(sun_angles)

        self.unique_days = unique_days
        self.days_linspace = days_linspace

        self.time_std = np.std(dates)
        self.sun_angle_std = np.std(sun_angles, axis=0)

        self.time_gap = days_linspace[1] - days_linspace[0]

        return dates, sun_angles

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image_path = self.image_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if self.data_factor > 1:
            image = cv2.resize(
                image,
                (
                    int(image.shape[1] / self.data_factor),
                    int(image.shape[0] / self.data_factor),
                ),
                interpolation=cv2.INTER_LANCZOS4,
            )

        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if image.shape[2] == 4:  # check if image has alpha channel
            alpha = image[..., 3:4]
        else:
            alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype)
        image = image[..., :3]  # remove alpha channel if present

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        time = self.dates[index]
        angle = self.sun_angles[index]

        clouds_path = os.path.join(os.path.dirname(image_path), "clouds.json")
        if os.path.exists(clouds_path):
            with open(clouds_path, "r") as f:
                cloudiness = json.load(f)
                clouds = (
                    float(cloudiness[os.path.splitext(os.path.basename(image_path))[0]])
                    / 100
                )

        else:
            clouds = 0

        H = image.shape[0]
        fx = fy = H / 2
        cy = image.shape[0] / 2
        cx = image.shape[1] / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        camtoworlds = np.eye(4)

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": index,  # the index of the image in the dataset
            "sun_angle": torch.tensor(angle).float(),
            "time": time,
            "clouds": clouds,
            "alpha": torch.from_numpy(alpha).float(),
        }

        return data
