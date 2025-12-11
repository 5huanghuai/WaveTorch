# This file is part of the WaveTorch project.
#
# Copyright (C) 2025 ZZ
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn


class ComputeResource(nn.Module):
    def __init__(self, device: str, batch_size: int = 1, **kwargs):
        super().__init__()
        self.batch_size: int = batch_size
        self.device: torch.device = self._get_device(device)

    def _get_device(self, device: str) -> torch.device:
        if device.lower() == "cpu":
            print("No GPU available, using CPU.")
            return torch.device("cpu")
        elif device.startswith("cuda:"):
            device_id: int = int(device.split(":")[1])
            if torch.cuda.is_available():
                num_devices: int = torch.cuda.device_count()
                if device_id >= num_devices:
                    raise ValueError(
                        f"Specified device cuda:{device_id} is out of range. Available devices are 0 to {num_devices - 1}."
                    )
                print(f"Using GPU: cuda:{device_id}")
                return torch.device(f"cuda:{device_id}")
            else:
                raise ValueError("No GPU available, but 'cuda' device was specified.")
        else:
            raise ValueError(
                "Invalid device specified. Use 'cpu' or 'cuda:{device_id}'."
            )

    def update_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device)
