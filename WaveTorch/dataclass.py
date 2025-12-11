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


from dataclasses import dataclass

import torch


@dataclass
class GridConfig:
    adjusted_all_size: list
    pixel_size: list
    dfx: float
    fx: float
    x: torch.Tensor
    dfy: float
    fy: float
    y: torch.Tensor
    dfz: float
    fz: float
    z: torch.Tensor


class WiggleDescriptor:
    px2: torch.Tensor
    py2: torch.Tensor
    pz2k: torch.Tensor
    gx: torch.Tensor
    gy: torch.Tensor
    gz: torch.Tensor
    gx_conj: torch.Tensor
    gy_conj: torch.Tensor
    gz_conj: torch.Tensor

    def __init__(
        self,
        px2=None,
        py2=None,
        pz2k=None,
        gx=None,
        gy=None,
        gz=None,
        gx_conj=None,
        gy_conj=None,
        gz_conj=None,
    ):
        self.px2 = px2
        self.py2 = py2
        self.pz2k = pz2k
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.gx_conj = gx_conj
        self.gy_conj = gy_conj
        self.gz_conj = gz_conj


@dataclass
class SimSetting:
    energy_threshold: float
    max_iteration: float
    freq: float
    pixel_size: float = 0
    boundary_type: str = (
        "ARL" # ARL or PBL ,
              # such as (ARL: Absorbing boundary layer, 
              # PBL1: 1-order Polynomial boundary layer, 
              # PBL2: 2-order Polynomial boundary layer etc.)
    )
    boundary_param: float | str = 'kaiser3', # boundary strength in PBL boundary type or window function in ARL boundary type
    boundary_widths: float = 0  # number of grid points per direction
    def __post_init__(self):
        self.max_iteration = int(self.max_iteration)
        self.skipcheck = 8  # doesn't need to be changed

    def __repr__(self):
        # Provide a compact, informative representation. Format floats with
        # limited precision to keep the string readable.
        return (
            f"SimSetting(energy_threshold={self.energy_threshold:.6g}, "
            f"max_iteration={self.max_iteration}, "
            f"boundary_widths={self.boundary_widths:.6g}, "
            f"freq={self.freq:.6g}, "
            f"pixel_size={self.pixel_size:.6g}, "
            f"skipcheck={getattr(self, 'skipcheck', None)})"
        )


class ConvergenceException(Exception):
    pass
