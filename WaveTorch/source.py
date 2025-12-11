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

import numpy as np
import torch
import torch.nn as nn


class PointSource(nn.Module):
    """
    Subclass for defining the mask of a point source.
    """

    def __init__(self, Domain):
        super().__init__()
        self.Domain = Domain
        self.dim = len(self.Domain)
        self.source = torch.from_numpy(np.zeros(self.Domain, dtype=np.float32))

    def _clone(self, value):
        if torch.tensor(value).is_complex():
            matrix = self.source.clone().type(torch.complex64)
        else:
            matrix = self.source.clone().type(torch.float32)
        return matrix

    def forward(self, src_coordiante: list = None):
        if src_coordiante == None and len(src_coordiante) == 0:
            raise ValueError("Source position must be provided")
        else:
            for item in src_coordiante:
                iz = item.get("iz", None)
                ix = item["ix"]
                iy = item["iy"]
                value = item["value"]

                if (ix <= 0 and ix >= self.Domain[0]) or (
                    iy <= 0 and iy >= self.Domain[1]
                ):
                    raise ValueError("Source position must be within the grid")
                else:

                    source = self._clone(value)
                    if self.dim == 1:
                        source[ix] = value
                    elif self.dim == 2:
                        source[ix, iy] = value
                    elif self.dim == 3:
                        if iz <= 0 and iz >= self.Domain[2]:
                            raise ValueError("Source position must be within the grid")
                        else:
                            source[ix, iy, iz] = value
        return source.unsqueeze(0)
