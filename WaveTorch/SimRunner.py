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
import torch.nn as nn

from WaveTorch.util import *


class SimRunner(nn.Module):
    def __init__(self, compute_resource, speed, sim_setting) -> None:
        super().__init__()
        self.compute_resource = compute_resource

        self.energy_threshold = sim_setting.energy_threshold
        self.max_iteration = sim_setting.max_iteration
        self.skipcheck = sim_setting.skipcheck
        self.batch_size = compute_resource.batch_size

        # Medium
        self.three_shape, self.adjusted_all_size, self.roi, bw_size, self.Flag_dim = (
            ensure_3d_shape(speed, sim_setting.boundary_widths)
        )
        speed = expand_dims(speed, self.Flag_dim)
        Grid = get_grid(
            compute_resource, sim_setting.pixel_size, self.adjusted_all_size
        )
        e_r, k0, k0c, lamb = calculate_wavenumbers(speed, sim_setting.freq)
        epsilon, k02e, self.gamma = RI2Potential(
            e_r, k0, k0c, lamb, self.adjusted_all_size, self.roi, bw_size, sim_setting
        )
        self.green, self.green_num = compute_green(
            compute_resource, self.Flag_dim, Grid, epsilon, k02e, sim_setting
        )
        self.source_intensity_scaler = 1j / epsilon / self.green_num
        self.get_croped_field = Get_croped_field(self.roi)

        self._get_dE = (
            self._get_dE_PBL
            if sim_setting.boundary_type.startswith('PBL')
            else self._get_dE_ARL
        )

    def _shift_src(self, src):
        shifted_src = (
            add_boundary_layer(src, self.roi, mode="constant")
            * self.source_intensity_scaler
        )
        return shifted_src

    def _get_dE_PBL(self, index, dE, Etmp):
        green = self.green[index]
        Etmp = torch.fft.fftn(Etmp, dim=[1, 2, 3])
        Etmp = Etmp / (green.px2 + green.py2 + green.pz2k)
        Etmp = torch.fft.ifftn(Etmp, dim=[1, 2, 3])
        dE = (1 - self.gamma) * dE - (1j * self.gamma**2) * Etmp
        return dE

    def _get_dE_ARL(self, index, dE, Etmp):
        wig = self.green[index]
        Etmp = torch.fft.fftn(Etmp * (wig.gx * wig.gy * wig.gz), dim=[1, 2, 3])
        Etmp = Etmp / (wig.px2 + wig.py2 + wig.pz2k)
        Etmp = torch.fft.ifftn(Etmp, dim=[1, 2, 3])
        Etmp = Etmp * (wig.gx_conj * wig.gy_conj * wig.gz_conj)
        dE = (1 - self.gamma) * dE - (1j * self.gamma**2) * Etmp
        return dE

    def _compute_energy(self, dE):
        return torch.mean(
            torch.sum(torch.abs(self.get_croped_field(dE)) ** 2, dim=(1, 2, 3))
        )

    def _check_convergence(self, iteration, dE):
        if iteration % self.skipcheck == 0:
            self.last_step_energy = self._compute_energy(dE)
        if iteration == 0:
            self.init_energy = self._compute_energy(dE)
            self.last_step_energy = self.init_energy
        if (iteration + 1) % (self.green_num) == 0:
            if (
                self.last_step_energy / self.init_energy < self.energy_threshold
            ):  # converged
                raise ConvergenceException(
                    f"Converged at iteration {iteration} and energy {self.last_step_energy} ,init_energy {self.init_energy}"
                )

    def forward(self, src, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        num_src = src.size(0)
        output = self.compute_resource(
            torch.from_numpy(np.zeros([num_src] + self.three_shape, dtype=np.complex64))
        )
        Batch = min(num_src, batch_size)
        num_batches = num_src // Batch
        for i_block in range(num_batches):
            src_mini_batch = self._shift_src(src[i_block::num_batches, ...])
            dE = self.compute_resource(
                torch.from_numpy(
                    np.zeros(
                        [src_mini_batch.size(0)] + self.adjusted_all_size[1:],
                        dtype=np.complex64,
                    )
                )
            )
            E = dE.clone()
            try:
                for iteration in range(self.max_iteration):
                    if iteration < self.green_num:
                        Etmp = add_to(src_mini_batch, dE)
                    else:
                        Etmp = dE.clone()
                    index_wiggle = iteration % self.green_num
                    dE = self._get_dE(index_wiggle, dE, Etmp)
                    E = E + dE
                    self._check_convergence(iteration, dE)
            except ConvergenceException as e:
                # print(e)
                pass
            output[i_block::num_batches, ...] = self.get_croped_field(
                E
            ) / self.get_croped_field(self.gamma)
        return output
