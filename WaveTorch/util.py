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

import math

import sympy
import torch
import torch.nn.functional as F

from WaveTorch.dataclass import GridConfig, WiggleDescriptor


def adjust_sizes(all_size, flag_dim):
    """
    Adjust sizes based on the specified flags and ensure factors are within certain limits.

    Args:
        all_size: List of sizes to adjust.
        flag_dim: List of flags indicating whether to adjust the corresponding size.

    Returns:
        List of adjusted sizes.
    """

    assert len(all_size) == len(
        flag_dim
    ), "The lengths of all_size and flag_dim must match."

    for i, item in enumerate(all_size):
        if not flag_dim[i]:
            continue

        while True:
            factors = list(sympy.factorint(item).keys())
            if max(factors) <= 11 and (len(factors) <= 2 or sorted(factors)[-2] <= 5):
                break
            item += 1

        all_size[i] = item

    return all_size


def ensure_3d_shape(tensor, boundary_widths: int, batch_size: int = 1):

    shape = tensor.shape
    original_dim = len(shape)

    if original_dim > 3:
        raise ValueError(
            "The tensor has more than 3 dimensions, which is not supported."
        )

    three_shape = list(shape + (1,) * (3 - original_dim))
    boundary_sizes = [
        boundary_widths if d else 0
        for d in [original_dim >= 1, original_dim >= 2, original_dim == 3]
    ]

    adjusted_all_size = adjust_sizes(
        [dim + 2 * bw for dim, bw in zip(three_shape, boundary_sizes)],
        [original_dim >= d for d in range(1, 4)],
    )

    adjusted_all_size = [batch_size] + list(adjusted_all_size)

    low_boundaries, up_boundaries = zip(
        *(
            (
                math.ceil((adjusted - original) / 2),
                math.floor((adjusted - original) / 2),
            )
            for adjusted, original in zip(
                adjusted_all_size[1:], three_shape
            )  # Ignore batch dimension
        )
    )

    # # add batch
    low_boundaries = list(low_boundaries)
    up_boundaries = list(up_boundaries)
    roi = [low_boundaries, up_boundaries]

    # Add batch dimension flag (always True)
    flag_dim = [False] + [original_dim >= d for d in range(1, 4)]

    return (
        three_shape,
        adjusted_all_size,
        roi,
        boundary_sizes,
        flag_dim,
    )


def expand_dims(tensor, flag_dim):
    for dim, flag in enumerate(flag_dim):
        if not flag:
            tensor = tensor.unsqueeze(dim)
    return tensor


def get_grid(resourcewrapper, pixel_size, adjusted_all_size):
    x = resourcewrapper(
        pixel_size * torch.arange(adjusted_all_size[1]).reshape(1, -1, 1, 1)
    )
    y = resourcewrapper(
        pixel_size * torch.arange(adjusted_all_size[2]).reshape(1, 1, -1, 1)
    )
    z = resourcewrapper(
        pixel_size * torch.arange(adjusted_all_size[3]).reshape(1, 1, 1, -1)
    )

    #
    dfx = 2 * torch.pi / (pixel_size * adjusted_all_size[1])
    dfy = 2 * torch.pi / (pixel_size * adjusted_all_size[2])
    dfz = 2 * torch.pi / (pixel_size * adjusted_all_size[3])

    fx = resourcewrapper(
        dfx
        * torch.fft.fftshift(
            torch.arange(adjusted_all_size[1])
            - torch.floor(torch.tensor(adjusted_all_size[1] / 2))
        ).reshape(1, -1, 1, 1)
    )

    fy = resourcewrapper(
        dfy
        * torch.fft.fftshift(
            torch.arange(adjusted_all_size[2])
            - torch.floor(torch.tensor(adjusted_all_size[2] / 2))
        ).reshape(1, 1, -1, 1)
    )

    fz = resourcewrapper(
        dfz
        * torch.fft.fftshift(
            torch.arange(adjusted_all_size[3])
            - torch.floor(torch.tensor(adjusted_all_size[3] / 2))
        ).reshape(1, 1, 1, -1)
    )

    gridconfig = GridConfig(
        adjusted_all_size, pixel_size, dfx, fx, x, dfy, fy, y, dfz, fz, z
    )

    return gridconfig


def calculate_wavenumbers(speed, freq):
    speed_mean = torch.mean(speed)
    refractive_index = speed_mean / speed
    e_r = refractive_index**2
    e_r_min = torch.min(e_r)
    e_r_max = torch.max(e_r)
    e_r_center = (e_r_max + e_r_min) / 2
    lamb = speed_mean / freq
    k0 = 2 * torch.pi / lamb
    k0c = torch.sqrt(e_r_center) * k0
    return e_r, k0, k0c, lamb


def validate_source_positions(source, roi, adjusted_all_size):
    Bl, Br = roi
    for item in source:
        for index_dim in range(len(item.position)):
            assert (
                item.position[index_dim] >= Bl[index_dim]
                and item.position[index_dim]
                <= adjusted_all_size[index_dim] - Br[index_dim]
            ), f"source.index={item.position[index_dim]} is out of range [{Bl[index_dim]}, {adjusted_all_size[index_dim] - Br[index_dim]}]"


def make_pml(N, c, k0_grid, Bmax):
    fact_N = torch.math.factorial(N)

    def P_N(r):
        return sum((c * r) ** n / torch.math.factorial(n) for n in range(N + 1))

    def f(r):
        num = c ** (N + 1) * r ** (N - 1) * (N + (2j * k0_grid - c) * r)
        den = fact_N * P_N(r)
        return num / den / k0_grid**2

    def leakage(Bmax):
        return torch.exp(-c * Bmax) * P_N(Bmax)  # * fact_N

    return f, leakage(Bmax)


def make_coord(bl, br, adjusted_all_size):
    left = torch.arange(bl, 0, -1)  # Bl(i):-1:1
    mid = torch.zeros(adjusted_all_size - bl - br)
    right = torch.arange(1, br + 1)  # 1:Br(i)
    return torch.cat([left, mid, right]) ** 2


def add_boundary_layer(matrix, roi, mode="replicate"):
    Bl, Bu = roi
    p3d = tuple([item for pair in zip(Bl[::-1], Bu[::-1]) for item in pair])
    # p3d = tuple([item for pair in zip(Bl, Bu) for item in pair])
    if len(matrix.shape) != 4:  # padding Batch soruce matrix
        for _ in range(4 - len(matrix.shape)):
            matrix = matrix.unsqueeze(-1)  # expand to [num_src,Nx,Ny,Nz]
    matrix_padded = F.pad(matrix, p3d, mode=mode, value=0)
    return matrix_padded


def RI2potential_PBL(
    e_r, k0, k0c, lamb, adjusted_all_size, roi, sim_setting, epsilonmin=3
):

    BL, BR = roi
    dist = torch.sqrt(
        make_coord(BL[0], BR[0], adjusted_all_size[1]).reshape(1, -1, 1, 1)
        + make_coord(BL[1], BR[1], adjusted_all_size[2]).reshape(1, 1, -1, 1)
        + make_coord(BL[2], BR[2], adjusted_all_size[3]).reshape(1, 1, 1, -1)
    ).to(e_r.device)

    padded_e_r = add_boundary_layer(e_r, roi)
    k0_grid = torch.mean(padded_e_r) * 2 * torch.pi / (lamb / sim_setting.pixel_size)
    alpha = sim_setting.boundary_param * k0_grid**2 / (2 * k0_grid)
    fun_edge, leakage = make_pml(
        int(sim_setting.boundary_type[3:]), alpha, k0_grid, max(BR)
    )
    padded_e_r = padded_e_r + fun_edge(dist)

    V_tot = padded_e_r * k0**2 - k0c**2

    Vabs_max = torch.max(torch.abs(V_tot))
    epsilon = max(Vabs_max.item(), epsilonmin)
    V = V_tot - 1j * epsilon

    gamma = 1j / epsilon * V

    k02e = k0c**2 / epsilon + 1j
    return epsilon, k02e, gamma


def nuttall(N, sflag="symmetric"):
    """Nuttall 4-term Blackman-Harris window."""
    a = [0.3635819, 0.4891775, 0.1365995, 0.0106411]
    x = torch.arange(0, N) * 2.0 * torch.pi / (N if sflag=="periodic" else N-1)
    return a[0] - a[1]*torch.cos(x) + a[2]*torch.cos(2*x) - a[3]*torch.cos(3*x)

def hamming(N, sflag="symmetric"):
    """Hamming window."""
    x = torch.arange(0, N) * 2.0 * torch.pi / (N if sflag=="periodic" else N-1)
    return 0.54 - 0.46 * torch.cos(x)

def hann(N, sflag="symmetric"):
    """Hann (Hanning) window."""
    x = torch.arange(0, N) * 2.0 * torch.pi / (N if sflag=="periodic" else N-1)
    return 0.5 - 0.5 * torch.cos(x)

def blackman(N, sflag="symmetric"):
    """Blackman window."""
    x = torch.arange(0, N) * 2.0 * torch.pi / (N if sflag=="periodic" else N-1)
    return 0.42 - 0.5 * torch.cos(x) + 0.08 * torch.cos(2*x)

def kaiser(N, beta=5.0, sflag="symmetric"):
    """Kaiser window."""
    x = torch.arange(0, N, dtype=torch.float32)
    alpha = N / 2.0 if sflag == "periodic" else (N - 1) / 2.0
    ratio = (x - alpha) / alpha
    w = torch.i0(beta * torch.sqrt(1 - ratio**2)) / torch.i0(torch.tensor(beta, dtype=torch.float32))
    return w

def rectangular(N, sflag="symmetric"):
    """Rectangular window."""
    return torch.ones(N)

def linear(N, sflag="symmetric"):
    """Rectangular window."""
    n = torch.arange(1, N + 1)
    return (n - 0.21) / (N + 0.66)

def choose_win(win_name, N, **kwargs):
    win_map = {
        'nuttall': nuttall,
        'hamming': hamming,
        'hann': hann,
        'blackman': blackman,
        'kaiser':kaiser,
        'rectangular': rectangular,
        'linear':linear,
    }

    fn = win_map.get(win_name.lower())
    if fn is None:
        raise ValueError(f"Unsupported window type: {win_name}")
    return fn(N, **kwargs)


def apply_edge_filters(V, N, roi, boundary_widths,sim_setting):
    Bl, Br = roi
    for dim in range(1, len(N)):
        bl = Bl[dim - 1]  # width of added boundary
        br = Br[dim - 1]
        roi_size = N[dim] - bl - br
        if bl > 0:
            L = boundary_widths[dim - 1]  # width of boundary layer
            boundary_param = sim_setting.boundary_param
            if boundary_param == 'linear':
                window = lambda B: choose_win(boundary_param, B)
            elif boundary_param.startswith('kaiser'):
                window = lambda B: choose_win(boundary_param[:6], 2 * L - 1,beta=float(boundary_param[6:]))[:B]
            else:
                window = lambda B: choose_win(boundary_param, 2 * L - 1)[:B]
            # Add a zero to the window if number of grid points is even
            if br == bl:
                smoothstep = torch.cat([torch.tensor([0.0]), window(L - 1)])
            else:  # For odd number of grid points, zeros are already added
                smoothstep = window(L)
            # Construct the filter
            filt = torch.cat(
                [
                    torch.zeros(bl - L),
                    smoothstep,
                    torch.ones(roi_size),
                    torch.flipud(smoothstep),
                    torch.zeros(br - L),
                ]
            ).to(V.device)

            # Reshape filter to match the required dimension
            filter_shape = [1] * V.dim()
            filter_shape[dim] = filt.size(0)
            filt = filt.view(filter_shape)

            # Apply filter to potential map
            V = V * filt
    return V


def RI2potential_ARL(e_r, k0, k0c, adjusted_all_size, roi, bw_size,sim_setting, epsilonmin=3):

    padded_e_r = add_boundary_layer(e_r, roi, mode="replicate")
    V_tot = padded_e_r * k0**2 - k0c**2

    Vabs_max = torch.max(torch.abs(V_tot))
    max_index = torch.argmax(torch.abs(V_tot))
    if torch.real(V_tot.flatten()[max_index]) < 0.05 * Vabs_max:
        Vabs_max *= 1.05
    epsilon = max(Vabs_max.item(), epsilonmin)
    V = padded_e_r * k0**2 - k0c**2 - 1j * epsilon
    V_filtered = apply_edge_filters(V, adjusted_all_size, roi, bw_size, sim_setting)
    k02e = k0c**2 / epsilon + 1j
    gamma = 1j / epsilon * V_filtered
    return epsilon, k02e, gamma


def RI2Potential(e_r, k0, k0c, lamb, adjusted_all_size, roi, bw_size, sim_setting):
    if sim_setting.boundary_type.startswith('PBL'):
        return RI2potential_PBL(e_r, k0, k0c, lamb, adjusted_all_size, roi, sim_setting)
    elif sim_setting.boundary_type == 'ARL':
        return RI2potential_ARL(e_r, k0, k0c, adjusted_all_size, roi, bw_size, sim_setting)
    else:
        raise ValueError(f"Unknown boundary type: {sim_setting.boundary_type}")


def compute_green_PBL(grid, epsilon, k02e):
    sqrt_epsilon = math.sqrt(epsilon)
    # Construct coordinates, shift quarter of a pixel when wiggling
    ### using python idnex style
    pxe = (grid.fx) / sqrt_epsilon
    pye = (grid.fy) / sqrt_epsilon
    pze = (grid.fz) / sqrt_epsilon
    px2 = pxe**2
    py2 = pye**2
    pz2k = pze**2 - k02e
    return [WiggleDescriptor(px2, py2, pz2k)], 1


def wiggle_perm(wiggle_flags):
    """
    Returns matrix with all possible combinations of enabled wiggle directions.
    """
    n_directions = len(wiggle_flags)  # Number of wiggle directions considered
    n_wiggles = sum(wiggle_flags)  # Number of enabled wiggle flags
    wiggle_set = torch.zeros((n_directions, 2**n_wiggles))
    # Powers of minus one generate rows of alternating ones and minus ones.
    wiggle_set[wiggle_flags, :] = (-1) ** (
        torch.ceil(
            torch.arange(1, 2**n_wiggles + 1) / 2 ** torch.arange(n_wiggles).view(-1, 1)
        )
        + 1
    )
    return wiggle_set


def wiggle_descriptor(resourcewrapper, grid, wig, epsilon, k02e):
    """
    Calculate phase ramps and coordinates for a given wiggle permutation.
    determine all permutations of wiggle directions
    """

    sqrt_epsilon = math.sqrt(epsilon)
    # Construct coordinates, shift quarter of a pixel when wiggling
    ### using python idnex style
    pxe = (grid.fx - grid.dfx * wig[1] / 4) / sqrt_epsilon
    pye = (grid.fy - grid.dfy * wig[2] / 4) / sqrt_epsilon
    pze = (grid.fz - grid.dfz * wig[3] / 4) / sqrt_epsilon

    px2 = pxe**2
    py2 = pye**2
    pz2k = pze**2 - k02e

    # Construct real space phase gradients to compensate for the pixel shift in k_space
    gx = resourcewrapper(
        torch.exp(
            (1j * wig[1] * grid.x)
            * (torch.pi / 2)
            / (grid.pixel_size * grid.x.shape[1])
        )
    )
    gy = resourcewrapper(
        torch.exp(
            (1.0j * wig[2] * grid.y)
            * (torch.pi / 2)
            / (grid.pixel_size * grid.y.shape[2])
        )
    )
    gz = resourcewrapper(
        torch.exp(
            (1.0j * wig[3] * grid.z)
            * (torch.pi / 2)
            / (grid.pixel_size * grid.z.shape[3])
        )
    )
    gx_conj = gx.conj()
    gy_conj = gy.conj()
    gz_conj = gz.conj()

    wd = WiggleDescriptor(px2, py2, pz2k, gx, gy, gz, gx_conj, gy_conj, gz_conj)
    return wd


def compute_green_ARL(resourcewrapper, Flag_dim, grid, epsilon, k02e):
    wiggle_set = wiggle_perm(Flag_dim)
    # Calculate phase ramps and coordinates for every different wiggle
    Nwiggles = wiggle_set.size(1)
    wiggle_descriptors = [None] * Nwiggles  # Pre-allocate memory
    for w_i in range(Nwiggles):
        wiggle_descriptors[w_i] = wiggle_descriptor(
            resourcewrapper, grid, wiggle_set[:, w_i], epsilon, k02e
        )
    return wiggle_descriptors, len(wiggle_descriptors)


def compute_green(resourcewrapper, Flag_dim, grid, epsilon, k02e, sim_setting):
    if sim_setting.boundary_type.startswith('PBL'):
        return compute_green_PBL(grid, epsilon, k02e)
    elif sim_setting.boundary_type == 'ARL':
        return compute_green_ARL(resourcewrapper, Flag_dim, grid, epsilon, k02e)
    else:
        raise ValueError(f"Unknown boundary type: {sim_setting.boundary_type}")


def Get_croped_field(roi):
    Bl, Bu = roi

    def fun(dE):
        cropped_dE = dE[
            :,
            Bl[0] : -Bu[0] if Bu[0] > 0 else None,
            Bl[1] : -Bu[1] if Bu[1] > 0 else None,
            Bl[2] : -Bu[2] if Bu[2] > 0 else None,
        ]
        return cropped_dE

    return fun


class ConvergenceException(Exception):
    pass


def add_to(source, E):
    E_source = E.clone()
    E_source += source
    return E_source


def Get_Ana(freq, dh, speed, ix=0, iy=0, iz=0):
    freq = freq * 1e3
    omega = 2 * torch.pi * freq
    dh = dh / 1e3
    speed = speed
    if len(speed.shape) == 2:
        # speed = speed[:,:,0]
        [Nx, Ny] = speed.shape
        x = torch.arange(Nx)
        y = torch.arange(Ny)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        xx = (xx - ix) * dh
        yy = (yy - iy) * dh
        distance = torch.sqrt((xx) ** 2 + (yy) ** 2)

        input_hankle = omega / speed * distance

        approx_homo = (
            1j
            / 4
            * (
                torch.special.bessel_j0(input_hankle)
                + 1j * torch.special.bessel_y0(input_hankle)
            )
        )
    elif len(speed.shape) == 3:
        [Nx, Ny, Nz] = speed.shape
        x = torch.arange(Nx)
        y = torch.arange(Ny)
        z = torch.arange(Nz)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        xx = (xx - ix) * dh
        yy = (yy - iy) * dh
        zz = (zz - iz) * dh
        distance = torch.sqrt((xx) ** 2 + (yy) ** 2 + (zz) ** 2)
        approx_homo = (
            1 / (4 * torch.pi * distance) * torch.exp(-1j * omega * distance / speed)
        )
    return torch.nan_to_num(approx_homo, nan=0)
