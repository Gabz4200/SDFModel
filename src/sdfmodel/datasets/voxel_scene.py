from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VoxelSceneDataset(Dataset):
    """Dataset that samples coordinates from a voxel grid and returns (exist, RGB) targets.

    For each sample item, a fixed set of voxel coordinates is sampled from the
    target voxel grid. Targets are 4D vectors:
      [exist, red, green, blue]
    where `exist` is 1.0 for occupied voxels and 0.0 otherwise, and RGB values
    are in [0, 1] for occupied voxels or 0.0 for empty voxels.
    """

    def __init__(
        self,
        voxel_path: str | Path,
        num_samples: int = 1024,
        points_per_item: int = 256,
        bounds: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.points_per_item = points_per_item
        self.bounds = bounds

        voxel_array, self.voxel_size, self.voxel_origin = _load_voxel_grid(voxel_path)
        self.voxel_array = voxel_array
        self.grid_shape = voxel_array.shape[:3]
        self.occ_mask = voxel_array[..., 0] > 0.5
        self.color_grid = voxel_array[..., 1:]

        self._build_coordinate_tables()
        self._sample_items(seed)

    def _build_coordinate_tables(self) -> None:
        zz, yy, xx = np.mgrid[
            0 : self.grid_shape[0],
            0 : self.grid_shape[1],
            0 : self.grid_shape[2],
        ]
        self._all_coords = (
            np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.int32)
        )
        self._all_occ = self.occ_mask.reshape(-1)
        self._all_colors = self.color_grid.reshape(-1, 3)
        occ_idx = np.flatnonzero(self._all_occ)
        empty_idx = np.flatnonzero(~self._all_occ)
        self._occ_coords = self._all_coords[occ_idx]
        self._occ_colors = self._all_colors[occ_idx]
        self._empty_coords = self._all_coords[empty_idx]
        self._num_occ = len(self._occ_coords)
        self._num_empty = len(self._empty_coords)

    @property
    def voxel_array_4d(self) -> np.ndarray:
        out = np.zeros((*self.grid_shape, 4), dtype=np.float32)
        out[self.occ_mask] = np.concatenate(
            [
                np.ones((self.occ_mask.sum(), 1), dtype=np.float32),
                self.color_grid[self.occ_mask],
            ],
            axis=1,
        )
        return out

    def _sample_items(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n_occ = self.points_per_item // 2
        n_empty = self.points_per_item - n_occ
        sampled_coords = []
        sampled_targets = []
        for _ in range(self.num_samples):
            if self._num_occ > 0 and n_occ > 0:
                occ_idx = rng.integers(0, self._num_occ, size=n_occ)
                occ_coords = self._occ_coords[occ_idx]
                occ_colors = self._occ_colors[occ_idx]
                occ_targets = np.concatenate(
                    [
                        np.ones((n_occ, 1), dtype=np.float32),
                        occ_colors.astype(np.float32),
                    ],
                    axis=1,
                )
            else:
                occ_coords = np.empty((0, 3), dtype=np.int32)
                occ_targets = np.empty((0, 4), dtype=np.float32)
            if self._num_empty > 0 and n_empty > 0:
                empty_idx = rng.integers(0, self._num_empty, size=n_empty)
                empty_coords = self._empty_coords[empty_idx]
                empty_targets = np.zeros((n_empty, 4), dtype=np.float32)
            else:
                empty_coords = np.empty((0, 3), dtype=np.int32)
                empty_targets = np.empty((0, 4), dtype=np.float32)
            coords = np.concatenate([occ_coords, empty_coords], axis=0)
            targets = np.concatenate([occ_targets, empty_targets], axis=0)
            perm = rng.permutation(coords.shape[0])
            sampled_coords.append(coords[perm])
            sampled_targets.append(targets[perm])
        self.points_data = np.array(sampled_coords, dtype=np.float32)
        self.targets_data = np.array(sampled_targets, dtype=np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pts = torch.from_numpy(self.points_data[index])
        targets = torch.from_numpy(self.targets_data[index])
        return pts, targets


def _load_voxel_grid(voxel_path: str | Path) -> tuple[np.ndarray, float, np.ndarray]:
    """Load a voxel grid from a MagicaVoxel `.vox` file.

    Returns a numpy array of shape (Z, Y, X, 4) with channels:
      [exist, red, green, blue]
    """
    import struct

    with open(voxel_path, "rb") as f:
        data = f.read()

    if len(data) < 8 or data[:4] != b"VOX ":
        raise ValueError(f"Invalid .vox file: {voxel_path}")

    size = None
    voxels: list[tuple[int, int, int, int]] = []
    palette = None

    offset = 8
    while offset + 12 <= len(data):
        chunk_type = data[offset : offset + 4]
        chunk_size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        children_size = struct.unpack("<I", data[offset + 8 : offset + 12])[0]
        chunk_content = data[offset + 12 : offset + 12 + chunk_size]
        if chunk_type == b"SIZE":
            size = struct.unpack("<III", chunk_content[:12])
        elif chunk_type == b"XYZI":
            num = struct.unpack("<I", chunk_content[:4])[0]
            for i in range(num):
                vx = chunk_content[4 + i * 4]
                vy = chunk_content[5 + i * 4]
                vz = chunk_content[6 + i * 4]
                ci = chunk_content[7 + i * 4]
                voxels.append((vx, vy, vz, ci))
        elif chunk_type in (b"PAL", b"RGBA"):
            palette = []
            for i in range(256):
                r, g, b, a = chunk_content[i * 4 : i * 4 + 4]
                palette.append((r, g, b, a))
        elif chunk_type == b"MAIN":
            if size is None or not voxels:
                children_data = data[
                    offset + 12 + chunk_size : offset + 12 + chunk_size + children_size
                ]
                child_size, child_voxels, child_palette = _parse_children(children_data)
                if size is None:
                    size = child_size
                voxels.extend(child_voxels)
                if palette is None and child_palette is not None:
                    palette = child_palette
        offset += 12 + chunk_size + children_size

    if size is None:
        raise ValueError(f"Empty or invalid .vox file: {voxel_path}")

    x, y, z = size
    grid = np.zeros((z, y, x, 4), dtype=np.float32)
    default_palette = [(255, 255, 255, 255)] * 256

    for vx, vy, vz, ci in voxels:
        if ci > 0:
            color = (
                palette[ci - 1]
                if palette and ci - 1 < len(palette)
                else default_palette[ci - 1]
            )
            r, g, b, _ = color
            grid[vz, vy, vx, 0] = 1.0
            grid[vz, vy, vx, 1] = r / 255.0
            grid[vz, vy, vx, 2] = g / 255.0
            grid[vz, vy, vx, 3] = b / 255.0

    voxel_size = 1.0
    voxel_origin = np.array([-x / 2.0, -y / 2.0, -z / 2.0], dtype=np.float32)
    return grid, voxel_size, voxel_origin


def _parse_children(data: bytes):
    """Parse RIFF-style child chunks."""
    import struct

    size_out = None
    voxels: list[tuple[int, int, int, int]] = []
    palette = None

    offset = 0
    while offset + 12 <= len(data):
        chunk_type = data[offset : offset + 4]
        chunk_size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        children_size = struct.unpack("<I", data[offset + 8 : offset + 12])[0]
        chunk_content = data[offset + 12 : offset + 12 + chunk_size]
        if chunk_type == b"SIZE":
            size_out = struct.unpack("<III", chunk_content[:12])
        elif chunk_type == b"XYZI":
            num = struct.unpack("<I", chunk_content[:4])[0]
            for i in range(num):
                vx = chunk_content[4 + i * 4]
                vy = chunk_content[5 + i * 4]
                vz = chunk_content[6 + i * 4]
                ci = chunk_content[7 + i * 4]
                voxels.append((vx, vy, vz, ci))
        elif chunk_type in (b"PAL", b"RGBA"):
            palette = []
            for i in range(256):
                r, g, b, a = chunk_content[i * 4 : i * 4 + 4]
                palette.append((r, g, b, a))
        elif chunk_type == b"MAIN":
            child_size, child_voxels, child_palette = _parse_children(chunk_content)
            if size_out is None:
                size_out = child_size
            voxels.extend(child_voxels)
            if palette is None and child_palette is not None:
                palette = child_palette
        offset += 12 + chunk_size + children_size

    return size_out, voxels, palette


def build_voxel_dataloader(
    voxel_path: str | Path,
    num_samples: int = 1024,
    points_per_item: int = 256,
    batch_size: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, VoxelSceneDataset]:
    dataset = VoxelSceneDataset(
        voxel_path=voxel_path,
        num_samples=num_samples,
        points_per_item=points_per_item,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset
