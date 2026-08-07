import numpy as np
import sdf
import torch
from torch.utils.data import DataLoader, Dataset


def create_4_primitives_scene() -> sdf.d3.SDF3:
    """Create a 3D scene containing 4 primitives: sphere, box, torus, capped_cylinder."""
    s1 = sdf.sphere(0.35).translate((-0.4, -0.4, 0.0))
    s2 = sdf.box((0.3, 0.3, 0.3)).translate((0.4, -0.4, 0.0))
    s3 = sdf.torus(0.25, 0.08).translate((-0.4, 0.4, 0.0))
    s4 = sdf.capped_cylinder(-sdf.Z * 0.25, sdf.Z * 0.25, 0.2).translate(
        (0.4, 0.4, 0.0)
    )

    return s1 | s2 | s3 | s4


class Scene4PrimitivesDataset(Dataset):
    """Dataset sampling near-surface and uniform 3D points for a 4-primitive scene."""

    def __init__(
        self,
        num_samples: int = 1024,
        points_per_item: int = 256,
        bounds: float = 1.0,
        seed: int = 42,
        return_normals: bool = False,
        surface_eps: float = 0.1,
        sampler: str = "chaos_game",
        chaos_iters: int = 4,
        chaos_jitter: float = 0.05,
    ) -> None:
        super().__init__()
        if sampler not in ("chaos_game", "rejection"):
            raise ValueError(
                f"Unknown sampler '{sampler}' (expected 'chaos_game' or 'rejection')"
            )
        self.num_samples = num_samples
        self.points_per_item = points_per_item
        self.return_normals = return_normals
        self.bounds = bounds
        self.surface_eps = surface_eps
        self.sampler = sampler
        self.chaos_iters = chaos_iters
        self.chaos_jitter = chaos_jitter
        self.scene = create_4_primitives_scene()

        rng = np.random.default_rng(seed)

        primitive_centers = np.array(
            [
                [-0.4, -0.4, 0.0],
                [0.4, -0.4, 0.0],
                [-0.4, 0.4, 0.0],
                [0.4, 0.4, 0.0],
            ],
            dtype=np.float32,
        )
        self.primitive_centers = primitive_centers

        n_near = points_per_item // 2
        n_uniform = points_per_item - n_near

        sampled_points = []
        for _ in range(num_samples):
            if sampler == "chaos_game":
                near = self._sample_chaos_game(n_near, rng)
            else:
                near = self._sample_near_surface(n_near, rng)
            uniform = rng.uniform(-bounds, bounds, size=(n_uniform, 3)).astype(
                np.float32
            )

            pts = np.vstack([near, uniform])
            sampled_points.append(pts)

        self.points_data = np.array(sampled_points, dtype=np.float32)
        flat_points = self.points_data.reshape(-1, 3)
        flat_sdfs = self.scene(flat_points).squeeze(-1).astype(np.float32)
        self.sdf_data = flat_sdfs.reshape(num_samples, points_per_item, 1)

        flat_normals = self._scene_normals(flat_points).astype(np.float32)
        self.normal_data = flat_normals.reshape(num_samples, points_per_item, 3)

    def _scene_normals(self, points: np.ndarray) -> np.ndarray:
        """Unit surface normals of the analytic scene via central finite differences."""
        h = 1e-4
        ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        dx = (self.scene(points + h * ex) - self.scene(points - h * ex)) / (2 * h)
        dy = (self.scene(points + h * ey) - self.scene(points - h * ey)) / (2 * h)
        dz = (self.scene(points + h * ez) - self.scene(points - h * ez)) / (2 * h)

        flat_grads = np.hstack([dx, dy, dz])
        norms = np.linalg.norm(flat_grads, axis=-1, keepdims=True)
        return flat_grads / np.maximum(norms, 1e-8)

    def _sample_chaos_game(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Chaos-game surface sampler.

        Analog of the classic chaos game (IFS attractor rendering): seed random
        particles, then repeatedly (1) project each particle onto the zero level
        set via the analytic SDF (``p <- p - f(p) * n(p)`` — exact in one step
        for a union SDF with unit gradient) and (2) apply a random jitter so the
        particles keep exploring the surface, like the random transform choice
        in an IFS. After a short warm-up every particle sits on the surface —
        including thin structures such as the torus rim that rejection sampling
        statistically misses. A final normal offset spreads the signed-distance
        targets inside the ``surface_eps`` band.
        """
        if n == 0:
            return np.empty((0, 3), dtype=np.float32)
        pts = rng.uniform(-self.bounds, self.bounds, size=(n, 3)).astype(np.float32)
        for _ in range(self.chaos_iters):
            f = self.scene(pts).squeeze(-1)
            normals = self._scene_normals(pts)
            pts = pts - f[:, None] * normals
            pts = pts + rng.normal(0.0, self.chaos_jitter, size=pts.shape).astype(
                np.float32
            )
        f = self.scene(pts).squeeze(-1)
        normals = self._scene_normals(pts)
        pts = pts - f[:, None] * normals
        pts = (
            pts
            + rng.normal(0.0, self.surface_eps / 3.0, size=pts.shape).astype(np.float32)
            * normals
        )
        return pts

    def _sample_near_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Rejection-sample points within ``surface_eps`` of the analytic zero level set."""
        if n == 0:
            return np.empty((0, 3), dtype=np.float32)
        candidates_per_draw = max(n * 16, 1024)
        collected: list[np.ndarray] = []
        remaining = n
        for _ in range(8):
            if remaining <= 0:
                break
            cands = rng.uniform(
                -self.bounds, self.bounds, size=(candidates_per_draw, 3)
            ).astype(np.float32)
            f = self.scene(cands).squeeze(-1)
            mask = np.abs(f) < self.surface_eps
            accepted = cands[mask]
            if len(accepted) > 0:
                take = min(len(accepted), remaining)
                collected.append(accepted[:take])
                remaining -= take
        pts = (
            np.concatenate(collected)
            if collected
            else np.empty((0, 3), dtype=np.float32)
        )
        if pts.shape[0] < n:
            center_indices = rng.integers(0, 4, size=n - pts.shape[0])
            extra = self.primitive_centers[center_indices] + rng.normal(
                0.0, 0.25, size=(n - pts.shape[0], 3)
            ).astype(np.float32)
            pts = np.concatenate([pts, extra]) if pts.shape[0] else extra
        return pts[:n]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pts = torch.from_numpy(self.points_data[index])
        targets = torch.from_numpy(self.sdf_data[index])
        if self.return_normals:
            normals = torch.from_numpy(self.normal_data[index])
            return pts, targets, normals
        return pts, targets


def build_scene_dataloader(
    num_samples: int = 512,
    points_per_item: int = 256,
    batch_size: int = 2,
    seed: int = 42,
    return_normals: bool = False,
    surface_eps: float = 0.1,
    sampler: str = "chaos_game",
    chaos_iters: int = 4,
    chaos_jitter: float = 0.05,
) -> DataLoader:
    dataset = Scene4PrimitivesDataset(
        num_samples=num_samples,
        points_per_item=points_per_item,
        seed=seed,
        return_normals=return_normals,
        surface_eps=surface_eps,
        sampler=sampler,
        chaos_iters=chaos_iters,
        chaos_jitter=chaos_jitter,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
