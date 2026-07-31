from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import sdf
import torch
from torch import nn

from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel


def create_sdf3_wrapper(
    model: nn.Module,
    embedding: torch.Tensor | None = None,
    batch_size: int = 65536,
    device: str = "cpu",
) -> sdf.d3.SDF3:
    """Wrap a PyTorch implicit SDF model into an sdf.d3.SDF3 object for rendering."""
    model = model.to(device).eval()

    if isinstance(model, CrossAttnSDFModel):
        if embedding is None:
            embedding = CrossAttnSDFModel.create_learnable_embedding(
                batch_size=1,
                seq_len=4,
                hidden_dim=model.hidden_dim,
                device=torch.device(device),
            )
        else:
            if embedding.ndim == 2:
                embedding = embedding.unsqueeze(0)
            embedding = embedding.to(device)

    def eval_sdf(points: np.ndarray) -> np.ndarray:
        num_points = points.shape[0]
        results = []

        with torch.no_grad():
            for i in range(0, num_points, batch_size):
                batch_pts = points[i : i + batch_size]
                cords_t = torch.from_numpy(batch_pts).to(
                    device=device, dtype=torch.float32
                )

                if isinstance(model, CrossAttnSDFModel):
                    # CrossAttnSDFModel expects (B, N, 3) and (B, S, D)
                    cords_batch = cords_t.unsqueeze(0)
                    out = model(cords_batch, embedding)
                    out = out.squeeze(0).squeeze(-1)
                else:
                    # Standard SDF MLP expects (B, 3) or (N, 3)
                    out = model(cords_t)
                    if out.ndim > 1:
                        out = out.squeeze(-1)

                results.append(out.cpu().numpy())

        return np.concatenate(results, axis=0)

    return sdf.d3.SDF3(eval_sdf)


def render_sdf_slice(
    sdf_obj: sdf.d3.SDF3,
    resolution: int = 256,
    x: float | None = None,
    y: float | None = None,
    z: float | None = 0.0,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
    ),
    show: bool = True,
    show_abs: bool = False,
    title: str = "Implicit SDF Cross-Section Slice",
) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    """Sample and render a 2D cross-section slice of the Signed Distance Field."""
    grid, extent, axes = sdf.core.sample_slice(
        sdf_obj,
        w=resolution,
        h=resolution,
        x=x,
        y=y,
        z=z,
        bounds=bounds,
    )

    if show:
        fig, ax = plt.subplots(figsize=(7, 6))
        display_data = np.abs(grid) if show_abs else grid
        im = ax.imshow(
            display_data,
            extent=extent,
            origin="lower",
            cmap="twilight_shifted" if not show_abs else "viridis",
        )
        ax.set_xlabel(f"{axes[0]} axis")
        ax.set_ylabel(f"{axes[1]} axis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Signed Distance")
        plt.tight_layout()
        plt.show()

    return grid, extent, axes


def export_sdf_mesh(
    sdf_obj: sdf.d3.SDF3,
    output_path: str,
    step: float = 0.05,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
    ),
    verbose: bool = False,
) -> None:
    """Generate 3D isosurface mesh via Marching Cubes and export to STL/OBJ format."""
    sdf_obj.save(output_path, step=step, bounds=bounds, verbose=verbose)


def render_sdf_3d(
    sdf_obj: sdf.d3.SDF3,
    step: float = 0.05,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
    ),
    show: bool = True,
    title: str = "3D Implicit SDF Mesh Visualization",
) -> Any:
    """Extract 3D isosurface mesh points and render interactively using matplotlib mplot3d."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    points = sdf.core.generate(
        sdf_obj, step=step, bounds=bounds, sparse=False, verbose=False
    )
    triangles = np.array(points).reshape(-1, 3, 3)

    if len(triangles) == 0:

        def fallback_eval(p: np.ndarray) -> np.ndarray:
            raw = sdf_obj(p)
            return raw - np.median(raw)

        fallback_obj = sdf.d3.SDF3(fallback_eval)
        points = sdf.core.generate(
            fallback_obj, step=step, bounds=bounds, sparse=False, verbose=False
        )
        triangles = np.array(points).reshape(-1, 3, 3)

    if show and len(triangles) > 0:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        mesh_collection = Poly3DCollection(
            triangles, alpha=0.8, edgecolor="k", linewidths=0.1
        )
        mesh_collection.set_facecolor([0.2, 0.6, 1.0])
        ax.add_collection3d(mesh_collection)

        # Set equal 3D aspect ratio bounds
        all_pts = triangles.reshape(-1, 3)
        min_b, max_b = all_pts.min(axis=0), all_pts.max(axis=0)
        ax.set_xlim(min_b[0], max_b[0])
        ax.set_ylim(min_b[1], max_b[1])
        ax.set_zlim(min_b[2], max_b[2])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    return triangles


class LiveSDFViewer:
    """Interactive Matplotlib viewer updating 2D slice or 3D mesh SDF renderings dynamically during training."""

    def __init__(
        self,
        title: str = "Live Scene SDF Reconstruction",
        resolution: int = 128,
        step: float = 0.12,
        slice_axis: str = "z",
        slice_pos: float = 0.0,
        view_mode: str = "3d",
    ) -> None:
        self.resolution = resolution
        self.step = step
        self.title = title
        self.slice_axis = slice_axis
        self.slice_pos = slice_pos
        self.view_mode = view_mode if view_mode in ("2d", "3d") else "3d"

        self.fig = plt.figure(figsize=(7, 6))
        self.im: Any = None
        self.mesh_collection: Any = None
        self.ax: Any = None

        if self.view_mode == "3d":
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.ax.set_xlim(-1.0, 1.0)
            self.ax.set_ylim(-1.0, 1.0)
            self.ax.set_zlim(-1.0, 1.0)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
        else:
            self.ax = self.fig.add_subplot(111)

        plt.ion()
        self.fig.show()

    def update(self, sdf_obj: sdf.d3.SDF3, step: int, loss: float) -> None:
        if self.view_mode == "3d":
            points = sdf.core.generate(
                sdf_obj,
                step=self.step,
                bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                sparse=False,
                verbose=False,
            )
            triangles = np.array(points).reshape(-1, 3, 3)

            if len(triangles) == 0:

                def fallback_eval(p: np.ndarray) -> np.ndarray:
                    raw = sdf_obj(p)
                    return raw - np.median(raw)

                fallback_obj = sdf.d3.SDF3(fallback_eval)
                points = sdf.core.generate(
                    fallback_obj,
                    step=self.step,
                    bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                    sparse=False,
                    verbose=False,
                )
                triangles = np.array(points).reshape(-1, 3, 3)

            if self.mesh_collection is not None:
                self.mesh_collection.remove()
                self.mesh_collection = None

            if len(triangles) > 0:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                self.mesh_collection = Poly3DCollection(
                    triangles, alpha=0.8, edgecolor="k", linewidths=0.1
                )
                self.mesh_collection.set_facecolor([0.2, 0.6, 1.0])
                self.ax.add_collection3d(self.mesh_collection)

            self.ax.set_title(
                f"{self.title} (3D Mesh)\n(Step {step} | MSE Loss {loss:.6f})"
            )
        else:
            x_val = self.slice_pos if self.slice_axis == "x" else None
            y_val = self.slice_pos if self.slice_axis == "y" else None
            z_val = self.slice_pos if self.slice_axis == "z" else None

            grid, extent, axes = sdf.core.sample_slice(
                sdf_obj,
                w=self.resolution,
                h=self.resolution,
                x=x_val,
                y=y_val,
                z=z_val,
                bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            )

            if self.im is None:
                self.im = self.ax.imshow(
                    grid,
                    extent=extent,
                    origin="lower",
                    cmap="twilight_shifted",
                )
                self.ax.set_xlabel(f"{axes[0]} axis")
                self.ax.set_ylabel(f"{axes[1]} axis")
                self.fig.colorbar(self.im, ax=self.ax, label="Signed Distance")
            else:
                self.im.set_data(grid)
                self.im.set_clim(vmin=grid.min(), vmax=grid.max())

            self.ax.set_title(
                f"{self.title} (2D Slice)\n(Step {step} | MSE Loss {loss:.6f})"
            )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self, keep_open: bool = True) -> None:
        if keep_open and plt.fignum_exists(self.fig.number):
            plt.ioff()
            self.ax.set_title(f"{self.title} (Final Reconstruction)")
            self.fig.canvas.draw_idle()
            plt.show()
        else:
            plt.ioff()
            plt.close(self.fig)


def render_interactive_interpolation(
    model: nn.Module,
    embeddings: torch.Tensor,
    step: float = 0.10,
    resolution: int = 128,
    view_mode: str = "3d",
    device: str = "cpu",
    title: str = "Interactive Primitive Embedding Interpolation",
) -> None:
    """Render an interactive Matplotlib GUI window with sliders to interpolate object embeddings in real time."""
    from matplotlib.widgets import Slider

    model = model.to(device).eval()
    base_emb = embeddings.detach().to(device)
    if base_emb.ndim == 2:
        base_emb = base_emb.unsqueeze(0)

    num_objs = base_emb.shape[1]
    alphas = [0.0] * num_objs

    fig = plt.figure(figsize=(9, 8))
    ax: Any = None

    if view_mode == "3d":
        ax = cast(Any, fig.add_axes((0.1, 0.28, 0.8, 0.68), projection="3d"))
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax = cast(Any, fig.add_axes((0.15, 0.28, 0.7, 0.68)))

    slider_axes = [
        fig.add_axes((0.25, 0.20 - i * 0.04, 0.55, 0.03)) for i in range(num_objs)
    ]
    sliders = [
        Slider(
            slider_axes[i],
            f"Primitive {i + 1} Blend",
            0.0,
            1.0,
            valinit=0.0,
            valfmt="%.2f",
        )
        for i in range(num_objs)
    ]

    current_mesh: Any = None
    current_im: Any = None

    def compute_interpolated_embeddings() -> torch.Tensor:
        current_emb = base_emb.clone()
        for i in range(num_objs):
            a = alphas[i]
            target_idx = (i + 1) % num_objs
            current_emb[0, i] = (1.0 - a) * base_emb[0, i] + a * base_emb[0, target_idx]
        return current_emb

    def update_render(_val: float | None = None) -> None:
        nonlocal current_mesh, current_im
        for i in range(num_objs):
            alphas[i] = float(sliders[i].val)

        curr_emb = compute_interpolated_embeddings()
        sdf_obj = create_sdf3_wrapper(model, embedding=curr_emb, device=device)

        if view_mode == "3d":
            points = sdf.core.generate(
                sdf_obj,
                step=step,
                bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                sparse=False,
                verbose=False,
            )
            triangles = np.array(points).reshape(-1, 3, 3)

            if len(triangles) == 0:

                def fallback_eval(p: np.ndarray) -> np.ndarray:
                    raw = sdf_obj(p)
                    return raw - np.median(raw)

                fallback_obj = sdf.d3.SDF3(fallback_eval)
                points = sdf.core.generate(
                    fallback_obj,
                    step=step,
                    bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                    sparse=False,
                    verbose=False,
                )
                triangles = np.array(points).reshape(-1, 3, 3)

            if current_mesh is not None:
                current_mesh.remove()
                current_mesh = None

            if len(triangles) > 0:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                current_mesh = Poly3DCollection(
                    triangles, alpha=0.85, edgecolor="k", linewidths=0.1
                )
                current_mesh.set_facecolor([0.2, 0.6, 1.0])
                ax.add_collection3d(current_mesh)

            alpha_str = ", ".join(f"P{i + 1}:{a:.2f}" for i, a in enumerate(alphas))
            ax.set_title(f"{title} (3D)\n({alpha_str})")
        else:
            grid, extent, axes = sdf.core.sample_slice(
                sdf_obj,
                w=resolution,
                h=resolution,
                z=0.0,
                bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            )
            if current_im is None:
                current_im = ax.imshow(
                    grid,
                    extent=extent,
                    origin="lower",
                    cmap="twilight_shifted",
                )
                ax.set_xlabel(f"{axes[0]} axis")
                ax.set_ylabel(f"{axes[1]} axis")
                fig.colorbar(current_im, ax=ax, label="Signed Distance")
            else:
                current_im.set_data(grid)
                current_im.set_clim(vmin=grid.min(), vmax=grid.max())

            alpha_str = ", ".join(f"P{i + 1}:{a:.2f}" for i, a in enumerate(alphas))
            ax.set_title(f"{title} (2D Slice)\n({alpha_str})")

        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update_render)

    update_render()
    plt.show()
