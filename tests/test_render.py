import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

from sdfmodel.models import SDFMLP, CrossAttnSDFModel
from sdfmodel.render import (
    LiveSDFViewer,
    create_sdf3_wrapper,
    export_interpolation_frames,
    export_sdf_mesh,
    render_interactive_interpolation,
    render_sdf_slice,
)


def test_create_sdf3_wrapper_cross_attn() -> None:
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(1, 4, hidden_dim)

    sdf_obj = create_sdf3_wrapper(model, embedding=embedding)
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)

    values = sdf_obj(pts)
    assert values.shape == (2, 1) or values.shape == (2,)


def test_create_sdf3_wrapper_sdf_mlp() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=3)
    sdf_obj = create_sdf3_wrapper(model)

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    values = sdf_obj(pts)
    assert len(values) == 2


def test_create_sdf3_wrapper_large_point_chunking() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=2)
    sdf_obj = create_sdf3_wrapper(model, batch_size=1000)

    # Simulate 10,000 points chunked into batches of 1,000
    pts = np.random.randn(10000, 3).astype(np.float32)
    values = sdf_obj(pts)
    assert len(values) == 10000
    assert values.shape in ((10000, 1), (10000,))


def test_render_sdf_slice_sampling() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=2)
    sdf_obj = create_sdf3_wrapper(model)

    grid, extent, axes = render_sdf_slice(sdf_obj, resolution=32, z=0.0, show=False)
    assert grid.shape == (32, 32)
    assert axes == "YX"
    assert len(extent) == 4


def test_export_sdf_mesh(tmp_path) -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=2)
    sdf_obj = create_sdf3_wrapper(model)

    output_path = tmp_path / "test_sphere.stl"
    export_sdf_mesh(sdf_obj, str(output_path), step=0.5, verbose=False)
    assert os.path.exists(output_path)


def test_live_sdf_viewer_modes() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=2)
    sdf_obj = create_sdf3_wrapper(model)

    viewer_2d = LiveSDFViewer(view_mode="2d", resolution=32)
    assert viewer_2d.ax_orig is not None
    assert viewer_2d.ax_recon is not None
    assert viewer_2d.ax is viewer_2d.ax_recon
    viewer_2d.update(sdf_obj, step=1, loss=0.5)
    viewer_2d.close(keep_open=True)

    viewer_3d = LiveSDFViewer(view_mode="3d", step=0.5)
    assert viewer_3d.ax_orig is not None
    assert viewer_3d.ax_recon is not None
    assert viewer_3d.ax_recon._shareview is viewer_3d.ax_orig
    assert viewer_3d.ax is viewer_3d.ax_recon
    viewer_3d.update(sdf_obj, step=1, loss=0.5)
    viewer_3d.close(keep_open=False)


def test_live_sdf_viewer_side_by_side_and_synced_view() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=2)
    sdf_obj = create_sdf3_wrapper(model)

    viewer = LiveSDFViewer(view_mode="3d", step=0.5)
    assert viewer.ax_orig is not None
    assert viewer.ax_recon is not None
    assert viewer.ax_recon._shareview is viewer.ax_orig

    # Verify 3D view changes sync bidirectionally
    viewer.ax_orig.view_init(elev=35, azim=60, share=True)
    assert viewer.ax_recon.elev == 35
    assert viewer.ax_recon.azim == 60

    viewer.ax_recon.view_init(elev=15, azim=120, share=True)
    assert viewer.ax_orig.elev == 15
    assert viewer.ax_orig.azim == 120

    viewer.update(sdf_obj, step=5, loss=0.01)
    assert viewer.ax_orig.get_title() == "Original Scene"
    assert "Reconstruction (Step 5)" in viewer.ax_recon.get_title()
    viewer.close(keep_open=False)


def test_live_sdf_viewer_custom_original_sdf() -> None:
    import sdf

    custom_orig = sdf.sphere(0.5)
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=2)
    sdf_obj = create_sdf3_wrapper(model)

    viewer = LiveSDFViewer(view_mode="2d", resolution=32, original_sdf=custom_orig)
    assert viewer.original_sdf is custom_orig
    viewer.update(sdf_obj, step=1, loss=0.1)
    viewer.close(keep_open=False)


def test_live_sdf_viewer_loss_dict_display() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=2)
    sdf_obj = create_sdf3_wrapper(model)

    loss_dict = {
        "loss": 0.045,
        "mse_loss": 0.012,
        "eikonal_loss": 0.005,
        "normal_loss": 0.028,
    }

    viewer = LiveSDFViewer(view_mode="2d", resolution=32)
    viewer.update(sdf_obj, step=10, loss=loss_dict)
    assert viewer.ax is not None
    title_text = viewer.ax.get_title()
    assert "Total Loss: 0.045000" in title_text
    assert "MSE: 0.012000" in title_text
    assert "Eikonal: 0.0050" in title_text
    assert "Normal: 0.0280" in title_text
    viewer.close(keep_open=False)


def test_render_interactive_interpolation() -> None:
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embeddings = CrossAttnSDFModel.create_learnable_embedding(1, 4, hidden_dim)

    # Calling with plt.show mocked/agg shouldn't raise any errors
    render_interactive_interpolation(
        model=model,
        embeddings=embeddings,
        step=0.5,
        resolution=32,
        view_mode="2d",
    )
    render_interactive_interpolation(
        model=model,
        embeddings=embeddings,
        step=0.5,
        resolution=32,
        view_mode="3d",
    )


def test_export_interpolation_frames(tmp_path) -> None:
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embeddings = CrossAttnSDFModel.create_learnable_embedding(1, 4, hidden_dim)

    gif_path = tmp_path / "morph.gif"
    frames = export_interpolation_frames(
        model=model,
        embeddings=embeddings,
        num_frames=3,
        resolution=32,
        step=0.5,
        view_mode="2d",
        output_path=gif_path,
    )

    assert isinstance(frames, np.ndarray)
    assert frames.shape[0] == 3
    assert frames.ndim == 4
    assert frames.shape[-1] == 4  # RGBA
    assert gif_path.exists()
