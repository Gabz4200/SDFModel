# SDF Vector Model — Task Continuation

## Original Request

Add a variant of the current `CrossAttnSDFModel` that outputs a **3D vector** (origin at the evaluated coordinate, pointing toward the closest surface, magnitude = SDF signed distance) instead of only the scalar signed distance. Reuse existing gradient/normal computations for training. Add a **new vector-specific loss** (cosine similarity, magnitude difference, gradient, L2, etc.). Add a CLI flag to choose which model to train/visualize. For inference, auto-detect the model type.

## Plan (TDD — Vertical Slices)

### Phase 1: New Model Implementation
1. **Create `VectorSDFModel`** in `src/sdfmodel/models/vector_sdf.py`
   - Subclass `CrossAttnSDFModel` (DRY — inherit all backbone, PE, cross-attn layers)
   - Override `forward()` to return `(N, 3)` vector = `normal * distance`
   - Normal computed via autograd gradients (same as `compute_sdf_normals(method="autograd")`)
   - Magnitude = SDF distance (already computed by `dist_head`)
   - Output shape: `(B, N, 3)` for 3D input, `(N, 3)` for 2D input (matching CrossAttnSDFModel conventions)

2. **Register model** in `src/sdfmodel/models/__init__.py` as `"vector_sdf"`

3. **Update `create_sdf3_wrapper`** in `src/sdfmodel/render.py`
   - Detect `VectorSDFModel` instances
   - For SDF3 eval: compute `sdf_value = vector.norm(dim=-1)` (magnitude = signed distance)
   - Sign determined by direction relative to surface (or just use magnitude; fogleman/sdf expects scalar SDF)

### Phase 2: New Vector Loss Function
4. **Add `compute_vector_sdf_loss`** in `src/sdfmodel/engine/metrics.py`
   - Inputs: model, points, target_sdf, target_normals, embedding, target_vectors (computed from target_sdf * target_normals)
   - Loss terms:
     - **Cosine similarity**: between predicted vector and target vector (1 - cos_sim)
     - **Magnitude MSE**: `(pred_vector.norm() - |target_sdf|)^2`
     - **Eikonal loss**: reuse `compute_eikonal_loss` (gradient of predicted SDF magnitude should be 1)
     - **Normal cosine loss**: reuse `compute_normal_loss` (predicted normal vs target normal)
     - **L2 vector loss**: `MSE(pred_vector, target_vector)`
   - Return dict with `loss` + individual terms

### Phase 3: Training Integration
5. **Update `SceneTrainer`** in `src/sdfmodel/engine/scene_trainer.py`
   - Accept `model_type` parameter (`"cross_attn_sdf"` or `"vector_sdf"`)
   - Use `compute_vector_sdf_loss` for vector model, `compute_combined_sdf_loss` for scalar model
   - Pass `target_normals` to compute target vectors when using vector loss

6. **Update `Trainer`** in `src/sdfmodel/engine/trainer.py`
   - Accept `model_type` parameter
   - Use appropriate loss function in `train_epoch`

### Phase 4: CLI & Inference Auto-Detection
7. **Update `cli.py`** `run_train()`
   - Add `--model-type` flag (choices: `"scalar_sdf"`, `"vector_sdf"`, default `"scalar_sdf"`)
   - Pass to `Trainer`/`SceneTrainer`

8. **Update `cli.py`** `run_render()` and `run_eval_sdfmodel()`
   - Auto-detect model type from checkpoint or model class
   - For `VectorSDFModel`, `create_sdf3_wrapper` already handles conversion

9. **Update `scripts/train_scene.py`**
   - Add `--model-type` flag
   - Pass to `SceneTrainer`

### Phase 5: Tests (TDD order)
10. **Test VectorSDFModel** — shape contract, gradient flow, 2D/3D input
11. **Test vector loss** — returns dict with expected keys, gradients flow, loss > 0
12. **Test SceneTrainer with vector model** — step runs, loss decreases
13. **Test auto-detection in create_sdf3_wrapper** — vector model outputs scalar SDF correctly

## Current State (as of exploration)

### Files Read/Understood
- `src/sdfmodel/models/cross_attn_sdf.py` — Full implementation understood (CrossAttnSDFModel outputs scalar `(N, 1)` SDF)
- `src/sdfmodel/models/base.py` — BaseModel ABC with `forward(*args, **kwargs)`
- `src/sdfmodel/models/__init__.py` — Registry pattern with `register_model`, `build_model`, `list_models`
- `src/sdfmodel/models/sdf_mlp.py` — SDFMLP (scalar output, no embeddings)
- `src/sdfmodel/models/fourier_pe.py` — FourierPositionEncoding
- `src/sdfmodel/engine/metrics.py` — Full: `_eval_model`, `compute_eikonal_loss`, `compute_sdf_normals`, `compute_normal_loss`, `compute_combined_sdf_loss`, `compute_sdf_metrics`
- `src/sdfmodel/engine/scene_trainer.py` — SceneTrainer with `compute_combined_sdf_loss`, LiveSDFViewer support
- `src/sdfmodel/engine/trainer.py` — Trainer with MSE loss, AMP support, checkpointing
- `src/sdfmodel/render.py` — `create_sdf3_wrapper` (auto-detects CrossAttnSDFModel), `LiveSDFViewer`, rendering functions
- `src/sdfmodel/datasets/scene_sdf.py` — Scene4PrimitivesDataset returns `(pts, targets[, normals])`
- `src/sdfmodel/datasets/spatial_sdf.py` — SyntheticSDFDataset (sphere SDF)
- `src/sdfmodel/utils/config.py` — ExperimentConfig, ModelConfig, etc.
- `src/sdfmodel/cli.py` — Full CLI with train/render/train-scene/eval-sdfmodel subcommands
- `scripts/train_scene.py` — Standalone training script
- `scripts/eval_sdfmodel.py` — Interactive evaluation script
- `scripts/render_sdf.py` — Inference/rendering script

### Existing Tests (all pass)
- `tests/test_cross_attn_sdf.py` — 20 tests covering registry, shape, 2D/3D input, position invariance, gradient flow, batch independence, embedding optimization, error cases, chunked forward
- `tests/test_models.py` — 7 tests for SDFMLP + registry
- `tests/test_trainer.py` — 5 tests including eikonal loss, normals, combined loss
- `tests/test_scene_trainer.py` — 1 test for step + loss decrease
- `tests/test_scene_dataset.py` — 3 tests for dataset shapes, dataloader, normals
- `tests/test_datasets.py` — 2 tests for SyntheticSDFDataset + dataloaders
- `tests/test_render.py` — 10+ tests for SDF3 wrapper, slicing, mesh export, viewer modes

## What Has Been Done
- Codebase exploration complete — all relevant files read and understood
- Architecture plan designed (VectorSDFModel inherits CrossAttnSDFModel, reuses normal/gradient computation)
- TDD plan defined with 5 phases and 13 test cases

## What Still Needs to Be Done
- [ ] **Phase 1**: Implement `VectorSDFModel` class + register in `__init__.py`
- [ ] **Phase 1**: Update `create_sdf3_wrapper` for vector model auto-detection
- [ ] **Phase 2**: Implement `compute_vector_sdf_loss` in `metrics.py`
- [ ] **Phase 3**: Update `SceneTrainer` to accept model_type + use vector loss
- [ ] **Phase 3**: Update `Trainer` to accept model_type + use vector loss
- [ ] **Phase 4**: Add `--model-type` CLI flag to `cli.py`
- [ ] **Phase 4**: Update `scripts/train_scene.py` with `--model-type` flag
- [ ] **Phase 5**: Write tests (TDD order: red → green → refactor)
- [ ] **Phase 5**: Run test suite to verify all tests pass
- [ ] **Phase 5**: Run existing tests to verify no regressions

## DRY Principle
- `VectorSDFModel` inherits from `CrossAttnSDFModel` — all backbone (Fourier PE, cross-attn transformer blocks, MLP) is reused
- Loss function reuses `compute_eikonal_loss`, `compute_sdf_normals`, `compute_normal_loss` from `metrics.py`
- `create_sdf3_wrapper` handles both model types with isinstance detection
- CLI rendering/evaluation auto-detects model type from class
