# Relative joint regularization

## Adopted default

The online VO sample and callers using `vo.online_graph.DEFAULT_PARAMS` use the
bounded canonical graph (`canonical_marginal`, `canonical_cap=True`,
`canonical_max_keyframes=12`, `loop_window=60`, so the active bound is W=60), the
bearing loop observation model (`bearing_sigma_d=0.2`, quadratic direction
kernel) and relative joint soft (Wiener) regularization: alpha=10, tau=1, all
three sigma parameters=1, NL-Reg disabled, TSVD disabled, `loop_robust=none`, and
20 iterations. Each segment solves only the active reduced-graph keyframes;
exited keyframes are Schur-absorbed and bearing-loop endpoints are freed by
retiring the oldest loops (`n_retired_loops`). Per-segment cost is bounded and no
longer grows with sequence length. Regularization is solver-only and is never
marginalized as observations.

To restore the previous ridge + SE(3) + unbounded configuration, override the
shared defaults together:

```python
params = {
    **DEFAULT_PARAMS,
    "canonical_regularization": "relative_joint_ridge",
    "loop_observation_model": "se3",
    "canonical_cap": False,
}
```

To restore the previous legacy/NL-Reg configuration, override the shared defaults
with all of the following together:

```python
params = {
    **DEFAULT_PARAMS,
    "marginalization_backend": "legacy",
    "canonical_regularization": "legacy",
    "loop_observation_model": "se3",
    "nl_reg": True,
    "loop_iterations": 10,
}
```

`OnlinePoseGraph` consumes the supplied parameter dict as-is; standalone partial
dicts are not automatically merged with `DEFAULT_PARAMS`.

The loop tuner includes inherited evaluation defaults in its study identity and
saves full effective parameters with every new trial. Reusing an SQLite file
therefore starts a separate study after this default change instead of mixing
different configurations. Older trials saved only partial parameters; replaying
those requires explicitly supplying their original settings.

## Spectral (soft/Wiener) regularization

The canonical backend supports a smooth prior on the total change from each
solve's initial relative poses and persistent log-scales. It uses the complete
unit-normalized pose/scale curvature, including cross terms. For eigenvalue
`lambda`, the regularization information is `alpha*tau/(lambda+tau)`.
There is no eigenvalue threshold, truncated direction, or scale step clamp.

Example parameter overrides for `scripts/ab_segment_boundary.py --params-json`:

```json
{
  "nl_reg": false,
  "seq_tsvd_ratio": 0,
  "tsvd_ratio": 0,
  "loop_robust": "none",
  "canonical_regularization": "relative_joint_soft",
  "relative_reg_alpha": 1.0,
  "relative_reg_tau": 1.0,
  "relative_sigma_r": 1.0,
  "relative_sigma_t": 1.0,
  "relative_sigma_l": 1.0
}
```

Select a canonical backend through `--modes canonical_marginal`. This is the
adopted default regularizer (the example above lowers `alpha`/`tau` for a
weaker prior); `relative_joint_none` (LM only) is a diagnostic alternative.

The sigma parameters describe tolerated prediction changes in radians,
internal distance units, and log-scale units. LM and step stopping use the
corresponding dimensionless increment coordinates. The prediction and spectral
weights stay fixed throughout a solve; the exact regularization residual and
Jacobian are updated at each iterate. Acceptance uses the true fixed block-Huber
objective plus squared priors and regularization.

Each structurally relative connected component has one fixed coordinate frame.
Relative marginal priors are evaluated after alignment to their own saved
component anchors. This preserves local first-order information when the stored
prior has the proper SE(3) nullspace and makes its finite common-frame transform
invariant. It changes the old prior's nonlinear extension away from its origin.
Regenerate online priors from the sequence start when comparing implementations.

Regularization never adds factor rows to marginalization. Its influence on the
estimated linearization state does remain. Variable edge scales must use stable
names (`add_named_edge`); unnamed variable scales are rejected at solve/export.

## Numerical robustness

The `soft` (Wiener) regularizer needs `chol(H0 + tau*I)`. At a huge Hessian
dynamic range `H0` can be numerically indefinite (roundoff-only negative
eigenvalues), where the Cholesky is undefined. In that case the factor is
built from a PSD projection: `eigh(0.5*(H0+H0.T))`, negatives clipped to zero,
and the symmetric inverse square root (`B^T B = alpha*tau*(P(H0)+tau*I)^-1` for
the projected `P(H0)`, the same information as the triangular factor). The
positive spectrum is untouched; the negative eigenvalues here are roundoff
artifacts at the huge dynamic range. The normal PD path still uses Cholesky
unchanged.

When the LM schedule is exhausted without finding a decreasing step, the solver
returns the last accepted iterate with `relative_diagnostics["status"] ==
"no_progress"` instead of raising (matching the legacy sliding-window solver's
no-progress semantics). The state is finite (the initial estimate if no step was
accepted) and is a valid best-effort estimate, so a single stalled window never
aborts the whole run. The stall is surfaced through the status (plus the usual
rejection/lambda diagnostics); the returned estimate is never clipped or
non-finite. The loop iteration cap is 20: 40 iterations let the solver wander
into the degenerate bearing region and crashed the room/fr2 sequences, while 20
completes them with equal desk/desk2 accuracy.

## Current limits

- Dense experimental solver, intended for small canonical windows.
- Absolute/partial pose priors and explicitly fixed input poses are rejected.
- Legacy NL-Reg, TSVD and additional loop robust kernels (including GNC) are
  rejected. Block Huber remains active; `loop_robust=none` does not disable Huber.
- Root-choice dependence, relative rotations near pi and marginalization's
  numerical rank boundaries remain separate concerns.
- This method supplies a prediction bias, not new observations or metric scale.
  Accuracy must be evaluated against both legacy and simpler ridge baselines.

`relative_diagnostics` records status, rejected trials, failure reasons, final
LM damping, accepted costs, initial spectrum and regularization/data costs.
An exhausted LM schedule returns the last accepted iterate with status
`no_progress`; the returned estimate is never clipped or non-finite.
