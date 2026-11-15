# Multi-backend support plan (TensorFlow + PyTorch, with TPTP)

Internal design note for maintainers. User-facing docs would come later, once
the public API for selecting a backend is decided.

## Why

LTN today is hard-wired to TensorFlow 2 / Keras. Users increasingly want PyTorch
(the ecosystem already has a separate port, [LTNtorch](https://github.com/tommasocarraro/LTNtorch)).
At the same time, `feat/tptp-parser` (#51) adds a TPTP front-end whose training
loop is also TF-specific. If we add backends after merging TPTP without a plan,
we will re-port the parser grounding twice.

This note proposes a layered approach: keep Real Logic and the TPTP AST
backend-free, put tensor math behind a small ops surface, and isolate model /
autograd / training helpers per backend. Target backends for v1 of this work:
**TensorFlow (current) and PyTorch**. JAX is out of scope unless a concrete need
appears (YAGNI).

## Current coupling (master)

| Module | Role | TF surface |
| --- | --- | --- |
| `ltn/core.py` (~645 LOC) | Expressions, vars, predicates, functions, broadcast, quantifiers | ~76 `tf.*` uses: tensors, `GradientTape`, Keras `Model`/`Input`/`layers`, ragged masks |
| `ltn/fuzzy_ops.py` | Fuzzy connectives and aggregators | Elementwise / reduce ops only (~20 uses) |
| `ltn/utils.py` | Equality predicates, deprecated `LogitsToPredicateModel` | Same as above |

The Real Logic *ideas* (free vars, broadcast, diag, wrappers) are already
framework-agnostic. The *implementation* is not: every tensor op and every model
type is named after TF/Keras.

Hard TF-only spots (not a simple rename):

1. **Masked quantification** uses `tf.ragged.boolean_mask` then aggregates on a
   `RaggedTensor`. PyTorch has no ragged type; LTNtorch instead passes a boolean
   `mask=` into the aggregators and uses `torch.where` / masked reduces.
2. **`Variable.from_constants(..., tape=)`** checks `tf.GradientTape._recording`.
   Torch uses a different autograd model (no tape object to pass).
3. **`Predicate` / `Function`** wrap `tf.keras.Model`; factories (`MLP`,
   `FromLogits`, `Lambda`) build Keras graphs. Torch expects `nn.Module`.
4. **Trainable `Constant` / `Proposition`** use `tf.Variable` with a clip
   constraint. Torch uses `nn.Parameter` and usually clips in the step or via
   parametrizations.
5. **`tf.function` tracing** assumptions in `Expression.take` and elsewhere.

## What `feat/tptp-parser` already gets right

The branch splits the parser into three layers:

| Layer | Files | Backend? |
| --- | --- | --- |
| Syntax tree | `ltn/parser/tree.py` | No — pure dataclasses |
| TPTP reader | `ltn/parser/tptp.py`, `tptp.lark` | No — Lark only |
| Grounding | `ltn/parser/grounding.py` | Yes — TF throughout |

`Grounding` builds default embeddings / MLPs, evaluates the AST into LTN
formulas, and trains with `tf.GradientTape` + `@tf.function` (including
`reuse_graph`). Refutation uses Keras optimizers and activations.

So TPTP *parsing* is already multi-backend-ready. Only `grounding.py` (and any
future training helpers) need a backend story. **Merge TPTP first** (or in
parallel with an ops extraction that does not change behavior), but do not grow
more TF-only helpers inside `grounding.py` without going through the shared
core API.

## Options considered

### A. Keep this repo TF-only; point users at LTNtorch

Simplest. Does not give one package, one TPTP front-end, or shared tests.
Rejected as the end state; fine as interim.

### B. Rewrite tensor code on `keras.ops` (Keras 3 multi-backend)

Attractive if we drop Keras 2 / TF 2.15 support: one ops vocabulary, Keras
models can run on Torch. Still leaves gaps: ragged masks, `GradientTape` vs
Torch training, and `Grounding.train`'s `tf.function` reuse. Also conflicts with
the current CI matrix that still tests TF 2.15.

### C. Thin `ltn.ops` + per-backend adapters (recommended)

Introduce a small ops module that covers the primitives LTN actually uses
(~50 names today), implemented first by TF (behavior unchanged), then by Torch.
Keep high-level Real Logic in `core` / `fuzzy_ops` against that surface. Put
model factories and training loops in backend-specific modules.

This is more code than B in the short term, but it:

- does not force dropping Keras 2 on day one,
- matches how LTNtorch already solved masking (mask-in-aggregator),
- keeps the public Real Logic API stable,
- fits TPTP: `tree`/`tptp` stay as they are; `Grounding` calls core + a thin
  `train_step` helper.

### D. Full pluggable backend plugin system

Register arbitrary backends via entry points. Overkill for two backends (YAGNI).
If a third backend appears, revisit.

**Recommendation: C**, with Torch as the second backend only. Prefer converging
masked quantification on the LTNtorch-style mask-in-aggregator API so both
backends share one quantifier path (and TF can drop ragged).

## Target architecture

```
ltn/
  ops/                  # tensor primitives (backend-selected)
    __init__.py         # re-exports active backend
    tensorflow.py
    torch.py
  core.py               # Real Logic; imports ltn.ops, not tf/torch directly
  fuzzy_ops.py          # same; aggregators accept optional mask=
  nn/                   # optional sugar; may stay in core at first
    tensorflow.py       # MLP / FromLogits / Lambda on Keras
    torch.py            # same on nn.Module
  parser/
    tree.py             # unchanged, backend-free
    tptp.py             # unchanged, backend-free
    grounding.py        # uses ltn core + ltn.train helpers; no raw tf/torch
  backend.py            # set_backend / get_backend (import-time or explicit)
```

Public usage sketch (exact names TBD):

```python
import ltn
ltn.set_backend("torch")  # or env LTN_BACKEND=torch before import

x = ltn.Variable("x", values)           # values: numpy / list / backend tensor
P = ltn.Predicate.MLP([4])              # builds a Torch or Keras MLP
```

Packaging:

- Core install should not pull both frameworks.
- Something like `ltn[tensorflow]` (default for compat) and `ltn[torch]`, or
  keep TF as the default extra until a major version flips the default.
- `ltn[parser]` stays Lark-only and works with either backend.

## Phased plan

### Phase 0 — Land TPTP on TF (prerequisite / parallel)

Finish and merge `feat/tptp-parser` as TF-only. Keep `tree.py` / `tptp.py`
free of tensor imports (already true). Avoid adding new direct `tf.*` call
sites in `grounding.py` that duplicate what `ltn.core` / `ltn.utils` already
expose (e.g. prefer shared smooth-equality helpers once ops exist).

Exit: TPTP tests green on master; parser README accurate.

### Phase 1 — Extract `ltn.ops` behind the TF backend (no user-visible change)

1. List the ops inventory (constant, convert, cast, shape, reshape, expand_dims,
   repeat, transpose, gather, gather_nd, stack, concat, where, reductions,
   elementwise math, random.uniform, boolean_mask equivalent, etc.).
2. Implement `ltn.ops.tensorflow` and route `core` / `fuzzy_ops` / `utils`
   through it.
3. Change masked quantification to the mask-in-aggregator style **on TF first**,
   with parity tests against current ragged behavior (empty domains, masks that
   only guard free vars of the result, etc. — the cases already covered in
   `test_core.py`).
4. Keep Keras model types in Phase 1; only tensor math moves.

Exit: existing test suite green; no API break; possible small numerical
differences only where ragged vs dense-mask is documented and tested.

### Phase 2 — Autograd and trainable symbols abstraction

Introduce a minimal interface used by core:

- `variable(value, trainable, constraint=...)`
- `trainable_variables(obj)` / collect parameters from Predicate/Function
- `Variable.from_constants` without requiring a TF tape: document that callers
  must be inside an active gradient context; optional backend-specific check
  (TF can still warn if a tape is passed and not recording).

Exit: Constants/Propositions creatable through ops; GradientTape checks become
optional/backend-specific rather than hard-coded in core signatures.

### Phase 3 — PyTorch ops + nn adapters

1. Implement `ltn.ops.torch`.
2. `Predicate`/`Function` accept either a callable module with a clear contract
   (`__call__(tensor | list[tensor]) -> tensor`) rather than requiring
   `tf.keras.Model` specifically.
3. Backend-specific `MLP` / `FromLogits` / `Lambda`.
4. Parametrize core tests over backends (pytest), plus a few Torch-only smoke
  tests for `nn.Module` integration.
5. Decide relationship with LTNtorch: prefer behavioral parity tests over
   copying its larger API surface; this repo's API (Expression/Term/Formula,
   Wrapper_*) stays canonical.

Exit: `LTN_BACKEND=torch uv run pytest test` covers core + fuzzy_ops; TF job
unchanged.

### Phase 4 — Backend-aware `Grounding` (TPTP)

Refactor `ltn/parser/grounding.py`:

- Default symbols via `Predicate.MLP` / `Function.MLP` / `Constant` (already
  mostly true).
- Replace `@tf.function` + `GradientTape` training with a small helper, e.g.
  `ltn.training.maximize_sat(params, sat_fn, steps, optimizer=...)`, with TF and
  Torch implementations.
- `reuse_graph` becomes TF-specific (or a generic "compile/cache step" hook).
  Document that Torch eager mode has nothing to reuse, or optionally
  `torch.compile` later (YAGNI for v1).
- Refutation loss uses backend ops for `elu`, not `tf.keras.activations.elu`.

Exit: the Socrates README example and `test_tptp*.py` pass on TF and Torch.

### Phase 5 — Packaging, CI, docs

- Optional deps and import errors that say which extra to install.
- CI matrix: TF (current Python × TF versions) + one Torch job (Linux, latest
  stable). Avoid a full cross-product at first.
- Tutorials: one short "Using PyTorch" notebook or a section in tutorial 1;
  keep existing notebooks on TF until examples are ported selectively.
- README: stop describing the project as TF-only; keep the LTNtorch link as
  historical / alternate API until/unless that project defers to this one.

## Risks and open decisions

| Topic | Risk | Lean |
| --- | --- | --- |
| Keras 2 (TF 2.15) CI | Ops extraction is fine; Keras-3-only APIs are not | Keep TF 2.15 until Phase 3 forces a call; then drop or gate |
| Masked quantifier semantics | Subtle empty-domain / axis-order bugs | Port tests before changing TF path; match LTNtorch only where tests agree |
| Default backend | Breaking change if we flip to Torch | Keep TF default until a major version |
| `set_backend` after import | Half-initialized modules | Set only before first use, or process-wide at import via env |
| LTNtorch divergence | Duplicate ecosystems | Aim for formula-level parity; do not promise API identity |
| `Grounding.reuse_graph` | Torch has no TF graph | TF-only flag; no-op or warning on Torch |
| Examples | Large rewrite | Leave examples TF until core is stable; port one (e.g. smokes_friends_cancer) as a Torch reference |

## What not to do (for now)

- Do not introduce JAX or a plugin entry-point system.
- Do not rewrite tutorials and all examples in the first PR series.
- Do not make `ltn.parser.tptp` depend on a backend.
- Do not keep two quantifier implementations (ragged TF + masked Torch) longer
  than Phase 1 — converge early.

## Suggested PR sequence (after this plan)

1. Merge TPTP (`feat/tptp-parser`) if not already on master.
2. `ref/ops-tensorflow`: extract `ltn.ops`, TF-only, plus mask-in-aggregator.
3. `feat/backend-torch-ops`: Torch ops + parametrized core tests.
4. `feat/backend-torch-nn`: MLP / FromLogits / Lambda on Torch.
5. `feat/parser-backend-train`: Grounding train/refute via shared helpers.
6. `docs/multi-backend`: README, parser README, short tutorial note.

Each PR stays reviewable; none need to ship Torch to users until 3–4 are green.

## References

- Master core: `ltn/core.py`, `ltn/fuzzy_ops.py`, `ltn/utils.py`
- TPTP branch: `origin/feat/tptp-parser` (`ltn/parser/{tree,tptp,grounding}.py`)
- External Torch port (mask-in-aggregator precedent):
  [tommasocarraro/LTNtorch](https://github.com/tommasocarraro/LTNtorch)
- Issue that motivated the parser: #26
