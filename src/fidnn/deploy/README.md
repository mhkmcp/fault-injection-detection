# fidnn.deploy

`fidnn export` writes a portable bundle with:
- one ONNX graph holding the classifier, the D2 monitor and the D1 per-tap monitors;
- classifier-only versions of that model for smaller devices;
- a browser UI that runs the ONNX graph with ONNX Runtime Web;
- a runner that needs only `onnxruntime`, `numpy` and `Pillow`.

This is a C0 addition. It is presentation and deployment only and produces **no thesis number**.

| File | Role |
|---|---|
| `graph.py` | `MonitoredClassifier`: uint8 NHWC → normalisation → model → tap descriptors → Block E → robust normaliser → D2 RBF-SVDD score → alarm per α; with `per_tap`, also D1 score per tap → alarm per tap and α |
| `formats.py` | Classifier-only exports: ONNX FP32, static INT8 QDQ, ORT format, TFLite through an isolated `uv` env |
| `tflite_convert.py` | onnx2tf → TFLite FP32/FP16/full INT8, a C header, and per-file accuracy. Runs **outside** the project env |
| `export.py` | Builds everything from fitted artifacts and gates on parity. Records accuracy, size and hash per file, the weight byte-offset table and overhead |
| `bundle/` | Copied verbatim into every bundle: `run.py`, the recipient's `README.md`, `requirements.txt`, `ui/index.html` |

Output: `artifacts/deploy/{model}_{precision}_{tap_set}_seed{n}/` and a `.zip` of it.
- Install with `uv sync --group deploy --group ui`. `uv sync --group deploy` alone uninstalls
  Gradio.
- The first TFLite run downloads TensorFlow into uv's cache, about 1 GB; `--no-tflite` skips it.
- ONNX Runtime Web is downloaded once, pinned, into `artifacts/cache/onnxruntime-web-*`.

## Contract

- **Nothing is fitted here (C2, C3).** The constants come from these sources:
  - channel stats: the `train` split;
  - saturation levels: the extract sidecar (`clean_fit`);
  - normaliser and SVDD: the `fidnn fit` bundle;
  - τ_α: the `_calibration.parquet` (`clean_cal`);
  - per-tap τ_α for D1: the (1−α) quantile of D1's per-tap scores on `clean_cal`, at the same
    fixed α (C3). The calibration parquet holds D1 only as a max over taps, so export computes
    these from the clean features; no fault data is read;
  - INT8 and TFLite calibration: a `clean_fit` subsample, as in `fidnn quantize`.
- **The SVDD is written out as kernel arithmetic:**
  `score = −(Σ dual_coef·exp(−γ‖z − sv‖²) + intercept)`. That is `−decision_function` (SPEC
  §6.1), with no covariance (C1).
- **Parity is a gate, not a hope.** Export fails unless ONNX Runtime reproduces the fidnn alarm
  decision on every `clean_test` image at every α, and the score to within `SCORE_TOL`. The
  residual is FP32 kernel rounding.
- **Every file carries its own measurement.** The manifest's `formats` gives accuracy on
  `clean_test` for each file, measured in the runtime that file targets. That is ORT for
  `.onnx`/`.ort` and LiteRT for `.tflite`.
- **Overhead travels with detection (C5).** The manifest records single-image CPU latency of the
  classifier alone and of the monitored graph, both under ORT.
- **Only `model.onnx` / `model.ort` carry the monitor.** The classifier-only formats exist for
  devices that can't afford it.
- **Calibration matches the model it runs on.** Re-quantising `model.onnx` would quantise the
  monitor's features and silently invalidate τ_α. A quantised monitor needs a fresh
  `extract → fit` on the quantised model.

## Decisions

- **Twin descriptor.** `graph.tap_features` is a twin of `taps.features.tap_features`, not a
  reuse of it. `torch.quantile` and data-dependent indexing don't export, so it uses sorted rows
  with static interpolation indices instead. `tests/test_deploy.py` holds the two to parity.
- **Entropy uses `xlogy(p, p + ε)`.** The literal `p·log(p + ε)` exports to a graph that ONNX
  Runtime evaluates as NaN at p = 0, which happens on dead stem channels.
- **Tap capture.** Tap outputs are captured by swapping each `Tap` identity for a recorder. The
  model's own `forward` is untouched.
- **INT8 ONNX skips symbolic shape inference** (`quant_pre_process(skip_symbolic_shape=True)`).
  It fails on the `Range` node of the option-A shortcut under a dynamic batch.
- **TFLite starts from a separate graph:** NCHW float input, static batch 1.
  - With a dynamic batch, onnx2tf can't convert the shortcut's `Range`.
  - With an NHWC input, it inserts a second transpose and breaks full-INT8 calibration.
  - From NCHW it produces a native NHWC TFLite input.
  - onnx2tf's built-in sample download is a pickled `.npy` that NumPy 2 refuses to load, so it
    is replaced with the clean calibration images.
- **TFLite never enters the project env.** TensorFlow pins its own protobuf and numpy, so
  conversion runs under `uv run --isolated` with pinned versions.
- **Browser fault simulation** flips bytes at the offsets in `manifest.weights`. Those are the
  positions of each FP32 weight's raw data inside `model.onnx`, recorded only when the bytes
  occur exactly once.
- **The UI's fault controls are the §4.2 grid, not a copy of it.** `manifest.faults` is generated
  from the same sources the experiments use:
  - layer buckets from `inject.targets.BUCKET_OF`;
  - FP32 bit strata from `inject.plan.STRATA[32]`, with the plain-language labels from
    `demo.app.SEVERITY`;
  - budgets 1/4/16.

  The page's `sampleFlips` mirrors `inject.plan.sample_flips`: distinct (element, bit) pairs
  across the bucket's weights. Its flip accumulation and "Surprise me" mirror `DemoSession`.
- **Weights in `model.onnx` are BN-folded.** The ONNX export folds each BatchNorm into the
  preceding convolution, so `model.onnx` has no separate γ tensors. A browser flip therefore hits
  `w·γ/√(σ²+ε)` rather than the raw conv weight. That is the right target for a deployed model,
  but not identical to a `fidnn inject` flip.
- **"Injection in weight" is a C0 addition, demo only.** `export.WEIGHT_FORMATS` lists FP32,
  FP16, FP8 (E4M3 "FN"), INT2, INT4, INT8 with their bit strata; FP32 and INT8 reuse
  `inject.plan.STRATA` unchanged, the others use the same stratum names on their own bits. The
  page converts every weight to the chosen format, writes the decoded value back as FP32 (ONNX
  Runtime still runs an FP32 graph), and flips XOR the format's code. Floats round to nearest
  even and saturate; integers are symmetric per output channel. A format change clears flips. The
  monitor's τ_α stays the FP32 calibration, so for other formats the page measures detection
  against that format's unflipped model and says the false-alarm rate may exceed α. Bundles
  exported before this existed fall back to FP32 only.
- **The m-of-n window is stateful**, so it lives in `run.py` and the UI, not in the graph.
- **Detector-design comparison (UI section 5) is recorded, not live.** If
  `artifacts/results/{model}_{precision}_{tap_set}_seed{n}_granularity.parquet` exists (from
  `fidnn granularity`), `manifest.comparison[precision]` carries its rows, the paired differences
  against the joint SVDD, the H2 verdict, per-layer localisation and feature cost per K. The page
  says these were measured offline on the fault sweep with the stated seed count, and hides the
  section when no results exist. Only the D2 monitor (and the D1 per-tap view) run live.
- **Per-layer view (D1) is a C0 addition, demo only.** The graph also returns `tap_scores`
  (N, K) and `tap_alarms` (N, K, A) from the fitted D1 detectors, under the same parity gate as
  D2. `manifest.monitor.per_layer` carries tap labels, per-tap thresholds, a display range
  (clean_cal 1 % quantile to 1.5× the 0.1 % threshold span) and clean_test FPR per tap and for
  "any tap". Each tap alarms at α on its own, so "any tap" is not an α-level detector; the UI
  says so and keeps D2 as the headline verdict. `manifest.weights[].tap` is the first tap after
  that weight's module in execution order (`_tap_after`), which is where the UI puts the ⚡.
  Bundles from a fit without D1, or exported before this, run without the per-layer view.
