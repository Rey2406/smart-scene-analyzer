# Round 2 — the on-device `Invoke()` failure, actually root-caused

**Supersedes round1's "refuted, cause unknown" status.** The `RESIZE_NEAREST_NEIGHBOR`
op-version hypothesis stays refuted (unchanged from round 1). This round found the real
cause, and corrects a wrong intermediate conclusion reached mid-investigation — reported
here rather than quietly overwritten, per this project's own documentation ethos.

## The real cause

`detector.tflite`'s constant weight tensors are stored two different ways in the same
file. Smaller tensors use the FlatBuffer's classic inline `Buffer.Data()` vector. Larger
ones — 90 of the model's 176 CONV_2D/DEPTHWISE_CONV_2D weight and bias tensors — use a
newer convention instead: `Buffer.Offset()`/`Buffer.Size()`, which store the bytes
out-of-line, appended after the FlatBuffer root, for zero-copy mmap loading. This is a
real, legitimate TFLite feature, not a bug in the export.

`ai_edge_litert.Interpreter(model_path=...)` loads from a **file path** and resolves
`Offset()`-based buffers correctly (it can mmap the real file). `TensorFlowLiteC 2.17.0`'s
C API — which `react-native-fast-tflite` uses — exposes two different model-loading
entry points: `TfLiteModelCreateFromFile(path)` and `TfLiteModelCreate(buffer, size)`.
`HybridTfliteModule.cpp:36` calls the **buffer** variant, handing it an `ArrayBuffer`
already copied fully into memory (`HybridAssetLoader.swift` fetches/reads the asset, then
`ArrayBuffer.copy(data:)`s it). The buffer-based loader does not resolve offset-based
buffers — plausibly because that resolution was only ever wired up for the mmap/file path,
which is the feature's whole reason to exist. `AllocateTensors()` still succeeds (shape
and dtype are metadata, always present); `Invoke()` fails the instant the graph reads the
first offset-based tensor it never wired up, with the exact, specific native error:

```
Input tensor 207 lacks data
```

(captured from the iOS unified log via `mobile_get_device_logs`, filtered to the app's
process, while a repeating on-device diagnostic call was active — Hermes `console.log`
alone would have shown only the generic JS-level status string, not this.)

## Evidence chain

1. **The native error, captured directly** (not inferred): "Input tensor 207 lacks data",
   logged by `TensorFlowLiteC`'s own error reporter at the exact moment `Invoke()` fails.
2. **Tensor 207 identified**: a CONV_2D weight, `[32, 3, 3, 16]` int8, buffer index 207.
   `Buffer.DataLength()` for that buffer is 0.
3. **First (wrong) conclusion**: treated `DataLength() == 0` as "missing data" and scanned
   all 176 weight/bias tensors the same way — found 90 with the same pattern, spanning
   most of the backbone. Reported this as a corrupt/incomplete export.
4. **Reproduced independently**: re-ran `export_detector()` fresh from the same checkpoint
   (`best.pt`) to a scratch path. The new file (different SHA-256, same byte size) has the
   **exact same 90 tensor indices** empty by the `DataLength()` test — fully deterministic,
   ruling out file-transfer/bundling corruption as the cause.
5. **The correction**: checked for legitimate sparse-tensor encoding first (`Sparsity()`
   is `None` on every checked tensor — ruled out). Then checked `Buffer.Offset()`/`Size()`
   — non-zero for tensor 207 (offset 3,013,120, size 4,608 bytes, exactly the tensor's
   expected byte count). Read those 4,608 bytes directly from the file: 4,447/4,608
   non-zero, consistent with real quantized weight data, not padding. **The data is
   present. The file is not corrupt.** `DataLength()` alone is the wrong check for a file
   using this buffer convention — it only reports the inline-vector tensors.
6. **Why the existing parity test still passed**: `test_detection_parity_per_class`
   (`tests/test_export_parity.py`) uses `ai_edge_litert.Interpreter(model_path=...)`,
   which resolves offset-buffers correctly via the file path — consistent with its
   genuine, trustworthy 91.9% (34/37) match rate. That test's result stands; it was never
   affected by any of this.
7. **The actual mismatch isolated**: `TfLiteModelCreateFromFile` exists in the same
   `TensorFlowLiteC.framework/Headers/c_api.h` this app links — `react-native-fast-tflite`
   calls the other one (`TfLiteModelCreate`, buffer-based).

## What this means for round 1

Round 1's Python-verification paradox ("the original, unpatched file also loads and runs
fine under `ai_edge_litert`") is now fully explained rather than left as an open
oddity: `ai_edge_litert` was never affected by the offset-buffer issue in the first place
(file-path loading resolves it correctly), so of course it ran fine on both the original
and the patched file — it was never testing the thing that actually fails on-device. The
`RESIZE_NEAREST_NEIGHBOR` hypothesis remains refuted on its own terms (unrelated op,
unrelated mechanism); this round found a completely different, independent cause.

It also explains an earlier, separate anomaly from this project's own history: a prior
attempt to FlatBuffer-round-trip this file via `ModelT.Pack()` (not part of this round's
work) produced a corrupted ~434 KB output instead of ~3 MB. That tool's unpack/repack path
almost certainly has the same blind spot — reading only inline buffers, silently dropping
every offset-based one during repacking. Consistent with, not independent evidence for,
this round's finding.

## Fix attempt 1: native loader patch — tried, refuted

Patched `HybridTfliteModule.cpp`'s `createModel()` (iOS) to write the already-fetched
model bytes to a temp file and call `TfLiteModelCreateFromFile` instead of
`TfLiteModelCreate`, via `patch-package`. Rebuilt and retested on the Simulator, twice
(once with an immediate `unlink()` of the temp file after load, once without, to rule out
a lazy-mmap race as a separate variable).

**Both attempts failed identically**: same native error, same tensor —
`Input tensor 207 lacks data` — captured fresh via `mobile_get_device_logs` after each
rebuild. Model creation and `AllocateTensors()` still succeed either way (no "Failed to
create TFLite model" error), so the file loaded; `Invoke()` still cannot read the
offset-based tensor.

**This refutes the file-vs-buffer-API hypothesis.** The dividing line is not which
`TfLiteModelCreate*` entry point is used — `TensorFlowLiteC 2.17.0`'s underlying TFLite
core apparently does not resolve the `Buffer.Offset()`/`Size()` convention through
*either* loading path. `ai_edge_litert.Interpreter(model_path=...)` succeeding is more
likely explained by it being a newer, separately-versioned runtime (the "LiteRT"
successor, a different package from classic `tensorflow-lite`) with core parsing logic
that resolves this convention — a version/implementation gap between the two runtimes,
not an API-choice gap within one of them.

Patch fully reverted: `node_modules` source restored, `patch-package` uninstalled,
`patches/` removed, `App.tsx`'s temporary diagnostic block removed, rebuilt once more and
confirmed clean.

## Fix paths remaining, not yet attempted

Both avoid touching the export pipeline — the artifact is correct as exported.

1. **Force the export to skip the offset-buffer convention**, if `ultralytics`'s
   `format="litert", quantize=8` path (or the underlying `ai-edge-litert` converter it
   calls) exposes a flag for it. Not yet investigated — needs checking against the
   installed package's actual source/docs, not memory, per this project's own
   verification rule for fast-moving toolchains. Likely the more promising path now that
   the native-loader patch is refuted.
2. **Find a newer `TensorFlowLiteC` pod version** (or an alternative TFLite-C-API-based
   runtime) that does resolve this buffer convention, and get `react-native-fast-tflite`
   onto it — a heavier change, since the library pins `2.17.0` with "no flexibility" per
   its podspec, and there is no evidence yet that a newer compatible version exists.

Attempt 1 (above) is now closed as refuted. Neither of these two remaining paths has been
attempted.
