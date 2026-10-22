# Round 1 — on-device patched-detector test (diagnostic only, not a fix)

**Question:** does downgrading the `RESIZE_NEAREST_NEIGHBOR` op in `detector.tflite` from
operator version 3 (`half_pixel_centers=True`) to version 1 (`half_pixel_centers=False`)
resolve `react-native-fast-tflite`'s `Invoke()` failure against `TensorFlowLiteC 2.17.0`
(the version `react-native-fast-tflite@3.0.1`'s podspec pins, confirmed in
`app/node_modules/react-native-fast-tflite/react-native-fast-tflite.podspec` and
`app/ios/Podfile.lock`)?

**Answer: no.** The patch was applied, confirmed well-formed in Python, built into the app,
and run on-device. `Invoke()` still fails, with the same generic error as before the patch.

## What was done

1. Reproduced the byte-level FlatBuffer patch (vtable-offset mutation of the OperatorCode's
   `Version` field and both `ResizeNearestNeighborOptions.HalfPixelCenters` fields) against
   `app/assets/models/detector.tflite`, writing the result to `/tmp/detector_patched.tflite`.
   The original was backed up first at `/tmp/detector_original_backup.tflite` (confirmed
   byte-identical to the pre-patch `detector.tflite` via `cmp`).
2. **Python verification, both files**, via `ai_edge_litert.Interpreter` (the same runtime
   `tests/test_export_parity.py` uses):
   - Original: `AllocateTensors`/`Invoke` succeed, output `[1,14,8400]`, mean 0.0906.
   - Patched: `AllocateTensors`/`Invoke` succeed, output `[1,14,8400]`, mean 0.0906.
   - **Both succeed.** This is new information this round surfaced: `ai_edge_litert`
     (the LiteRT-generation Python runtime) was never actually sensitive to the op-version
     difference — it runs the unpatched file fine too. The earlier "Python-verified" claim
     for the patch was true but weaker evidence than it read: it confirmed the patched file
     is well-formed, not that the patch fixes anything specific to the classic
     `TensorFlowLiteC 2.17.0` runtime the app actually ships. Flagging this explicitly
     rather than letting the earlier framing stand uncorrected.
3. Swapped the patched file into `app/assets/models/detector.tflite`, rebuilt with
   `npx expo run:ios --device DACAF190-5F03-4001-89BE-8F336DCDCB15` (build succeeded, 0
   errors), reinstalled and launched.
4. Ran the existing `DEV_DEBUG_INVOKE` diagnostic (`app/App.tsx`, already present from an
   earlier session) — `loadDetectorModel()` + `model.runSync()` on an all-zero
   `[1,3,640,640]` float32 input. Metro's network log confirmed the app fetched the
   patched file live from the packager (`response_bytes=3043766` ≈ the patched file's
   3,043,455 bytes + HTTP headers), so the on-device test ran against the actual patched
   model, not a stale cached copy.
5. Console output isn't reachable from this session (Hermes/New Architecture routes
   `console.log` over the Metro websocket, not into the iOS unified log —
   confirmed empty via `xcrun simctl spawn ... log show` with several predicates), so the
   result was captured visually instead: `DEV_DEBUG_INVOKE`'s result was temporarily
   rendered as an on-screen banner (matching the project's existing `__DEV__`-gated
   dev-instrumentation pattern, e.g. `runDevBenchmark`/`devBenchSummary` in `PhotoTab`)
   and read via a simulator screenshot.

**On-screen result, patched model:**

```
[DEV_DEBUG_INVOKE] runSync FAILED: Error:
TfliteModel.runSync(...): TFLite: Failed to run TFLite Model!
Status: error
```

Identical generic failure to the pre-patch behavior already documented in
`app/src/inference/decode.ts`'s header and `app/src/pipeline/analyzeImage.ts`'s header.

## What this means

The `RESIZE_NEAREST_NEIGHBOR` version-3/`half_pixel_centers` diagnosis was precise about
*what* differs between the graph and a strict reading of `TensorFlowLiteC 2.17.0`'s
documented forward-compatibility guarantees, but neutralizing that one difference did not
fix `Invoke()`. Either the real blocker is a different op/attribute this diagnosis didn't
reach, or the version-3 resize op is not actually what `TensorFlowLiteC 2.17.0` rejects at
runtime (the generic `Status: error` gives no op-level detail to confirm either way without
instrumenting the native pod itself, which is out of scope for this report). **This
hypothesis is refuted, not confirmed** — reported as a negative result rather than silently
dropped.

## Cleanup (verified)

- `app/assets/models/detector.tflite` restored from `/tmp/detector_original_backup.tflite`.
  SHA-256 `3f5bf68248e5c36abf4f2fb260c38f1ec27d76203a22cc61e054f510b93ad83d`, matching both
  the backup and the `sha256` field already on record in
  `app/assets/models/model_contract.json`.
- `app/App.tsx` reverted to its pre-task content (792 lines, byte-for-byte match on
  re-read) — the temporary on-screen-banner version of `DEV_DEBUG_INVOKE` and the
  temporary `DEV_DEBUG_DEPTH_DUMP` block (used separately, see `app-depth-raw.json`'s
  provenance) were both removed, not left in place.
- Rebuilt once more (`npx expo run:ios`) with the restored file and reverted `App.tsx`, and
  confirmed via screenshot that the running simulator app shows no debug banner — a clean
  state, not just a clean source tree.
