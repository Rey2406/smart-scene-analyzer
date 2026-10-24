import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  ActivityIndicator,
  AppState,
  Image as RNImage,
  Linking,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import type { LayoutChangeEvent } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Device from 'expo-device';
import * as ImagePicker from 'expo-image-picker';
import { createVideoPlayer, type VideoPlayer } from 'expo-video';

import { analyzeImageUri, type AnalyzeImageResult } from './src/pipeline/analyzeImage';
import { DetectionOverlay, type DetectionOverlayState } from './src/components/DetectionOverlay';
import { startLiveLoop, type LiveLoopHandle } from './src/live/liveLoop';
import type { StillCaptureSource } from './src/capture/types';
import { CameraCaptureSource } from './src/capture/cameraSource';
import { ClipCaptureSource } from './src/capture/clipSource';
import { BundledPhotosCaptureSource } from './src/capture/bundledPhotosSource';
import { recordPhase, resetSamples, summarizeAll } from './src/dev/phaseTimers';

/**
 * The real app, replacing every throwaway diagnostic screen Lesson 05 built up through
 * step 8 (git history has them). Two tabs:
 *   - Photo: pick (or fall back to a bundled) image, run it once.
 *   - Live: throttled stills from a camera, a recorded clip, or the same bundled
 *     fallback, fed into the SAME pipeline function Photo mode calls
 *     (src/pipeline/analyzeImage.ts's analyzeImageUri) -- see live/liveLoop.ts and
 *     capture/*.ts for how.
 *
 * JUDGMENT CALL: this app has exactly two top-level screens and no nested routes, so
 * tab state is a plain useState switch here rather than expo-router (app/AGENTS.md's
 * usual recommendation for navigation). Pulling in a router is a new dependency this
 * project's "ask before installing" rule would apply to, for a problem (two sibling
 * screens) it does not need solved by one. Both tabs stay mounted -- see LiveTab's own
 * `active` prop below -- so switching away and back preserves state (a picked clip, a
 * running loop's counters) exactly the way a router's screen-kept-alive behavior would.
 */

const LIVE_INTERVAL_MS = 1000;

type AppTab = 'photo' | 'live';

export default function App() {
  const [tab, setTab] = useState<AppTab>('photo');

  return (
    <View style={styles.app}>
      <StatusBar style="auto" />
      <View style={styles.tabBar}>
        <TabButton label="Photo" active={tab === 'photo'} onPress={() => setTab('photo')} />
        <TabButton label="Live" active={tab === 'live'} onPress={() => setTab('live')} />
      </View>

      <View style={[styles.tabContent, tab !== 'photo' && styles.hidden]}>
        <PhotoTab />
      </View>
      <View style={[styles.tabContent, tab !== 'live' && styles.hidden]}>
        <LiveTab active={tab === 'live'} />
      </View>
    </View>
  );
}

function TabButton({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    // ACCESSIBILITY NOTE: accessibilityRole="tab" was tried first (the semantically
    // correct role, and what a `tablist` parent is for) but was confirmed, live on-device
    // via mobile-mcp's element list, to make this element vanish from the accessibility
    // tree entirely on this RN/iOS combination -- not a cosmetic miss, a real regression
    // for a VoiceOver user too, since both read the same underlying tree. "button" is
    // less precise but actually reachable; `accessibilityState.selected` still conveys
    // which one is active.
    <TouchableOpacity
      style={[styles.tabButton, active && styles.tabButtonActive]}
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ selected: active }}
    >
      <Text style={[styles.tabButtonText, active && styles.tabButtonTextActive]}>{label}</Text>
    </TouchableOpacity>
  );
}

// -----------------------------------------------------------------------------
// Photo tab -- pick one image, run the pipeline once.
// -----------------------------------------------------------------------------

// DEV-ONLY: Lesson 05 step 11's warm-run count. __DEV__-gated below, never reached in a
// release build.
const DEV_BENCH_WARM_RUNS = 25;

function PhotoTab() {
  const bundledSourceRef = useRef(new BundledPhotosCaptureSource());

  const [uri, setUri] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeImageResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewLayout, setViewLayout] = useState<{ width: number; height: number } | null>(null);
  const [devBenchStatus, setDevBenchStatus] = useState<'idle' | 'running' | 'done'>('idle');
  const [devBenchSummary, setDevBenchSummary] = useState<string | null>(null);

  const runOn = useCallback((pickedUri: string) => {
    setUri(pickedUri);
    setResult(null);
    setError(null);
    setViewLayout(null);
    analyzeImageUri(pickedUri)
      .then(setResult)
      .catch((e) => setError(String(e?.stack ?? e)));
  }, []);

  const pickPhoto = useCallback(async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError('Photo library permission was not granted.');
      return;
    }
    const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 1 });
    if (picked.canceled || picked.assets.length === 0) return;
    runOn(picked.assets[0].uri);
  }, [runOn]);

  const useBundledPhoto = useCallback(async () => {
    const bundledUri = await bundledSourceRef.current.captureStill();
    runOn(bundledUri);
  }, [runOn]);

  /**
   * DEV-ONLY: Lesson 05 step 11's Round 2 benchmark. Not reachable in a release build
   * -- the whole button this calls is wrapped in `{__DEV__ && ...}` below, and this
   * function's own body short-circuits too in case anything ever calls it directly.
   * Runs `DEV_BENCH_WARM_RUNS` warm passes of analyzeImageUri on the bundled test image,
   * discarding one throwaway pass first (first-call JIT/model-cache effects would
   * otherwise land in the warm distribution), and times "render" per run as the gap
   * between committing a new `result` and the next PAINTED frame (double
   * requestAnimationFrame -- the standard RN technique for "has this frame actually
   * drawn," not just "has React committed").
   */
  const runDevBenchmark = useCallback(async () => {
    if (!__DEV__) return;
    setDevBenchStatus('running');
    setDevBenchSummary(null);
    resetSamples();

    const bundledUri = await bundledSourceRef.current.captureStill();

    // Throwaway warm-up: loads/caches the depth model (loadDepthModel's own cache) and
    // exercises every code path once before any sample counts, exactly like deciding to
    // measure "warm" runs in the first place.
    await analyzeImageUri(bundledUri);
    resetSamples();

    for (let i = 0; i < DEV_BENCH_WARM_RUNS; i++) {
      const runResult = await analyzeImageUri(bundledUri);
      await new Promise<void>((resolve) => {
        const renderStarted = Date.now();
        setResult(runResult);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            recordPhase('render', Date.now() - renderStarted);
            resolve();
          });
        });
      });
    }

    const summary = summarizeAll();
    console.log('[dev-bench] phase summary (ms):', JSON.stringify(summary, null, 2));
    setDevBenchSummary(JSON.stringify(summary, null, 2));
    setDevBenchStatus('done');
  }, []);

  const onImageWrapLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setViewLayout({ width, height });
  }, []);

  const overlayState: DetectionOverlayState = error
    ? { status: 'error', message: error }
    : result
      ? result.overlayState
      : { status: 'loading' };

  return (
    <View style={styles.screen}>
      <View style={styles.row}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={pickPhoto}
          accessibilityRole="button"
          accessibilityLabel="Pick a photo"
        >
          <Text style={styles.actionButtonText}>Pick a photo</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionButton, styles.actionButtonSecondary]}
          onPress={useBundledPhoto}
          accessibilityRole="button"
          accessibilityLabel="Use bundled test photo"
        >
          <Text style={styles.actionButtonText}>Use bundled test photo</Text>
        </TouchableOpacity>
      </View>

      {__DEV__ && (
        <View style={styles.row}>
          <TouchableOpacity
            style={[styles.actionButton, devBenchStatus === 'running' && styles.actionButtonDisabled]}
            disabled={devBenchStatus === 'running'}
            onPress={runDevBenchmark}
            accessibilityRole="button"
            accessibilityLabel={devBenchStatus === 'running' ? 'Running benchmark' : 'Run latency benchmark (DEV)'}
            accessibilityState={{ disabled: devBenchStatus === 'running' }}
          >
            <Text style={styles.actionButtonText}>
              {devBenchStatus === 'running' ? 'Running benchmark…' : 'Run latency benchmark (DEV)'}
            </Text>
          </TouchableOpacity>
        </View>
      )}
      {__DEV__ && devBenchSummary && (
        <Text style={[styles.hint, { fontFamily: 'Courier', fontSize: 10 }]} selectable>
          {devBenchSummary}
        </Text>
      )}

      {!uri && <Text style={styles.hint}>Pick a photo, or use the bundled one, to run detection + depth.</Text>}

      {uri && result && (
        <View
          style={[styles.imageWrap, { aspectRatio: result.sourceSize.width / result.sourceSize.height }]}
          onLayout={onImageWrapLayout}
        >
          <RNImage source={{ uri }} style={styles.image} resizeMode="contain" />
          {viewLayout && (
            <DetectionOverlay
              state={overlayState}
              letterbox={result.detectorLetterbox}
              sourceImageSize={result.sourceSize}
              screenViewSize={viewLayout}
            />
          )}
        </View>
      )}
      {uri && !result && !error && (
        <View style={styles.centeredFill}>
          <ActivityIndicator color="#5b6b75" />
          <Text style={styles.hint}>Scanning the scene{'…'}</Text>
        </View>
      )}
      {uri && error && <Text style={[styles.hint, styles.errorText]}>ERROR: {error}</Text>}
    </View>
  );
}

// -----------------------------------------------------------------------------
// Live tab -- camera, clip-replay, or bundled-photos capture source, throttled
// through live/liveLoop.ts into the SAME analyzeImageUri the Photo tab calls.
// -----------------------------------------------------------------------------

type SourceKind = 'camera' | 'clip' | 'bundled';

/** The three states Round 2 requires, each with its own visible text --
 * "no camera" is NOT a flavor of "denied"; they are different facts about the device. */
type CameraAvailability =
  | { kind: 'checking' }
  | { kind: 'no-camera' }
  | { kind: 'not-asked' }
  | { kind: 'denied'; canAskAgain: boolean }
  | { kind: 'granted' };

function LiveTab({ active }: { active: boolean }) {
  const [sourceKind, setSourceKind] = useState<SourceKind>('bundled');

  // ---- camera availability (Round 2's three states) ----
  const [hasCamera, setHasCamera] = useState<boolean | null>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView | null>(null);

  useEffect(() => {
    // CameraView.isAvailableAsync() is unusable here: expo-camera's iOS native module
    // never implements it (confirmed in node_modules/expo-camera/ios/CameraViewModule.swift
    // -- no such method exists), so the call always throws UnavailabilityError on iOS,
    // Simulator or real device alike. Device.isDevice is the Expo-documented way to tell
    // them apart instead -- false only in the Simulator, true on every real handset.
    setHasCamera(Device.isDevice);
  }, []);

  const cameraAvailability: CameraAvailability = useMemo(() => {
    if (hasCamera === null) return { kind: 'checking' };
    if (!hasCamera) return { kind: 'no-camera' }; // the iOS Simulator case -- NOT "denied"
    if (!permission) return { kind: 'checking' };
    if (permission.status === 'granted') return { kind: 'granted' };
    if (permission.status === 'undetermined') return { kind: 'not-asked' };
    return { kind: 'denied', canAskAgain: permission.canAskAgain };
  }, [hasCamera, permission]);

  // ---- clip-replay source ----
  const [clipPlayer, setClipPlayer] = useState<VideoPlayer | null>(null);
  const [clipStatus, setClipStatus] = useState<'none' | 'loading' | 'ready' | 'error'>('none');
  const [clipError, setClipError] = useState<string | null>(null);
  const clipSourceRef = useRef<{ player: VideoPlayer; source: ClipCaptureSource } | null>(null);

  useEffect(() => {
    if (!clipPlayer) return;
    const sub = clipPlayer.addListener('statusChange', (payload) => {
      if (payload.status === 'readyToPlay') setClipStatus('ready');
      else if (payload.status === 'error') {
        setClipStatus('error');
        setClipError('The picked video failed to load.');
      }
    });
    return () => sub.remove();
  }, [clipPlayer]);

  const pickClip = useCallback(async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      setClipStatus('error');
      setClipError('Photo library permission was not granted.');
      return;
    }
    const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['videos'] });
    if (picked.canceled || picked.assets.length === 0) return;

    setClipStatus('loading');
    setClipError(null);
    const player = createVideoPlayer(picked.assets[0].uri);
    setClipPlayer(player);
  }, []);

  // ---- bundled-photos fallback (also the test-double role; see bundledPhotosSource.ts) ----
  const bundledSourceRef = useRef(new BundledPhotosCaptureSource());

  // ---- app foreground/background (Round 3: stop capturing when backgrounded) ----
  const [appState, setAppState] = useState(AppState.currentState);
  useEffect(() => {
    const sub = AppState.addEventListener('change', setAppState);
    return () => sub.remove();
  }, []);

  // ---- the chosen capture source, or null while its prerequisite isn't ready ----
  const captureSource: StillCaptureSource | null = useMemo(() => {
    if (sourceKind === 'bundled') return bundledSourceRef.current;
    if (sourceKind === 'camera') {
      if (cameraAvailability.kind !== 'granted') return null;
      return new CameraCaptureSource(() => cameraRef.current);
    }
    if (sourceKind === 'clip') {
      if (!clipPlayer || clipStatus !== 'ready') return null;
      if (clipSourceRef.current?.player !== clipPlayer) {
        clipSourceRef.current = {
          player: clipPlayer,
          source: new ClipCaptureSource(clipPlayer, LIVE_INTERVAL_MS / 1000),
        };
      }
      return clipSourceRef.current.source;
    }
    return null;
  }, [sourceKind, cameraAvailability.kind, clipPlayer, clipStatus]);

  // ---- the throttle loop itself ----
  const [running, setRunning] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [skippedCount, setSkippedCount] = useState(0);
  const [lastFrame, setLastFrame] = useState<{ uri: string; result: AnalyzeImageResult } | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const [viewLayout, setViewLayout] = useState<{ width: number; height: number } | null>(null);
  const loopRef = useRef<LiveLoopHandle | null>(null);

  const appIsActive = appState === 'active';
  const shouldRun = running && active && appIsActive && captureSource !== null;

  useEffect(() => {
    if (!shouldRun || !captureSource) {
      loopRef.current?.stop();
      loopRef.current = null;
      return;
    }
    if (loopRef.current) return; // already running against the current capture source

    const source = captureSource;
    const handle = startLiveLoop<AnalyzeImageResult>({
      intervalMs: LIVE_INTERVAL_MS,
      capture: () => source.captureStill(),
      analyze: analyzeImageUri,
      onResult: (result, uri) => {
        setLastFrame({ uri, result });
        setLastError(null);
      },
      onError: (err) => setLastError(String((err as { stack?: string })?.stack ?? err)),
      onCountsChanged: (counts) => {
        setFrameCount(counts.frameCount);
        setSkippedCount(counts.skippedCount);
      },
    });
    loopRef.current = handle;

    return () => {
      handle.stop();
      if (loopRef.current === handle) loopRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- captureSource identity is the real dependency
  }, [shouldRun, captureSource]);

  const onImageWrapLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setViewLayout({ width, height });
  }, []);

  const canStart = captureSource !== null;

  return (
    <View style={styles.screen}>
      <View style={styles.row}>
        <SourceButton label="Camera" active={sourceKind === 'camera'} onPress={() => setSourceKind('camera')} />
        <SourceButton label="Clip" active={sourceKind === 'clip'} onPress={() => setSourceKind('clip')} />
        <SourceButton label="Bundled" active={sourceKind === 'bundled'} onPress={() => setSourceKind('bundled')} />
      </View>

      {sourceKind === 'camera' && (
        <CameraSourcePanel
          availability={cameraAvailability}
          cameraRef={cameraRef}
          onRequestPermission={() => requestPermission()}
        />
      )}
      {sourceKind === 'clip' && (
        <ClipSourcePanel status={clipStatus} error={clipError} onPickClip={pickClip} />
      )}
      {sourceKind === 'bundled' && (
        <Text style={styles.hint}>
          Cycles through the bundled test photos in app/assets/test-images/ -- the fallback with no
          camera and no clip picked, and the double this screen shares in spirit with liveLoop.test.ts.
        </Text>
      )}

      <View style={styles.row}>
        <TouchableOpacity
          style={[styles.actionButton, (!canStart || running) && styles.actionButtonDisabled]}
          disabled={!canStart || running}
          onPress={() => setRunning(true)}
          accessibilityRole="button"
          accessibilityLabel="Start live"
          accessibilityState={{ disabled: !canStart || running }}
        >
          <Text style={styles.actionButtonText}>Start live</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionButton, styles.actionButtonSecondary, !running && styles.actionButtonDisabled]}
          disabled={!running}
          onPress={() => setRunning(false)}
          accessibilityRole="button"
          accessibilityLabel="Stop"
          accessibilityState={{ disabled: !running }}
        >
          <Text style={styles.actionButtonText}>Stop</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.counters} selectable>
        {`frames=${frameCount}  skipped=${skippedCount}` +
          (!active ? '  (tab hidden -- paused)' : !appIsActive ? '  (app backgrounded -- paused)' : running ? '  (running)' : '  (stopped)')}
      </Text>

      {lastFrame && (
        <View
          style={[
            styles.imageWrap,
            { aspectRatio: lastFrame.result.sourceSize.width / lastFrame.result.sourceSize.height },
          ]}
          onLayout={onImageWrapLayout}
        >
          {/* The STILL that produced the overlay below -- never the live preview, per
              resources/prompts/06-live-mode.md: the preview keeps moving while
              inference runs, so a box computed on frame N would land on frame N+k. */}
          <RNImage source={{ uri: lastFrame.uri }} style={styles.image} resizeMode="contain" />
          {viewLayout && (
            <DetectionOverlay
              state={lastFrame.result.overlayState}
              letterbox={lastFrame.result.detectorLetterbox}
              sourceImageSize={lastFrame.result.sourceSize}
              screenViewSize={viewLayout}
            />
          )}
        </View>
      )}
      {!lastFrame && running && <Text style={styles.hint}>Waiting for the first frame{'…'}</Text>}
      {!lastFrame && !running && <Text style={styles.hint}>Press "Start live" to begin.</Text>}
      {lastError && <Text style={[styles.hint, styles.errorText]}>ERROR: {lastError}</Text>}
    </View>
  );
}

function SourceButton({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    // See TabButton's ACCESSIBILITY NOTE above -- same "tab" role, same confirmed
    // vanishing-from-the-tree bug, same fix.
    <TouchableOpacity
      style={[styles.sourceButton, active && styles.sourceButtonActive]}
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ selected: active }}
    >
      <Text style={[styles.sourceButtonText, active && styles.sourceButtonTextActive]}>{label}</Text>
    </TouchableOpacity>
  );
}

function CameraSourcePanel({
  availability,
  cameraRef,
  onRequestPermission,
}: {
  availability: CameraAvailability;
  cameraRef: React.RefObject<CameraView | null>;
  onRequestPermission: () => void;
}) {
  if (availability.kind === 'checking') {
    return (
      <StatePanel>
        <ActivityIndicator color="#5b6b75" />
        <Text style={styles.hint}>Checking for a camera{'…'}</Text>
      </StatePanel>
    );
  }

  // Distinct from "denied" on purpose: the iOS Simulator has no camera hardware at
  // all, and showing a permission error here would send a user to Settings to fix
  // something that physically cannot be fixed there.
  if (availability.kind === 'no-camera') {
    return (
      <StatePanel>
        <Text style={styles.stateIcon}>📵</Text>
        <Text style={styles.stateTitle}>No camera on this device</Text>
        <Text style={styles.hint}>
          This is the iOS Simulator -- there is no camera to ask permission for. Use the Clip or
          Bundled source instead.
        </Text>
      </StatePanel>
    );
  }

  if (availability.kind === 'not-asked') {
    return (
      <StatePanel>
        <Text style={styles.stateIcon}>❔</Text>
        <Text style={styles.stateTitle}>Camera access has not been requested yet</Text>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={onRequestPermission}
          accessibilityRole="button"
          accessibilityLabel="Allow camera access"
        >
          <Text style={styles.actionButtonText}>Allow camera access</Text>
        </TouchableOpacity>
      </StatePanel>
    );
  }

  if (availability.kind === 'denied') {
    return (
      <StatePanel>
        <Text style={styles.stateIcon}>🚫</Text>
        <Text style={styles.stateTitle}>Camera access was denied</Text>
        {availability.canAskAgain && (
          <TouchableOpacity
            style={styles.actionButton}
            onPress={onRequestPermission}
            accessibilityRole="button"
            accessibilityLabel="Try again"
          >
            <Text style={styles.actionButtonText}>Try again</Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity
          style={[styles.actionButton, styles.actionButtonSecondary]}
          onPress={() => Linking.openSettings()}
          accessibilityRole="button"
          accessibilityLabel="Open Settings"
        >
          <Text style={styles.actionButtonText}>Open Settings</Text>
        </TouchableOpacity>
      </StatePanel>
    );
  }

  // granted -- the actual live preview. Deliberately small: the overlay never draws
  // here (see the still-image box below in LiveTab), so this stays a secondary view
  // that makes "overlay is on the still, not the preview" visible side by side.
  return (
    <View style={styles.cameraPreviewWrap}>
      {/* collapsable={false} + CameraView as its ONLY child: under Fabric, a sibling
          reordering anywhere in this subtree's ancestor chain (e.g. LiveTab swapping
          which source panel is mounted when sourceKind changes) commits as a native
          Remove + Insert of CameraView, not a prop update -- removeFromSuperview() tears
          the AVCaptureSession down (CameraView.swift's stopSession()), and the
          re-insertion's didMoveToWindow() is a no-op because startSessionIfNeeded()'s
          didStartSession guard is already true, so the session never rebuilds. Confirmed
          on-device: onCameraReady fires and resumePreview() resolves fine, yet the
          preview stays black -- consistent with the session having been torn down by a
          mutation JS never saw, not a permission/hardware/event-delivery problem. Root
          cause and the collapsable={false}-sole-child fix are both documented in
          https://github.com/expo/expo/issues/50015. No siblings inside this wrapper --
          any conditional content goes outside it, never between it and CameraView. */}
      <View collapsable={false} style={styles.cameraPreview}>
        {/* animateShutter={false}: unrelated to the Fabric bug above -- expo-camera's
            shutter flash briefly drops previewLayer.opacity on every takePictureAsync()
            call, which Live mode does roughly every LIVE_INTERVAL_MS. Pointless and
            visually distracting there, so disabled outright; harmless either way. */}
        <CameraView ref={cameraRef} style={styles.cameraPreviewInner} facing="back" animateShutter={false} />
      </View>
      <Text style={styles.hint}>live preview -- boxes are drawn on the captured still below, not here</Text>
    </View>
  );
}

function ClipSourcePanel({
  status,
  error,
  onPickClip,
}: {
  status: 'none' | 'loading' | 'ready' | 'error';
  error: string | null;
  onPickClip: () => void;
}) {
  return (
    <StatePanel>
      <TouchableOpacity
        style={styles.actionButton}
        onPress={onPickClip}
        accessibilityRole="button"
        accessibilityLabel="Pick a video"
      >
        <Text style={styles.actionButtonText}>Pick a video</Text>
      </TouchableOpacity>
      {status === 'loading' && <Text style={styles.hint}>Loading clip{'…'}</Text>}
      {status === 'ready' && <Text style={styles.hint}>Clip ready -- press "Start live" below.</Text>}
      {status === 'error' && <Text style={[styles.hint, styles.errorText]}>{error}</Text>}
      {status === 'none' && <Text style={styles.hint}>No clip picked yet.</Text>}
    </StatePanel>
  );
}

function StatePanel({ children }: { children: ReactNode }) {
  return <View style={styles.statePanel}>{children}</View>;
}

const styles = StyleSheet.create({
  app: {
    flex: 1,
    backgroundColor: '#fff',
    paddingTop: 60,
  },
  tabBar: {
    flexDirection: 'row',
    paddingHorizontal: 12,
    gap: 8,
    marginBottom: 8,
  },
  tabButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#eee',
    alignItems: 'center',
  },
  tabButtonActive: {
    backgroundColor: '#2b6b6a',
  },
  tabButtonText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#333',
  },
  tabButtonTextActive: {
    color: '#fff',
  },
  tabContent: {
    flex: 1,
  },
  hidden: {
    display: 'none',
  },
  screen: {
    flex: 1,
    paddingHorizontal: 12,
  },
  row: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 10,
  },
  actionButton: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#2b6b6a',
  },
  actionButtonSecondary: {
    backgroundColor: '#5b6b75',
  },
  actionButtonDisabled: {
    opacity: 0.4,
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
  },
  sourceButton: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
    backgroundColor: '#eee',
  },
  sourceButtonActive: {
    backgroundColor: '#2b6b6a',
  },
  sourceButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#333',
  },
  sourceButtonTextActive: {
    color: '#fff',
  },
  hint: {
    fontSize: 12,
    color: '#5b6b75',
    marginBottom: 8,
  },
  errorText: {
    color: '#8a2b2b',
  },
  counters: {
    fontSize: 11,
    fontFamily: 'Courier',
    marginBottom: 8,
  },
  imageWrap: {
    width: '100%',
    backgroundColor: '#ddd',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  centeredFill: {
    alignItems: 'center',
    paddingVertical: 24,
  },
  statePanel: {
    alignItems: 'center',
    paddingVertical: 16,
    marginBottom: 8,
    backgroundColor: '#f5f7f8',
    borderRadius: 10,
    gap: 6,
  },
  stateIcon: {
    fontSize: 24,
  },
  stateTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#2b2b2b',
  },
  cameraPreviewWrap: {
    marginBottom: 8,
  },
  cameraPreview: {
    width: '100%',
    height: 160,
    borderRadius: 10,
    backgroundColor: '#000',
    marginBottom: 4,
    overflow: 'hidden',
  },
  cameraPreviewInner: {
    width: '100%',
    height: '100%',
  },
});
