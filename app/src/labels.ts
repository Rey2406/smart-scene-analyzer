/**
 * Detection class label order — the single source of truth on the TypeScript side.
 *
 * Canonical source: docs/taxonomy.md, "Class list" table (ID column). Reconciled
 * against the real exported model in docs/model-card-v1.md ("Label order" row:
 * "matches exactly. Confirmed independently via `YOLO(best.pt).names`, index-for-index
 * against docs/taxonomy.md") and restated as a comment (not a second array) in
 * app/src/inference/detector.ts.
 *
 * Index == class_id == the position the detector's class-confidence channels are in.
 * Never duplicate this array inline anywhere else -- import LABELS (or labelFor) from
 * here. A second copy drifts the moment the taxonomy changes (docs/taxonomy.md:
 * "Reordering these rows silently mislabels every downstream artifact and nothing
 * raises an error").
 */
export const LABELS = [
  'bed',
  'bin',
  'chair',
  'door',
  'lamp',
  'pillow',
  'screen',
  'sofa',
  'storage',
  'table',
] as const;

export type Label = (typeof LABELS)[number];

/** Class index -> label. Throws on an out-of-range index rather than returning undefined,
 * so a decoder bug surfaces immediately instead of drawing a box with label "undefined". */
export function labelFor(classId: number): Label {
  const label = LABELS[classId];
  if (label === undefined) {
    throw new Error(`labelFor: class id ${classId} is out of range for LABELS (length ${LABELS.length})`);
  }
  return label;
}
