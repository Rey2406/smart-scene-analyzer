"""A named dataset version to training inputs.

**Single responsibility:** resolve an immutable dataset version into local splits, and
refuse to be vague about which version that was.

- **A version, never "the latest."** The version number is an argument, it is recorded in
  the run (N19), and it appears in `model_contract.json` so that an artifact can be traced
  to the data that produced it.
- **`data/` is not modified here.** Dataset changes go through Roboflow, where they are
  versioned and reviewable. This module reads an export; it does not curate one.
- The class list this module reads is an **open decision** (the taxonomy, Lesson 02).
  Nothing here hardcodes one or computes on the class count.

**Integration-marked.** It reads files and may download.

Planned public surface:

    def resolve(version: int, cache_dir: Path) -> DatasetSplits
        \"\"\"-> train/val/test paths plus the ordered label list, tagged with `version`.\"\"\"

    def summarize(splits: DatasetSplits) -> DatasetSummary
        \"\"\"-> per-split image and instance counts, per class. The input to the DATA /
        TAXONOMY / MODEL triage in `.claude/skills/error-triage`.\"\"\"
"""
