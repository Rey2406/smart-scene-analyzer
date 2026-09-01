"""The run-tracking seam (contract C13). **MLflow hosting is an OPEN decision.**

**Single responsibility:** let training record a reproducible run without any module
knowing where that record is stored.

`docs/requirements.md` lists MLflow hosting as open, blocking N19. This module is the
reason that can stay open without blocking anything else: nothing under `src/` calls
MLflow directly. Callers depend on the `RunTracker` protocol, and the backend is
constructed once, from `MLFLOW_TRACKING_URI`, at the top of an entry point.

The three branches and what each costs (architecture §10.3):

| Branch | Consequence |
|---|---|
| Local file store | Zero setup; runs are not shareable; N19 satisfied for one person |
| Self-hosted server | A service to run — and one that must never appear on the inference path |
| Managed | An account and a bill outside the 20-credit ledger |

**No module names a backend or a default URI.** A default pointing at a network is how an
open decision gets made by accident.

`NullTracker` is the offline double: it satisfies the protocol, records nothing, and lets
the whole training path be exercised in a unit test with no server anywhere.

Planned public surface:

    class RunTracker(Protocol):
        def start_run(self, name: str) -> str: ...
        def log_params(self, params: Mapping[str, object]) -> None: ...
        def log_metrics(self, metrics: Mapping[str, float], step: int | None) -> None: ...
        def log_artifact(self, path: Path) -> None: ...
        def register_model(self, path: Path, name: str) -> str:
            \"\"\"-> the registry version string that ends up in model_contract.json (F7).\"\"\"
        def end_run(self) -> None: ...

    class NullTracker:   # the test double. Records nothing, needs nothing
    class MlflowTracker: # constructed from the environment; hosting still open
"""
