"""Batch scoring against the frozen champion pipeline.

The artifact saved in Step 4 is a complete ``Pipeline([("preprocessor", ...),
("model", ...)])``, so this module contains **no preprocessing logic of its own**. It
loads that pipeline, reconciles the caller's columns with the schema the pipeline was
fitted on, and applies the decision threshold that was frozen on validation.

Every artifact is validated before it can score anything:

    metadata fields -> promotion status -> threshold -> classifier classes -> schema

The checks share a theme: each guards a failure that would otherwise produce
**well-formed but wrong output** rather than a crash. A rejected model still returns
probabilities; a NaN threshold still returns classes (all zero); a guessed
probability column still returns numbers in [0, 1]. None of that would be caught
downstream, so it is caught here, at load.

Three things it deliberately does not do:

*Refit anything.* There is no `fit`, no `fit_transform`, and no code path that reaches
one. Preprocessing at inference is transform-only, using statistics learned from the
training split — recomputing an imputer median or a scaler mean from the batch being
scored would make a prediction depend on which other applicants happened to be in the
same file.

*Fall back to 0.5.* `predict` applies the threshold from metadata. sklearn's own
`predict` hard-codes 0.5, which is not the cut-off this project selected, so calling it
on the champion would silently score at the wrong operating point.

*Serve an unpromoted champion.* A model is unpromoted precisely because Step 4 measured
it and found it wanting. Scoring it needs an explicit `allow_unpromoted=True`, which
exists for debugging and is off by default.

Schema policy (training is the authority, never the batch):

    missing column   InferenceSchemaError — the two sides disagree about what the
                     features are; imputing it would hide that behind a plausible number
    extra column     dropped, logged
    wrong order      reordered
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.features.data_preprocessing import align_to_training_schema, get_input_feature_names
from src.models.serialization import CHAMPION_PIPELINE_FILENAME, METADATA_FILENAME
from src.utils.exceptions import InferenceSchemaError, ModelArtifactError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PREPROCESSOR_STEP = "preprocessor"
MODEL_STEP = "model"

# Production-critical metadata. None of these has a safe default, so all are required:
# a missing 'promoted' must never read as approved, and a missing 'optimal_threshold'
# must never fall back to 0.5. Names match the keys scripts/train.py actually writes.
REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "model_name",               # which champion this is
    "optimal_threshold",        # frozen on validation in Step 4
    "promoted",                 # passed its quality gates
    "input_feature_columns",    # the raw feature contract
)

# The target is binary default / non-default. Anything else is a different problem, and
# scoring it through this Predictor would misinterpret the probability columns.
EXPECTED_CLASSES = frozenset({0, 1})
POSITIVE_CLASS = 1

ID_OUTPUT_COLUMN = "SK_ID_CURR"
PROBABILITY_COLUMN = "DEFAULT_PROBABILITY"
CLASS_COLUMN = "PREDICTED_CLASS"
OUTPUT_COLUMNS = (ID_OUTPUT_COLUMN, PROBABILITY_COLUMN, CLASS_COLUMN)


class Predictor:
    """Scores raw applicant frames with the frozen champion pipeline and threshold.

    Construct via `Predictor.load()` in production; the initialiser takes an in-memory
    pipeline and metadata so tests can build one from a small synthetic fit.

    Attributes:
        pipeline: The fitted preprocessing+model Pipeline.
        metadata: The deployment metadata sidecar.
    """

    def __init__(self, pipeline, metadata: dict, *, allow_unpromoted: bool = False):
        """Validate the artifact end to end, then hold it ready for scoring.

        Every check runs at construction rather than at first predict, so a bad artifact
        fails on load instead of halfway through a batch — or, worse, silently produces
        plausible numbers for the whole batch.

        Args:
            pipeline: Fitted `Pipeline` with a preprocessing step and a model step.
            metadata: Deployment metadata sidecar.
            allow_unpromoted: Debugging escape hatch. Scores a champion that failed its
                Step 4 quality gates. Defaults to False and must stay that way: a model
                is unpromoted precisely because it was measured and found wanting, and
                the only thing worse than no prediction is a confidently wrong one.

        Raises:
            ModelArtifactError: for any artifact that cannot be served safely.
        """
        self._validate_pipeline_shape(pipeline)
        self.pipeline = pipeline
        self.metadata = dict(metadata or {})

        # Order mirrors the documented safety flow: metadata -> promotion -> threshold
        # -> classifier -> (schema, per batch).
        self._validate_required_metadata()
        self._check_promotion(allow_unpromoted)
        self._threshold = self._resolve_threshold()
        self._positive_index = self._resolve_positive_class_index()

        self._id_column = str(self.metadata.get("id_column") or ID_OUTPUT_COLUMN)
        self._expected_columns = self._resolve_expected_columns()

        logger.info(
            "Predictor ready: model=%s, threshold=%.4f, %d expected raw feature(s)",
            self.model_name, self._threshold, len(self._expected_columns),
        )

    # -------------------------------------------------------------------- validation

    @staticmethod
    def _validate_pipeline_shape(pipeline) -> None:
        """Reject anything that is not a fitted preprocessing+model Pipeline."""
        if not isinstance(pipeline, Pipeline):
            raise ModelArtifactError(
                f"Predictor requires a sklearn Pipeline carrying its own preprocessing, "
                f"got {type(pipeline).__name__}; a bare estimator cannot score raw data."
            )
        for step in (PREPROCESSOR_STEP, MODEL_STEP):
            if step not in pipeline.named_steps:
                raise ModelArtifactError(
                    f"Champion pipeline has no '{step}' step (found: "
                    f"{list(pipeline.named_steps)}); it cannot serve raw input."
                )
        try:
            check_is_fitted(pipeline.named_steps[PREPROCESSOR_STEP])
        except Exception as err:
            raise ModelArtifactError(
                f"Champion pipeline's '{PREPROCESSOR_STEP}' step is not fitted: {err}. "
                f"Inference is transform-only and will not fit it."
            ) from err

    def _validate_required_metadata(self) -> None:
        """Require every field production inference depends on.

        No field here has a safe default. Defaulting `promoted` would serve a rejected
        model, defaulting `optimal_threshold` would score at an operating point nobody
        selected, and defaulting the feature list would invent a schema contract. All
        missing fields are reported at once — an artifact broken in one way is usually
        broken in several.
        """
        missing = [
            key for key in REQUIRED_METADATA_KEYS
            if key not in self.metadata or self.metadata[key] is None
        ]
        if missing:
            raise ModelArtifactError(
                f"Deployment metadata is missing required field(s): {sorted(missing)}. "
                f"These are production-critical and have no safe defaults; regenerate "
                f"the artifact with scripts/train.py."
            )

    def _check_promotion(self, allow_unpromoted: bool) -> None:
        """Refuse to serve a champion that failed its Step 4 quality gates.

        The type is checked strictly rather than coerced with `bool()`, because the
        string ``"false"`` is truthy in Python — a malformed metadata file would
        otherwise read as *promoted* and serve the exact model this gate exists to stop.
        """
        promoted = self.metadata["promoted"]
        if not isinstance(promoted, (bool, np.bool_)):
            raise ModelArtifactError(
                f"Metadata field 'promoted' must be a boolean, got "
                f"{type(promoted).__name__} ({promoted!r}). A non-boolean cannot be "
                f"interpreted safely: coercing it risks reading a rejected model as "
                f"approved."
            )

        if bool(promoted):
            return

        if allow_unpromoted:
            logger.warning(
                "Serving UNPROMOTED champion '%s' because allow_unpromoted=True. This "
                "model failed its Step 4 quality gates: %s. Debugging use only — never "
                "production.",
                self.model_name, self._gate_failure_summary(),
            )
            return

        raise ModelArtifactError(
            f"Champion '{self.model_name}' is not promoted and will not be served. It "
            f"failed its Step 4 quality gates: {self._gate_failure_summary()}. Retrain "
            f"or fix the model; pass allow_unpromoted=True only for local debugging."
        )

    def _gate_failure_summary(self) -> str:
        """Which gates the champion failed, for the refusal message."""
        failures = [
            *(self.metadata.get("pre_threshold_gate_failures") or []),
            *(self.metadata.get("operational_gate_failures") or []),
        ]
        return "; ".join(str(f) for f in failures) if failures else "no detail recorded"

    def _resolve_threshold(self) -> float:
        """Frozen decision threshold, validated as a real probability.

        A threshold outside [0, 1] is not a conservative setting, it is a constant
        classifier: above 1 nothing is ever flagged, below 0 everything is. NaN is worse
        still — every ``>=`` comparison is False, so the model silently predicts the
        negative class for the entire batch while looking like it ran fine.
        """
        raw = self.metadata["optimal_threshold"]

        # bool is a subclass of int; True would otherwise sail through as 1.0.
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise ModelArtifactError(
                f"Metadata 'optimal_threshold' must be numeric, got "
                f"{type(raw).__name__} ({raw!r})."
            )

        value = float(raw)
        if not math.isfinite(value):
            raise ModelArtifactError(
                f"Metadata 'optimal_threshold' is {value}, which is not a finite number. "
                f"Comparisons against NaN are always False, so the model would silently "
                f"predict the negative class for every applicant."
            )
        if not 0.0 <= value <= 1.0:
            raise ModelArtifactError(
                f"Metadata 'optimal_threshold' is {value}, outside the valid probability "
                f"range [0, 1]; no probability could ever cross it in one direction."
            )
        return value

    def _resolve_positive_class_index(self) -> int:
        """Column of `predict_proba` holding P(default), located via `classes_`.

        There is deliberately **no fallback** to column 1. The position depends on label
        ordering, so guessing it would invert every score in the batch while producing
        perfectly well-formed output — a failure that no downstream check would catch.
        An artifact that cannot say what its classes are is not servable.
        """
        model = self.pipeline.named_steps[MODEL_STEP]
        classes = getattr(model, "classes_", None)
        if classes is None:
            raise ModelArtifactError(
                f"Champion model ({type(model).__name__}) exposes no 'classes_', so the "
                f"positive-class probability column cannot be identified. Guessing it "
                f"would invert every score."
            )

        observed = list(classes.tolist() if hasattr(classes, "tolist") else classes)
        if set(observed) != EXPECTED_CLASSES:
            raise ModelArtifactError(
                f"Champion model was fitted on classes {observed}, but this project "
                f"scores a binary 0/1 default target. Refusing to serve an incompatible "
                f"classifier."
            )
        return observed.index(POSITIVE_CLASS)

    # ------------------------------------------------------------------ construction

    @classmethod
    def load(cls, model_dir, artifact_dir, *, allow_unpromoted: bool = False) -> "Predictor":
        """Load the champion pipeline and its metadata sidecar from disk.

        Args:
            model_dir: Directory holding `champion_pipeline.joblib`.
            artifact_dir: Directory holding `deployment_meta.json`.
            allow_unpromoted: Debugging override; see `__init__`. Defaults to False.

        Raises:
            ModelArtifactError: if either file is absent — with the path, since "run
                scripts/train.py first" is the usual cause and a bare FileNotFoundError
                does not say so — or if the loaded artifact fails any safety check.
        """
        pipeline_path = Path(model_dir) / CHAMPION_PIPELINE_FILENAME
        metadata_path = Path(artifact_dir) / METADATA_FILENAME

        if not pipeline_path.exists():
            raise ModelArtifactError(
                f"No champion pipeline at {pipeline_path}. Run scripts/train.py to "
                f"produce one; inference never trains a model."
            )
        if not metadata_path.exists():
            raise ModelArtifactError(
                f"No deployment metadata at {metadata_path}. It carries the frozen "
                f"decision threshold, which the pipeline itself cannot."
            )

        pipeline = joblib.load(pipeline_path)
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        logger.info("Loaded champion pipeline from %s", pipeline_path)
        return cls(pipeline, metadata, allow_unpromoted=allow_unpromoted)

    def _resolve_expected_columns(self) -> list[str]:
        """Raw columns the pipeline was fitted on.

        Read from the fitted preprocessor first: it is the artifact's own record of what
        it was trained on and therefore cannot drift from the model. Metadata is only a
        fallback, for a preprocessor fitted on an unnamed array.
        """
        try:
            return get_input_feature_names(self.pipeline.named_steps[PREPROCESSOR_STEP])
        except Exception:
            for key in ("input_feature_columns", "raw_feature_columns"):
                columns = self.metadata.get(key)
                if columns:
                    logger.warning(
                        "Preprocessor exposes no feature names; falling back to "
                        "metadata['%s'] for the input schema", key,
                    )
                    return [str(c) for c in columns]
            raise ModelArtifactError(
                "Cannot determine the training input schema: the fitted preprocessor "
                "exposes no feature names and metadata has neither "
                "'input_feature_columns' nor 'raw_feature_columns'."
            )

    # -------------------------------------------------------------------- properties

    @property
    def threshold(self) -> float:
        """Decision threshold frozen on validation in Step 4."""
        return self._threshold

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    @property
    def expected_columns(self) -> list[str]:
        """Raw feature columns required at inference, in training order (a copy)."""
        return list(self._expected_columns)

    # ------------------------------------------------------------------------ schema

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reconcile a raw frame with the training schema. Never mutates `X`.

        Missing columns are fatal; extras are dropped and the order is corrected by
        `align_to_training_schema`, so there is exactly one implementation of that
        policy in the codebase.

        Raises:
            InferenceSchemaError: if `X` is not a DataFrame, is empty, or lacks columns
                the pipeline was fitted on.
        """
        if not isinstance(X, pd.DataFrame):
            raise InferenceSchemaError(
                f"Inference input must be a pandas DataFrame, got {type(X).__name__}"
            )
        if len(X) == 0:
            raise InferenceSchemaError("Inference input is empty (0 rows)")

        missing = [c for c in self._expected_columns if c not in X.columns]
        if missing:
            preview = missing[:10]
            raise InferenceSchemaError(
                f"Inference input is missing {len(missing)} column(s) the champion was "
                f"trained on: {preview}{' ...' if len(missing) > len(preview) else ''}. "
                f"Build inference features with the same relational aggregation and "
                f"domain feature engineering used in training."
            )

        # Only drops and reorders now — the missing-column branch cannot trigger.
        return align_to_training_schema(X, self._expected_columns)

    # ------------------------------------------------------------------- predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(default) per row, as a 1-D array aligned to `X`'s row order."""
        features = self.prepare_features(X)
        proba = self.pipeline.predict_proba(features)
        return np.asarray(proba)[:, self._positive_index]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard 0/1 predictions at the **frozen** threshold, not sklearn's 0.5."""
        return (self.predict_proba(X) >= self._threshold).astype(int)

    def predict_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score a raw applicant frame into the deliverable prediction table.

        Args:
            X: Raw applicant features **including** the identifier column. The
                identifier is carried to the output and never reaches the model —
                `prepare_features` drops it as an extra column, because training built
                its feature matrix without it.

        Returns:
            A new frame of ``SK_ID_CURR, DEFAULT_PROBABILITY, PREDICTED_CLASS``, one row
            per input row in input order, with float probabilities and integer classes.

        Raises:
            InferenceSchemaError: if the identifier column is absent — predictions
                without a key cannot be joined back to an applicant.
        """
        if self._id_column not in X.columns:
            raise InferenceSchemaError(
                f"Inference input has no identifier column '{self._id_column}'; "
                f"predictions could not be attributed to an applicant."
            )

        probability = self.predict_proba(X)
        predictions = pd.DataFrame({
            ID_OUTPUT_COLUMN: X[self._id_column].to_numpy(),
            PROBABILITY_COLUMN: probability.astype(float),
            CLASS_COLUMN: (probability >= self._threshold).astype(int),
        })
        logger.info(
            "Scored %d applicant(s) with %s at threshold %.4f: %d flagged (%.2f%%)",
            len(predictions), self.model_name, self._threshold,
            int(predictions[CLASS_COLUMN].sum()),
            100.0 * float(predictions[CLASS_COLUMN].mean()),
        )
        return predictions
