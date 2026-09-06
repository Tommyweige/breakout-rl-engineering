"""The shared model-boundary contract for native and future browser inference."""

from __future__ import annotations

import json
import operator
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from breakout_rl.tensors import OBSERVATION_SHAPE, observation_to_tensor
from breakout_rl.training.diagnostics import ATARI_ACTION_NAMES


DEFAULT_INFERENCE_SPEC_PATH = Path("configs/inference/inference_spec.json")
"""Repository-relative location of the machine-readable inference contract."""

EXPECTED_ACTION_MEANINGS = tuple(
    ATARI_ACTION_NAMES[index] for index in range(len(ATARI_ACTION_NAMES))
)
"""The minimal ALE action order used by every exported policy."""

ShapeDimension = int | str


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _parse_shape(value: Any, *, name: str) -> tuple[ShapeDimension, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of dimensions")
    dimensions: list[ShapeDimension] = []
    for dimension in value:
        if isinstance(dimension, str):
            if dimension != "N":
                raise ValueError(f"{name} only permits the dynamic dimension 'N'")
            dimensions.append(dimension)
            continue
        if isinstance(dimension, bool):
            raise TypeError(f"{name} dimensions must be positive integers or 'N'")
        try:
            parsed = operator.index(dimension)
        except TypeError as error:
            raise TypeError(
                f"{name} dimensions must be positive integers or 'N'"
            ) from error
        if parsed < 1:
            raise ValueError(f"{name} dimensions must be positive")
        dimensions.append(int(parsed))
    if not dimensions:
        raise ValueError(f"{name} must not be empty")
    return tuple(dimensions)


def _parse_float_pair(value: Any, *, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must contain exactly two numbers")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    try:
        parsed = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain exactly two numbers") from error
    if not np.isfinite(parsed).all() or parsed[0] > parsed[1]:
        raise ValueError(f"{name} must be finite and ordered")
    return parsed


def _parse_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(parsed)


@dataclass(frozen=True)
class InferenceSpec:
    """Validated, machine-readable semantics at the model boundary."""

    schema_version: int
    contract_id: str
    environment_contract_id: str
    environment_contract_path: str
    environment_contract_sha256: str
    input_name: str
    input_dtype: str
    input_shape: tuple[ShapeDimension, ...]
    input_layout: str
    input_range: tuple[float, float]
    source_observation_dtype: str
    source_observation_shape: tuple[int, int, int]
    normalization_divisor: float
    output_name: str
    output_dtype: str
    output_shape: tuple[ShapeDimension, ...]
    action_meanings: tuple[str, ...]
    greedy_rule: str
    frame_stack: int
    screen_size: int
    model_relative_url: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("inference spec schema_version must be 1")
        _non_empty_string(self.contract_id, name="contract_id")
        _non_empty_string(
            self.environment_contract_id,
            name="environment_contract_id",
        )
        _non_empty_string(
            self.environment_contract_path,
            name="environment_contract_path",
        )
        if len(self.environment_contract_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.environment_contract_sha256.lower()
        ):
            raise ValueError("environment_contract_sha256 must be a SHA256 hex digest")
        if self.input_dtype != "float32":
            raise ValueError("input_dtype must be float32")
        if self.input_layout != "NCHW":
            raise ValueError("input_layout must be NCHW")
        if self.source_observation_dtype != "uint8":
            raise ValueError("source_observation_dtype must be uint8")
        if self.source_observation_shape != OBSERVATION_SHAPE:
            raise ValueError(
                "source_observation_shape must match the Contract v2 observation "
                f"shape {OBSERVATION_SHAPE}"
            )
        if self.input_shape != ("N", *OBSERVATION_SHAPE):
            raise ValueError(
                "input_shape must be ['N', 4, 84, 84] for the Breakout policy"
            )
        if self.input_range != (0.0, 1.0):
            raise ValueError("input_range must be [0, 1]")
        if self.normalization_divisor != 255.0:
            raise ValueError("normalization_divisor must be 255")
        if self.output_dtype != "float32":
            raise ValueError("output_dtype must be float32")
        if self.output_shape != ("N", len(EXPECTED_ACTION_MEANINGS)):
            raise ValueError("output_shape must be ['N', 4]")
        if self.action_meanings != EXPECTED_ACTION_MEANINGS:
            raise ValueError(
                "action_meanings must preserve the ALE order "
                f"{list(EXPECTED_ACTION_MEANINGS)}"
            )
        if self.greedy_rule != "argmax":
            raise ValueError("greedy_rule must be argmax")
        if self.frame_stack != OBSERVATION_SHAPE[0]:
            raise ValueError("frame_stack must be 4")
        if self.screen_size != OBSERVATION_SHAPE[1]:
            raise ValueError("screen_size must be 84")
        if (
            not self.model_relative_url
            or self.model_relative_url.startswith(("/", "\\"))
            or "://" in self.model_relative_url
            or "\\" in self.model_relative_url
        ):
            raise ValueError("model_relative_url must be a same-origin relative URL")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "InferenceSpec":
        """Build and validate a spec loaded from JSON or another mapping."""

        if not isinstance(values, Mapping):
            raise TypeError("inference spec must be a JSON object")
        input_values = values.get("input")
        output_values = values.get("output")
        environment_values = values.get("environment_contract")
        preprocessing_values = values.get("preprocessing")
        actions_values = values.get("actions")
        deployment_values = values.get("deployment")
        for name, candidate in (
            ("input", input_values),
            ("output", output_values),
            ("environment_contract", environment_values),
            ("preprocessing", preprocessing_values),
            ("actions", actions_values),
            ("deployment", deployment_values),
        ):
            if not isinstance(candidate, Mapping):
                raise TypeError(f"{name} must be a JSON object")

        source_shape = _parse_shape(
            preprocessing_values["source_observation_shape"],
            name="preprocessing.source_observation_shape",
        )
        if "N" in source_shape or len(source_shape) != 3:
            raise ValueError(
                "preprocessing.source_observation_shape must contain three fixed dimensions"
            )
        action_meanings = actions_values.get("meanings")
        if isinstance(action_meanings, (str, bytes)) or not isinstance(
            action_meanings,
            Sequence,
        ):
            raise TypeError("actions.meanings must be a sequence")

        return cls(
            schema_version=_parse_int(
                values.get("schema_version"),
                name="schema_version",
            ),
            contract_id=_non_empty_string(
                values.get("contract_id"),
                name="contract_id",
            ),
            environment_contract_id=_non_empty_string(
                environment_values.get("contract_id"),
                name="environment_contract.contract_id",
            ),
            environment_contract_path=_non_empty_string(
                environment_values.get("path"),
                name="environment_contract.path",
            ),
            environment_contract_sha256=_non_empty_string(
                environment_values.get("sha256"),
                name="environment_contract.sha256",
            ).lower(),
            input_name=_non_empty_string(
                input_values.get("name"),
                name="input.name",
            ),
            input_dtype=_non_empty_string(
                input_values.get("dtype"),
                name="input.dtype",
            ),
            input_shape=_parse_shape(input_values.get("shape"), name="input.shape"),
            input_layout=_non_empty_string(
                input_values.get("layout"),
                name="input.layout",
            ),
            input_range=_parse_float_pair(
                input_values.get("range"),
                name="input.range",
            ),
            source_observation_dtype=_non_empty_string(
                preprocessing_values.get("source_observation_dtype"),
                name="preprocessing.source_observation_dtype",
            ),
            source_observation_shape=tuple(int(dimension) for dimension in source_shape),  # type: ignore[arg-type]
            normalization_divisor=float(
                preprocessing_values.get("normalization_divisor")
            ),
            output_name=_non_empty_string(
                output_values.get("name"),
                name="output.name",
            ),
            output_dtype=_non_empty_string(
                output_values.get("dtype"),
                name="output.dtype",
            ),
            output_shape=_parse_shape(
                output_values.get("shape"),
                name="output.shape",
            ),
            action_meanings=tuple(str(value) for value in action_meanings),
            greedy_rule=_non_empty_string(
                actions_values.get("greedy_rule"),
                name="actions.greedy_rule",
            ),
            frame_stack=_parse_int(
                preprocessing_values.get("frame_stack"),
                name="preprocessing.frame_stack",
            ),
            screen_size=_parse_int(
                preprocessing_values.get("screen_size"),
                name="preprocessing.screen_size",
            ),
            model_relative_url=_non_empty_string(
                deployment_values.get("model_relative_url"),
                name="deployment.model_relative_url",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the contract."""

        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "environment_contract": {
                "contract_id": self.environment_contract_id,
                "path": self.environment_contract_path,
                "sha256": self.environment_contract_sha256,
            },
            "input": {
                "name": self.input_name,
                "dtype": self.input_dtype,
                "shape": list(self.input_shape),
                "layout": self.input_layout,
                "range": list(self.input_range),
            },
            "preprocessing": {
                "owner": "environment_then_model_adapter",
                "source_observation_dtype": self.source_observation_dtype,
                "source_observation_shape": list(self.source_observation_shape),
                "frame_stack": self.frame_stack,
                "screen_size": self.screen_size,
                "normalization": "divide uint8 values by 255.0 once",
                "normalization_divisor": self.normalization_divisor,
                "onnx_graph_owns": [],
            },
            "output": {
                "name": self.output_name,
                "dtype": self.output_dtype,
                "shape": list(self.output_shape),
                "meaning": "raw Q-values, not probabilities",
            },
            "actions": {
                "meanings": list(self.action_meanings),
                "greedy_rule": self.greedy_rule,
                "index_base": 0,
            },
            "deployment": {
                "runtime": "browser client-side",
                "model_relative_url": self.model_relative_url,
            },
        }


def load_inference_spec(
    path: str | Path = DEFAULT_INFERENCE_SPEC_PATH,
) -> InferenceSpec:
    """Load and validate the repository's machine-readable inference spec."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        values = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    return InferenceSpec.from_mapping(values)


def default_inference_spec() -> InferenceSpec:
    """Load the default contract from the current repository or package root."""

    candidates = (
        DEFAULT_INFERENCE_SPEC_PATH,
        Path(__file__).resolve().parents[1] / DEFAULT_INFERENCE_SPEC_PATH,
    )
    for candidate in candidates:
        if candidate.is_file():
            return load_inference_spec(candidate)
    raise FileNotFoundError(DEFAULT_INFERENCE_SPEC_PATH)


def _coerce_spec(spec: InferenceSpec | Mapping[str, Any] | None) -> InferenceSpec:
    if spec is None:
        return default_inference_spec()
    if isinstance(spec, InferenceSpec):
        return spec
    if isinstance(spec, Mapping):
        return InferenceSpec.from_mapping(spec)
    raise TypeError("spec must be an InferenceSpec, mapping, or None")


def prepare_model_input(
    observation: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    spec: InferenceSpec | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Convert uint8 CHW/BCHW observations to one normalized BCHW tensor.

    The environment remains responsible for grayscale conversion, resize,
    frame skip, and frame stacking.  This seam only changes the storage type,
    divides by 255 once, and adds the batch dimension for a single state.
    """

    contract = _coerce_spec(spec)
    if not isinstance(observation, np.ndarray):
        raise TypeError("observation must be a numpy.ndarray")
    if observation.dtype != np.dtype(contract.source_observation_dtype):
        raise TypeError(
            "observation must have dtype uint8; normalize only at the model boundary"
        )
    if observation.ndim == 3:
        if tuple(observation.shape) != contract.source_observation_shape:
            raise ValueError(
                "observation must have shape (4, 84, 84) or (B, 4, 84, 84); "
                f"received {tuple(observation.shape)}"
            )
    elif observation.ndim == 4:
        if (
            observation.shape[0] < 1
            or tuple(observation.shape[1:]) != contract.source_observation_shape
        ):
            raise ValueError(
                "observation must have shape (4, 84, 84) or (B, 4, 84, 84); "
                f"received {tuple(observation.shape)}"
            )
    else:
        raise ValueError("observation must have shape (4, 84, 84) or (B, 4, 84, 84)")

    tensor = observation_to_tensor(observation, device=device, add_batch_dim=True)
    if tensor.dtype != torch.float32 or tensor.ndim != 4:
        raise RuntimeError(
            "model input conversion did not produce normalized BCHW float32"
        )
    if (
        not torch.isfinite(tensor).all().item()
        or not (0.0 <= tensor).all().item()
        or not (tensor <= 1.0).all().item()
    ):
        raise RuntimeError(
            "normalized model input is outside the declared [0, 1] range"
        )
    return tensor


def _q_values_array(
    q_values: Any,
    *,
    action_count: int,
) -> tuple[np.ndarray, bool]:
    if isinstance(q_values, torch.Tensor):
        array = q_values.detach().cpu().numpy()
    else:
        array = np.asarray(q_values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
        was_single = True
    elif array.ndim == 2:
        was_single = False
    else:
        raise ValueError("q_values must have shape (4,) or (N, 4)")
    if array.shape[0] < 1 or int(array.shape[1]) != action_count:
        raise ValueError(
            f"q_values must have shape (4,) or (N, {action_count}); "
            f"received {tuple(array.shape)}"
        )
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("q_values must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError("q_values must contain only finite values")
    return array, was_single


def q_values_to_action(
    q_values: Any,
    *,
    spec: InferenceSpec | Mapping[str, Any] | None = None,
) -> int | np.ndarray:
    """Apply the contract's greedy rule and return ALE action indices.

    A single ``(4,)`` row returns one Python ``int``.  A batch ``(N, 4)``
    returns an ``int64`` array with one action per row.  NumPy/PyTorch argmax
    both choose the first index on ties, which keeps the action mapping stable.
    """

    contract = _coerce_spec(spec)
    array, was_single = _q_values_array(
        q_values,
        action_count=len(contract.action_meanings),
    )
    actions = np.argmax(array, axis=1).astype(np.int64, copy=False)
    if was_single:
        return int(actions[0])
    return actions


def action_meaning(
    action_index: Any,
    *,
    spec: InferenceSpec | Mapping[str, Any] | None = None,
) -> str:
    """Resolve one model action index to the declared ALE meaning."""

    contract = _coerce_spec(spec)
    if isinstance(action_index, bool):
        raise TypeError("action_index must be an integer")
    try:
        index = operator.index(action_index)
    except TypeError as error:
        raise TypeError("action_index must be an integer") from error
    if index < 0 or index >= len(contract.action_meanings):
        raise ValueError(
            f"action_index must be in [0, {len(contract.action_meanings) - 1}]"
        )
    return contract.action_meanings[index]


def validate_action_meanings(
    action_meanings: Sequence[Any],
    *,
    spec: InferenceSpec | Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Confirm that a runtime exposes the contract's ALE action order."""

    contract = _coerce_spec(spec)
    if isinstance(action_meanings, (str, bytes)):
        raise TypeError("action_meanings must be a sequence")
    observed = tuple(str(value) for value in action_meanings)
    if observed != contract.action_meanings:
        raise ValueError(
            "runtime action meanings do not match the inference contract: "
            f"observed={list(observed)}, expected={list(contract.action_meanings)}"
        )
    return observed


def _resolve_policy_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for PyTorchPolicy, but it is not available."
        )
    return resolved


def _load_onnxruntime() -> Any:
    """Import ONNX Runtime only when an ONNX policy is actually requested."""

    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - depends on environment setup
        raise RuntimeError(
            "ONNXRuntimePolicy requires the pinned 'onnxruntime-gpu' package; "
            "install the repository environment first"
        ) from error
    return ort


def _resolve_onnx_provider(provider: str) -> str:
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be 'cpu', 'cuda', or an ONNX provider name")
    normalized = provider.strip().lower()
    aliases = {
        "cpu": "CPUExecutionProvider",
        "cpuexecutionprovider": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cudaexecutionprovider": "CUDAExecutionProvider",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            "provider must be 'cpu', 'cuda', 'CPUExecutionProvider', or "
            "'CUDAExecutionProvider'"
        ) from error


def _validate_onnx_value_info(
    value_info: Any,
    *,
    expected_name: str,
    expected_shape: tuple[ShapeDimension, ...],
    kind: str,
) -> None:
    observed_name = str(getattr(value_info, "name", ""))
    if observed_name != expected_name:
        raise ValueError(
            f"ONNX {kind} name does not match the inference contract: "
            f"observed={observed_name!r}, expected={expected_name!r}"
        )
    observed_type = str(getattr(value_info, "type", ""))
    if observed_type != "tensor(float)":
        raise TypeError(
            f"ONNX {kind} must have dtype float32; observed {observed_type!r}"
        )
    observed_shape = tuple(getattr(value_info, "shape", ()))
    if len(observed_shape) != len(expected_shape):
        raise ValueError(
            f"ONNX {kind} rank does not match the inference contract: "
            f"observed={observed_shape}, expected={expected_shape}"
        )
    for index, (observed, expected) in enumerate(
        zip(observed_shape, expected_shape, strict=True)
    ):
        if expected == "N":
            if observed not in {None, "N"}:
                raise ValueError(
                    f"ONNX {kind} dimension {index} must be dynamic 'N'; "
                    f"observed={observed!r}"
                )
        elif observed != expected:
            raise ValueError(
                f"ONNX {kind} dimension {index} does not match the inference "
                f"contract: observed={observed!r}, expected={expected!r}"
            )


def _cuda_runtime_metadata(*, device_index: int) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_model: str | None = None
    cudnn_version: int | None = None
    if cuda_available:
        try:
            gpu_model = str(torch.cuda.get_device_name(device_index))
        except (RuntimeError, TypeError, ValueError):
            gpu_model = None
        try:
            cudnn_version_value = torch.backends.cudnn.version()
            cudnn_version = (
                int(cudnn_version_value)
                if cudnn_version_value is not None
                else None
            )
        except (RuntimeError, TypeError, ValueError):
            cudnn_version = None
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_index": device_index,
        "cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "gpu_model": gpu_model,
    }


def _graph_assignment_metadata(session: Any, *, requested_provider: str) -> dict[str, Any]:
    """Return provider assignment evidence for a strict CUDA session."""

    if requested_provider != "CUDAExecutionProvider":
        return {"status": "not_requested"}
    get_assignment = getattr(session, "get_provider_graph_assignment_info", None)
    if not callable(get_assignment):
        raise RuntimeError(
            "CUDAExecutionProvider session cannot expose graph assignment metadata; "
            "refusing to claim strict CUDA execution"
        )
    try:
        assignments = get_assignment()
    except Exception as error:  # pragma: no cover - provider/version specific
        raise RuntimeError(
            "CUDAExecutionProvider graph assignment metadata could not be read; "
            "refusing to claim strict CUDA execution"
        ) from error

    by_provider: dict[str, int] = {}
    nodes: list[dict[str, str]] = []
    for subgraph in assignments:
        provider = str(getattr(subgraph, "ep_name", ""))
        assigned_nodes = list(subgraph.get_nodes())
        by_provider[provider] = by_provider.get(provider, 0) + len(assigned_nodes)
        for node in assigned_nodes:
            nodes.append(
                {
                    "provider": provider,
                    "name": str(getattr(node, "name", "")),
                    "op_type": str(getattr(node, "op_type", "")),
                }
            )
    fallback_nodes = [
        node for node in nodes if node["provider"] != requested_provider
    ]
    if fallback_nodes:
        raise RuntimeError(
            "CUDAExecutionProvider graph assignment used a non-CUDA provider for "
            f"{len(fallback_nodes)} node(s); refusing silent fallback"
        )
    return {
        "status": "verified",
        "node_count": len(nodes),
        "nodes_by_provider": dict(sorted(by_provider.items())),
        "fallback_node_count": len(fallback_nodes),
    }


class PyTorchPolicy:
    """Run one PyTorch Q-network through the shared inference contract."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str = "cpu",
        spec: InferenceSpec | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.spec = _coerce_spec(spec)
        self.device = _resolve_policy_device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def predict_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return finite float32 Q-values with the canonical ``(N, 4)`` shape."""

        model_input = prepare_model_input(
            observation,
            device=self.device,
            spec=self.spec,
        )
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(model_input)
        if not isinstance(outputs, torch.Tensor):
            raise TypeError("model must return a torch.Tensor")
        if outputs.ndim != 2 or tuple(outputs.shape) != (
            model_input.shape[0],
            len(self.spec.action_meanings),
        ):
            raise ValueError(
                "model output must have shape " f"(N, {len(self.spec.action_meanings)})"
            )
        if outputs.dtype != torch.float32:
            raise TypeError("model output must have dtype torch.float32")
        if not torch.isfinite(outputs).all().item():
            raise ValueError("model output contains non-finite Q-values")
        return np.ascontiguousarray(outputs.detach().cpu().numpy())

    def select_actions(self, observation: np.ndarray) -> np.ndarray:
        """Return one greedy action index for every observation in a batch."""

        return np.asarray(
            q_values_to_action(self.predict_q_values(observation), spec=self.spec),
            dtype=np.int64,
        )

    def select_action(self, observation: np.ndarray) -> int:
        """Return one greedy action for a single CHW observation."""

        if not isinstance(observation, np.ndarray) or observation.ndim != 3:
            raise ValueError(
                "select_action expects one observation with shape (4, 84, 84)"
            )
        return int(
            q_values_to_action(
                self.predict_q_values(observation)[0],
                spec=self.spec,
            )
        )

    def __call__(self, observation: np.ndarray) -> int:
        return self.select_action(observation)


class ONNXRuntimePolicy:
    """Run one exported ONNX Q-network through the shared inference contract."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        provider: str = "cpu",
        device_index: int = 0,
        spec: InferenceSpec | Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = _coerce_spec(spec)
        if isinstance(device_index, bool) or not isinstance(device_index, int):
            raise TypeError("device_index must be a non-negative integer")
        if device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.requested_provider = _resolve_onnx_provider(provider)
        self.device_index = device_index
        ort = _load_onnxruntime()
        available_providers = tuple(str(value) for value in ort.get_available_providers())
        if self.requested_provider not in available_providers:
            raise RuntimeError(
                f"{self.requested_provider} requested but unavailable; "
                f"available providers={list(available_providers)}; "
                "refusing CPU fallback"
            )

        if self.requested_provider == "CUDAExecutionProvider":
            preload_dlls = getattr(ort, "preload_dlls", None)
            if callable(preload_dlls):
                try:
                    preload_dlls()
                except Exception as error:  # pragma: no cover - host dependent
                    raise RuntimeError(
                        "CUDAExecutionProvider requested but CUDA/cuDNN DLL "
                        "preloading failed; refusing CPU fallback"
                    ) from error

        session_options = ort.SessionOptions()
        if self.requested_provider == "CUDAExecutionProvider":
            try:
                session_options.add_session_config_entry(
                    "session.record_ep_graph_assignment_info",
                    "1",
                )
            except Exception as error:  # pragma: no cover - provider/version specific
                raise RuntimeError(
                    "CUDAExecutionProvider session cannot record graph assignment "
                    "metadata; refusing to claim strict CUDA execution"
                ) from error
        try:
            if self.requested_provider == "CUDAExecutionProvider":
                cuda_options: dict[str, str] = {"use_tf32": "0"}
                if device_index:
                    cuda_options["device_id"] = str(device_index)
                provider_options: list[Any] = [
                    (self.requested_provider, cuda_options)
                ]
            else:
                provider_options = [self.requested_provider]
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=provider_options,
            )
        except Exception as error:  # pragma: no cover - provider/version specific
            raise RuntimeError(
                f"failed to create {self.requested_provider} ONNX session; "
                "requested provider was not replaced with CPU"
            ) from error
        disable_fallback = getattr(self.session, "disable_fallback", None)
        if callable(disable_fallback):
            disable_fallback()

        active_providers = tuple(str(value) for value in self.session.get_providers())
        if not active_providers or active_providers[0] != self.requested_provider:
            raise RuntimeError(
                f"requested {self.requested_provider}, but ONNX Runtime selected "
                f"{list(active_providers)}; refusing silent fallback"
            )
        inputs = list(self.session.get_inputs())
        outputs = list(self.session.get_outputs())
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "ONNX model must expose exactly one input and one output; "
                f"observed inputs={len(inputs)}, outputs={len(outputs)}"
            )
        _validate_onnx_value_info(
            inputs[0],
            expected_name=self.spec.input_name,
            expected_shape=self.spec.input_shape,
            kind="input",
        )
        _validate_onnx_value_info(
            outputs[0],
            expected_name=self.spec.output_name,
            expected_shape=self.spec.output_shape,
            kind="output",
        )
        self._input_name = str(inputs[0].name)
        self._output_name = str(outputs[0].name)
        self._runtime_metadata = {
            "requested_provider": self.requested_provider,
            "available_providers": list(available_providers),
            "active_providers": list(active_providers),
            "actual_provider": active_providers[0],
            "onnxruntime_version": str(getattr(ort, "__version__", "unknown")),
            "onnxruntime_device": str(
                getattr(ort, "get_device", lambda: "unknown")()
            ),
            "session_provider_options": self.session.get_provider_options(),
            "graph_assignment": _graph_assignment_metadata(
                self.session,
                requested_provider=self.requested_provider,
            ),
            **_cuda_runtime_metadata(device_index=device_index),
        }

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        """Return provider and host-runtime facts captured at session creation."""

        return dict(self._runtime_metadata)

    def predict_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return finite float32 Q-values with the canonical ``(N, 4)`` shape."""

        model_input = prepare_model_input(
            observation,
            device="cpu",
            spec=self.spec,
        )
        input_array = np.ascontiguousarray(model_input.numpy(), dtype=np.float32)
        try:
            outputs = self.session.run(
                [self._output_name],
                {self._input_name: input_array},
            )
        except Exception as error:  # pragma: no cover - provider/version specific
            raise RuntimeError(
                f"{self.requested_provider} ONNX inference failed; "
                "the requested provider was not replaced with CPU"
            ) from error
        if len(outputs) != 1:
            raise ValueError(f"ONNX model returned {len(outputs)} outputs")
        q_values = np.asarray(outputs[0])
        expected_shape = (input_array.shape[0], len(self.spec.action_meanings))
        if q_values.dtype != np.dtype("float32"):
            raise TypeError(
                "ONNX model output must have dtype float32; "
                f"received {q_values.dtype}"
            )
        if tuple(q_values.shape) != expected_shape:
            raise ValueError(
                "ONNX model output must have shape "
                f"{expected_shape}; received {tuple(q_values.shape)}"
            )
        if not np.isfinite(q_values).all():
            raise ValueError("ONNX model output contains non-finite Q-values")
        return np.ascontiguousarray(q_values)

    def select_actions(self, observation: np.ndarray) -> np.ndarray:
        """Return one greedy action index for every observation in a batch."""

        return np.asarray(
            q_values_to_action(self.predict_q_values(observation), spec=self.spec),
            dtype=np.int64,
        )

    def select_action(self, observation: np.ndarray) -> int:
        """Return one greedy action for a single CHW observation."""

        if not isinstance(observation, np.ndarray) or observation.ndim != 3:
            raise ValueError(
                "select_action expects one observation with shape (4, 84, 84)"
            )
        return int(
            q_values_to_action(
                self.predict_q_values(observation)[0],
                spec=self.spec,
            )
        )

    def __call__(self, observation: np.ndarray) -> int:
        return self.select_action(observation)


__all__ = [
    "DEFAULT_INFERENCE_SPEC_PATH",
    "EXPECTED_ACTION_MEANINGS",
    "InferenceSpec",
    "ONNXRuntimePolicy",
    "PyTorchPolicy",
    "action_meaning",
    "default_inference_spec",
    "load_inference_spec",
    "prepare_model_input",
    "q_values_to_action",
    "validate_action_meanings",
]
