"""Optional TensorRT status, memory-policy, and CUDA runtime primitives."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


TENSORRT_STATUS_READY = "READY"
TENSORRT_STATUS_CLI_ONLY = "CLI_ONLY"
TENSORRT_STATUS_BLOCKED = "BLOCKED"


class TensorRTBlockedError(RuntimeError):
    """Raised when a formal TensorRT experiment cannot be honestly run."""


def strongly_typed_network_flags(trt: Any) -> int:
    """Return the current TensorRT flag bit for a strongly typed network."""

    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    flag = getattr(flag_enum, "STRONGLY_TYPED", None)
    if flag is None:
        raise TensorRTBlockedError(
            "TensorRT Python API does not expose STRONGLY_TYPED network creation"
        )
    try:
        return 1 << int(flag)
    except (TypeError, ValueError, OverflowError) as error:
        raise TensorRTBlockedError(
            "TensorRT STRONGLY_TYPED network flag is not usable"
        ) from error


def create_strongly_typed_network(builder: Any, trt: Any) -> Any:
    """Create a network only through TensorRT's strongly typed API."""

    flags = strongly_typed_network_flags(trt)
    try:
        network = builder.create_network(flags)
    except Exception as error:  # pragma: no cover - optional host dependency
        raise TensorRTBlockedError(
            "TensorRT could not create the required strongly typed network"
        ) from error
    if network is None:
        raise TensorRTBlockedError(
            "TensorRT returned no strongly typed network definition"
        )
    return network


def disable_tf32(builder_config: Any, trt: Any) -> bool:
    """Disable TF32 and verify the builder config reports it disabled."""

    builder_flag = getattr(trt, "BuilderFlag", None)
    tf32_flag = getattr(builder_flag, "TF32", None)
    clear_flag = getattr(builder_config, "clear_flag", None)
    get_flag = getattr(builder_config, "get_flag", None)
    if tf32_flag is None or not callable(clear_flag) or not callable(get_flag):
        raise TensorRTBlockedError(
            "TensorRT builder config cannot explicitly disable TF32"
        )
    try:
        clear_flag(tf32_flag)
        enabled = bool(get_flag(tf32_flag))
    except Exception as error:  # pragma: no cover - optional host dependency
        raise TensorRTBlockedError(
            "TensorRT TF32 state could not be verified after clear_flag"
        ) from error
    if enabled:
        raise TensorRTBlockedError(
            "TensorRT builder config still has TF32 enabled after clear_flag"
        )
    return False


def _as_shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"TensorRT returned an invalid shape: {value!r}") from error


def _torch_dtype(trt: Any, dtype: Any) -> torch.dtype:
    data_type = getattr(trt, "DataType", None)
    if data_type is not None and dtype == data_type.FLOAT:
        return torch.float32
    if data_type is not None and dtype == data_type.HALF:
        return torch.float16
    raise TypeError(f"unsupported TensorRT tensor dtype: {dtype!r}")


class TensorRTSession:
    """Execute one serialized TensorRT engine on a verified CUDA device.

    TensorRT is deliberately imported only when this class is constructed.
    The session accepts and returns CUDA tensors for the timed path, while
    ``predict_model_input`` provides a small NumPy bridge for parity fixtures.
    No CPU execution fallback is available.
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device_index: int = 0,
        tensorrt_site_packages: str | Path | None = None,
        expected_input_name: str = "observation",
        expected_output_name: str = "q_values",
        precision: str | None = None,
    ) -> None:
        if isinstance(device_index, bool) or not isinstance(device_index, int):
            raise TypeError("device_index must be a non-negative integer")
        if device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        if not torch.cuda.is_available():
            raise TensorRTBlockedError(
                "TensorRT runtime requires an NVIDIA CUDA device; CPU fallback is disabled"
            )
        if device_index >= torch.cuda.device_count():
            raise TensorRTBlockedError(
                f"CUDA device index {device_index} is unavailable"
            )
        if not isinstance(expected_input_name, str) or not expected_input_name.strip():
            raise ValueError("expected_input_name must be a non-empty string")
        if (
            not isinstance(expected_output_name, str)
            or not expected_output_name.strip()
        ):
            raise ValueError("expected_output_name must be a non-empty string")
        if precision is not None and precision not in {"float32", "float16"}:
            raise ValueError("precision must be float32, float16, or None")

        source = Path(engine_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if tensorrt_site_packages is not None:
            site_packages = Path(tensorrt_site_packages).resolve()
            if not site_packages.is_dir():
                raise FileNotFoundError(site_packages)
            sys.path.insert(0, str(site_packages))
        try:
            import tensorrt as trt
        except Exception as error:  # pragma: no cover - optional host dependency
            raise TensorRTBlockedError(
                "TensorRT Python API is unavailable; CPU fallback is disabled"
            ) from error

        self.engine_path = source
        self.device = torch.device(f"cuda:{device_index}")
        self.device_index = device_index
        self.precision = precision
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        serialized = source.read_bytes()
        self.engine_sha256 = hashlib.sha256(serialized).hexdigest()
        self.engine = self._runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {source}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")

        names = [
            str(self.engine.get_tensor_name(index))
            for index in range(self.engine.num_io_tensors)
        ]
        input_mode = trt.TensorIOMode.INPUT
        output_mode = trt.TensorIOMode.OUTPUT
        input_names = [
            name for name in names if self.engine.get_tensor_mode(name) == input_mode
        ]
        output_names = [
            name for name in names if self.engine.get_tensor_mode(name) == output_mode
        ]
        if len(input_names) != 1 or len(output_names) != 1:
            raise ValueError(
                "TensorRT engine must expose exactly one input and one output: "
                f"inputs={input_names}, outputs={output_names}"
            )
        if (
            input_names[0] != expected_input_name
            or output_names[0] != expected_output_name
        ):
            raise ValueError(
                "TensorRT engine I/O names do not match the inference contract: "
                f"input={input_names[0]!r}, output={output_names[0]!r}"
            )
        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.input_dtype = _torch_dtype(
            trt, self.engine.get_tensor_dtype(self.input_name)
        )
        self.output_dtype = _torch_dtype(
            trt, self.engine.get_tensor_dtype(self.output_name)
        )
        if self.input_dtype != torch.float32 or self.output_dtype != torch.float32:
            raise TypeError(
                "TensorRT engine I/O must remain float32 at the model boundary: "
                f"input={self.input_dtype}, output={self.output_dtype}"
            )
        self.input_shape = _as_shape(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = _as_shape(self.engine.get_tensor_shape(self.output_name))
        if self.input_shape[1:] != (4, 84, 84) or self.output_shape[1:] != (4,):
            raise ValueError(
                "TensorRT engine shapes do not match Breakout inference contract: "
                f"input={self.input_shape}, output={self.output_shape}"
            )
        self.profile_shapes = tuple(
            _as_shape(shape)
            for shape in self.engine.get_tensor_profile_shape(self.input_name, 0)
        )
        if len(self.profile_shapes) != 3:
            raise ValueError("TensorRT engine must expose min/opt/max profile shapes")
        self._runtime_metadata = {
            "runtime": "TensorRT",
            "requested_provider": "TensorRTExecutionProvider",
            "actual_provider": "TensorRT",
            "engine_filename": self.engine_path.name,
            "engine_sha256": self.engine_sha256,
            "tensorrt_version": str(getattr(trt, "__version__", "unknown")),
            "cuda_device_index": device_index,
            "gpu_model": torch.cuda.get_device_name(device_index),
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_dtype": str(self.input_dtype).replace("torch.", ""),
            "output_dtype": str(self.output_dtype).replace("torch.", ""),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "optimization_profile": {
                "min": list(self.profile_shapes[0]),
                "opt": list(self.profile_shapes[1]),
                "max": list(self.profile_shapes[2]),
            },
            "precision": precision,
        }

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        """Return immutable-at-call-time runtime facts for evidence artifacts."""

        return dict(self._runtime_metadata)

    def execute(self, model_input: torch.Tensor) -> torch.Tensor:
        """Validate input, launch the engine asynchronously, and return output."""

        return self._execute(model_input, validate=True)

    def execute_prevalidated(self, model_input: torch.Tensor) -> torch.Tensor:
        """Launch a prepared CUDA tensor without repeating value validation.

        This is the model-only benchmark seam. The caller has already prepared
        and range-checked the tensor outside the timed loop; synchronization for
        measuring the asynchronous CUDA launch remains the benchmark's job.
        """

        return self._execute(model_input, validate=False)

    def _execute(self, model_input: torch.Tensor, *, validate: bool) -> torch.Tensor:
        """Launch the engine, optionally performing the public-boundary checks."""

        if not isinstance(model_input, torch.Tensor):
            raise TypeError("model_input must be a torch.Tensor")
        if model_input.dtype != torch.float32:
            raise TypeError("model_input must have dtype torch.float32")
        if model_input.device != self.device:
            raise ValueError(
                "model_input must already be on the TensorRT CUDA device: "
                f"input={model_input.device}, engine={self.device}"
            )
        if model_input.ndim != 4 or tuple(model_input.shape[1:]) != (4, 84, 84):
            raise ValueError(
                "model_input must have shape (N, 4, 84, 84); "
                f"received {tuple(model_input.shape)}"
            )
        batch_size = int(model_input.shape[0])
        min_shape, _opt_shape, max_shape = self.profile_shapes
        if batch_size < min_shape[0] or batch_size > max_shape[0]:
            raise ValueError(
                "model_input batch is outside the TensorRT optimization profile: "
                f"batch={batch_size}, range=[{min_shape[0]}, {max_shape[0]}]"
            )
        contiguous_input = model_input.contiguous()
        if validate and (
            not torch.isfinite(contiguous_input).all().item()
            or not (
                (0.0 <= contiguous_input).all().item()
                and (contiguous_input <= 1.0).all().item()
            )
        ):
            raise ValueError("model_input must contain finite values in [0, 1]")
        try:
            shape_accepted = self.context.set_input_shape(
                self.input_name, tuple(int(value) for value in contiguous_input.shape)
            )
            if shape_accepted is False:
                raise RuntimeError("TensorRT rejected the input shape")
            output_shape = _as_shape(self.context.get_tensor_shape(self.output_name))
            if any(value < 0 for value in output_shape):
                raise RuntimeError(
                    f"TensorRT did not resolve output shape: {output_shape}"
                )
            expected_output_shape = (batch_size, 4)
            if output_shape != expected_output_shape:
                raise ValueError(
                    "TensorRT output shape does not match the inference contract: "
                    f"observed={output_shape}, expected={expected_output_shape}"
                )
            output = torch.empty(
                output_shape,
                dtype=self.output_dtype,
                device=self.device,
            )
            self.context.set_tensor_address(
                self.input_name, int(contiguous_input.data_ptr())
            )
            self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
            launched = self.context.execute_async_v3(
                torch.cuda.current_stream(self.device).cuda_stream
            )
            if launched is False:
                raise RuntimeError("TensorRT execution failed")
        except Exception as error:
            if isinstance(error, (RuntimeError, TypeError, ValueError)):
                raise
            raise RuntimeError(
                "TensorRT execution failed; CPU fallback is disabled"
            ) from error
        return output

    def predict_model_input(self, model_input: np.ndarray) -> np.ndarray:
        """Run prepared float32 input and return finite CPU float32 Q-values."""

        if not isinstance(model_input, np.ndarray):
            raise TypeError("model_input must be a numpy.ndarray")
        if model_input.dtype != np.dtype("float32"):
            raise TypeError("model_input must have dtype float32")
        if model_input.ndim != 4 or tuple(model_input.shape[1:]) != (4, 84, 84):
            raise ValueError(
                "model_input must have shape (N, 4, 84, 84); "
                f"received {tuple(model_input.shape)}"
            )
        tensor = torch.as_tensor(
            np.ascontiguousarray(model_input),
            dtype=torch.float32,
            device=self.device,
        )
        output = self.execute(tensor)
        torch.cuda.synchronize(self.device)
        values = output.detach().cpu().numpy()
        return np.ascontiguousarray(values, dtype=np.float32)


def classify_tensorrt_status(
    *,
    tensorrt_imported: bool,
    onnx_parser_available: bool,
    builder_available: bool,
    strongly_typed_network_available: bool,
    trtexec_available: bool,
) -> str:
    """Classify the strongest available build path without guessing support."""

    if (
        tensorrt_imported
        and onnx_parser_available
        and builder_available
        and strongly_typed_network_available
    ):
        return TENSORRT_STATUS_READY
    if (
        trtexec_available
        and onnx_parser_available
        and not (tensorrt_imported and builder_available)
    ):
        return TENSORRT_STATUS_CLI_ONLY
    return TENSORRT_STATUS_BLOCKED


def validate_workspace_budget(
    *,
    workspace_bytes: int,
    free_bytes: int,
    headroom_bytes: int,
) -> int:
    """Validate a builder workspace limit against observed free VRAM."""

    for value, name in (
        (workspace_bytes, "workspace_bytes"),
        (free_bytes, "free_bytes"),
        (headroom_bytes, "headroom_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if workspace_bytes > max(free_bytes - headroom_bytes, 0):
        raise TensorRTBlockedError(
            "TensorRT workspace limit would violate the required VRAM headroom: "
            f"workspace={workspace_bytes}, free={free_bytes}, headroom={headroom_bytes}"
        )
    return workspace_bytes


def require_gpu_baseline(
    *,
    torch_cuda_available: bool,
    ort_actual_provider: Any,
    ort_graph_assignment_status: Any,
) -> None:
    """Require a verified PyTorch/ORT CUDA baseline before TensorRT work."""

    if not torch_cuda_available:
        raise TensorRTBlockedError(
            "TensorRT experiment is blocked: PyTorch CUDA baseline is unavailable"
        )
    if str(ort_actual_provider) != "CUDAExecutionProvider":
        raise TensorRTBlockedError(
            "TensorRT experiment is blocked: ORT CUDA baseline is not active; "
            f"actual_provider={ort_actual_provider!r}"
        )
    if str(ort_graph_assignment_status) != "verified":
        raise TensorRTBlockedError(
            "TensorRT experiment is blocked: ORT CUDA graph assignment is not verified"
        )


__all__ = [
    "TENSORRT_STATUS_BLOCKED",
    "TENSORRT_STATUS_CLI_ONLY",
    "TENSORRT_STATUS_READY",
    "TensorRTBlockedError",
    "TensorRTSession",
    "classify_tensorrt_status",
    "create_strongly_typed_network",
    "disable_tf32",
    "require_gpu_baseline",
    "strongly_typed_network_flags",
    "validate_workspace_budget",
]
