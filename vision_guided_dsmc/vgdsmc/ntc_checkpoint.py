from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .moment_sampling import PhysicalMomentAccumulator
from .vhs_model import PhysicalCavityConfig, PhysicalParticleState
from .wall_sampling import LidWallEventAccumulator


CHECKPOINT_FORMAT = "vgdsmc-ntc-checkpoint"
CHECKPOINT_VERSION = 1
_MANIFEST_KEY = "__manifest__"


class NTCCheckpointError(ValueError):
    """Base class for checkpoint validation failures."""


class NTCCheckpointCorruptionError(NTCCheckpointError):
    """Raised when a checkpoint is malformed or fails an integrity check."""


class NTCCheckpointConfigMismatchError(NTCCheckpointError):
    """Raised when a checkpoint belongs to a different physical config."""


@dataclass
class NTCCheckpoint:
    """Complete mutable state needed to continue an NTC trajectory exactly."""

    state: PhysicalParticleState
    rng_state: Mapping[str, Any]
    step_index: int
    diagnostics: Mapping[str, int | float]
    moments: PhysicalMomentAccumulator
    wall_events: LidWallEventAccumulator
    temporal_sums: Mapping[str, np.ndarray]
    temporal_sums2: Mapping[str, np.ndarray]
    temporal_nsamples: int
    block_accumulators: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def sums2(self) -> Mapping[str, np.ndarray]:
        """Alias matching the live NTC solver's ``sums2`` local variable."""
        return self.temporal_sums2

    @property
    def nsamples(self) -> int:
        """Alias matching the live NTC solver's ``nsamples`` local variable."""
        return self.temporal_nsamples

    def restore_rng(self, rng: np.random.Generator) -> None:
        """Restore ``rng`` without consuming a draw.

        A bit-generator mismatch is rejected instead of relying on NumPy's
        implementation-specific coercion between generator families.
        """
        expected = self.rng_state.get("bit_generator")
        actual = type(rng.bit_generator).__name__
        if expected != actual:
            raise NTCCheckpointConfigMismatchError(
                f"checkpoint RNG uses {expected!r}, not {actual!r}"
            )
        try:
            rng.bit_generator.state = _copy_tree(self.rng_state)
        except (TypeError, ValueError) as exc:
            raise NTCCheckpointCorruptionError(
                "checkpoint contains an invalid NumPy RNG state"
            ) from exc


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _plain_json(value: Any) -> Any:
    """Convert config/metadata values to deterministic JSON primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_json(asdict(value))
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("JSON mapping keys must be strings")
            out[key] = _plain_json(value[key])
        return out
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("config and metadata floats must be finite")
        return result
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def config_fingerprint(cfg: PhysicalCavityConfig) -> str:
    """Return the SHA-256 fingerprint of the complete physical config."""
    return hashlib.sha256(_canonical_json(_plain_json(cfg))).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


class _TreeEncoder:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.descriptors: dict[str, dict[str, Any]] = {}

    def encode(self, value: Any, path: str = "payload") -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("object arrays are forbidden in NTC checkpoints")
            storage_key = f"array_{len(self.arrays):06d}"
            array = np.ascontiguousarray(value)
            self.arrays[storage_key] = array
            self.descriptors[storage_key] = {
                "logical_name": path,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": _array_sha256(array),
            }
            return {"kind": "array", "storage_key": storage_key}
        if isinstance(value, Mapping):
            items: dict[str, Any] = {}
            for key in sorted(value):
                if not isinstance(key, str):
                    raise TypeError("checkpoint mapping keys must be strings")
                items[key] = self.encode(value[key], f"{path}.{key}")
            return {"kind": "mapping", "items": items}
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [
                    self.encode(item, f"{path}[{index}]")
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, list):
            return {
                "kind": "list",
                "items": [
                    self.encode(item, f"{path}[{index}]")
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, (np.bool_, bool)):
            return {"kind": "bool", "value": bool(value)}
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return {"kind": "int", "value": int(value)}
        if isinstance(value, (np.floating, float)):
            number = float(value)
            if np.isnan(number):
                encoded = "nan"
            elif np.isposinf(number):
                encoded = "+inf"
            elif np.isneginf(number):
                encoded = "-inf"
            else:
                encoded = number
            return {"kind": "float", "value": encoded}
        if value is None:
            return {"kind": "none"}
        if isinstance(value, str):
            return {"kind": "str", "value": value}
        raise TypeError(
            f"checkpoint value of type {type(value).__name__} is unsupported"
        )


def _decode_tree(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise NTCCheckpointCorruptionError("malformed checkpoint payload tree")
    kind = value["kind"]
    if kind == "array":
        key = value.get("storage_key")
        if not isinstance(key, str) or key not in arrays:
            raise NTCCheckpointCorruptionError("payload references a missing array")
        return arrays[key].copy()
    if kind == "mapping":
        items = value.get("items")
        if not isinstance(items, dict) or not all(
            isinstance(key, str) for key in items
        ):
            raise NTCCheckpointCorruptionError("malformed mapping payload")
        return {key: _decode_tree(items[key], arrays) for key in sorted(items)}
    if kind in {"list", "tuple"}:
        items = value.get("items")
        if not isinstance(items, list):
            raise NTCCheckpointCorruptionError("malformed sequence payload")
        decoded = [_decode_tree(item, arrays) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "bool" and isinstance(value.get("value"), bool):
        return value["value"]
    if kind == "int" and isinstance(value.get("value"), int):
        return value["value"]
    if kind == "float":
        encoded = value.get("value")
        if encoded == "nan":
            return float("nan")
        if encoded == "+inf":
            return float("inf")
        if encoded == "-inf":
            return float("-inf")
        if isinstance(encoded, (int, float)) and not isinstance(encoded, bool):
            return float(encoded)
    if kind == "none" and set(value) == {"kind"}:
        return None
    if kind == "str" and isinstance(value.get("value"), str):
        return value["value"]
    raise NTCCheckpointCorruptionError(f"malformed {kind!r} payload node")


def _copy_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    return value


def _validate_runtime_checkpoint(
    checkpoint: NTCCheckpoint,
    cfg: PhysicalCavityConfig,
) -> None:
    if checkpoint.step_index < 0:
        raise ValueError("step_index must be nonnegative")
    if checkpoint.temporal_nsamples < 0:
        raise ValueError("temporal_nsamples must be nonnegative")
    if checkpoint.moments.cfg != cfg or checkpoint.wall_events.cfg != cfg:
        raise ValueError("accumulator config differs from checkpoint config")
    if not isinstance(checkpoint.rng_state, Mapping):
        raise TypeError("rng_state must be a mapping")
    if not isinstance(checkpoint.rng_state.get("bit_generator"), str):
        raise ValueError("rng_state does not identify a NumPy bit generator")
    for key, value in checkpoint.diagnostics.items():
        if not isinstance(key, str) or not isinstance(
            value,
            (bool, int, float, np.bool_, np.integer, np.floating),
        ):
            raise TypeError("diagnostics must map strings to numeric scalars")
    if set(checkpoint.temporal_sums) != set(checkpoint.temporal_sums2):
        raise ValueError("temporal_sums and temporal_sums2 keys differ")
    for key in checkpoint.temporal_sums:
        first = np.asarray(checkpoint.temporal_sums[key])
        second = np.asarray(checkpoint.temporal_sums2[key])
        if first.shape != second.shape:
            raise ValueError(f"temporal accumulator shape mismatch for {key!r}")


def _checkpoint_payload(checkpoint: NTCCheckpoint) -> dict[str, Any]:
    moments = checkpoint.moments
    wall = checkpoint.wall_events
    return {
        "state": {
            "pos": checkpoint.state.pos,
            "vel": checkpoint.state.vel,
            "weight": checkpoint.state.weight,
        },
        "rng_state": checkpoint.rng_state,
        "step_index": checkpoint.step_index,
        "diagnostics": checkpoint.diagnostics,
        "moments": {
            "samples": moments.samples,
            "simulated_count": moments.simulated_count,
            "m0": moments.m0,
            "m1": moments.m1,
            "m2": moments.m2,
            "energy": moments.energy,
            "energy_velocity": moments.energy_velocity,
        },
        "wall_events": {
            "event_count": wall.event_count,
            "inverse_flux_weight": wall.inverse_flux_weight,
            "weighted_slip": wall.weighted_slip,
            "weighted_relative_speed2": wall.weighted_relative_speed2,
        },
        "temporal_sums": checkpoint.temporal_sums,
        "temporal_sums2": checkpoint.temporal_sums2,
        "temporal_nsamples": checkpoint.temporal_nsamples,
        "block_accumulators": checkpoint.block_accumulators,
    }


def save_ntc_checkpoint(
    path: str | Path,
    cfg: PhysicalCavityConfig,
    checkpoint: NTCCheckpoint,
) -> dict[str, Any]:
    """Atomically save a deterministic, pickle-free NTC checkpoint.

    Arrays are stored in NPZ records.  A canonical-JSON manifest embedded in
    the same archive records the config fingerprint, logical array names,
    shapes, dtypes, and SHA-256 digests.  No timestamp is included, so equal
    runtime states produce byte-identical files.
    """
    _validate_runtime_checkpoint(checkpoint, cfg)
    metadata = _plain_json(checkpoint.metadata)
    cfg_json = _plain_json(cfg)
    encoder = _TreeEncoder()
    payload_tree = encoder.encode(_checkpoint_payload(checkpoint))
    core_manifest: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": cfg_json,
        "config_fingerprint": config_fingerprint(cfg),
        "metadata": metadata,
        "payload": payload_tree,
        "arrays": encoder.descriptors,
    }
    manifest_sha256 = hashlib.sha256(_canonical_json(core_manifest)).hexdigest()
    manifest = dict(core_manifest)
    manifest["manifest_sha256"] = manifest_sha256
    manifest_bytes = _canonical_json(manifest)

    archive_arrays = dict(encoder.arrays)
    archive_arrays[_MANIFEST_KEY] = np.frombuffer(manifest_bytes, dtype=np.uint8)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                **{key: archive_arrays[key] for key in sorted(archive_arrays)},
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return manifest


def _load_archive(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if _MANIFEST_KEY not in archive.files:
                raise NTCCheckpointCorruptionError(
                    "checkpoint has no embedded manifest"
                )
            raw_manifest = np.asarray(archive[_MANIFEST_KEY])
            if raw_manifest.dtype != np.uint8 or raw_manifest.ndim != 1:
                raise NTCCheckpointCorruptionError(
                    "checkpoint manifest record has the wrong type"
                )
            try:
                manifest = json.loads(raw_manifest.tobytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NTCCheckpointCorruptionError(
                    "checkpoint manifest is not valid canonical JSON"
                ) from exc
            descriptors = manifest.get("arrays")
            if not isinstance(descriptors, dict):
                raise NTCCheckpointCorruptionError(
                    "checkpoint manifest has no array descriptors"
                )
            expected_keys = set(descriptors) | {_MANIFEST_KEY}
            if set(archive.files) != expected_keys:
                raise NTCCheckpointCorruptionError(
                    "checkpoint archive has missing or unexpected records"
                )
            arrays = {
                key: np.array(archive[key], copy=True)
                for key in sorted(descriptors)
            }
    except NTCCheckpointError:
        raise
    except (OSError, EOFError, ValueError) as exc:
        raise NTCCheckpointCorruptionError(
            f"cannot read checkpoint archive {path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise NTCCheckpointCorruptionError("checkpoint manifest must be an object")
    return manifest, arrays


def _validate_manifest(
    manifest: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
    cfg: PhysicalCavityConfig,
) -> None:
    if manifest.get("format") != CHECKPOINT_FORMAT:
        raise NTCCheckpointCorruptionError("unknown checkpoint format")
    if manifest.get("version") != CHECKPOINT_VERSION:
        raise NTCCheckpointCorruptionError("unsupported checkpoint version")
    recorded_manifest_hash = manifest.get("manifest_sha256")
    core_manifest = dict(manifest)
    core_manifest.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(
        _canonical_json(core_manifest)
    ).hexdigest()
    if recorded_manifest_hash != actual_manifest_hash:
        raise NTCCheckpointCorruptionError("checkpoint manifest hash mismatch")
    expected_fingerprint = config_fingerprint(cfg)
    if manifest.get("config_fingerprint") != expected_fingerprint:
        raise NTCCheckpointConfigMismatchError(
            "checkpoint config fingerprint does not match the requested config"
        )
    if manifest.get("config") != _plain_json(cfg):
        raise NTCCheckpointConfigMismatchError(
            "checkpoint config metadata does not match the requested config"
        )
    descriptors = manifest.get("arrays")
    assert isinstance(descriptors, dict)
    for key, descriptor in descriptors.items():
        if not isinstance(descriptor, dict) or key not in arrays:
            raise NTCCheckpointCorruptionError("malformed array descriptor")
        array = arrays[key]
        if descriptor.get("dtype") != array.dtype.str:
            raise NTCCheckpointCorruptionError(f"dtype mismatch for {key}")
        if descriptor.get("shape") != list(array.shape):
            raise NTCCheckpointCorruptionError(f"shape mismatch for {key}")
        if descriptor.get("sha256") != _array_sha256(array):
            raise NTCCheckpointCorruptionError(f"array hash mismatch for {key}")


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NTCCheckpointCorruptionError(f"{name} payload must be a mapping")
    return value


def _required_array(mapping: Mapping[str, Any], key: str) -> np.ndarray:
    value = mapping.get(key)
    if not isinstance(value, np.ndarray):
        raise NTCCheckpointCorruptionError(f"{key} payload must be an array")
    return value


def _validate_loaded_shapes(
    state: PhysicalParticleState,
    moments: PhysicalMomentAccumulator,
    wall: LidWallEventAccumulator,
    temporal_sums: Mapping[str, Any],
    temporal_sums2: Mapping[str, Any],
    cfg: PhysicalCavityConfig,
) -> None:
    count = len(state.pos)
    if state.pos.shape != (count, 2):
        raise NTCCheckpointCorruptionError("particle positions have invalid shape")
    if state.vel.shape != (count, 3) or state.weight.shape != (count,):
        raise NTCCheckpointCorruptionError("particle velocity/weight shape mismatch")
    ncell = cfg.nx * cfg.ny
    expected_moment_shapes = {
        "simulated_count": (ncell,),
        "m0": (ncell,),
        "m1": (ncell, 3),
        "m2": (ncell, 3, 3),
        "energy": (ncell,),
        "energy_velocity": (ncell, 3),
    }
    for name, shape in expected_moment_shapes.items():
        if getattr(moments, name).shape != shape:
            raise NTCCheckpointCorruptionError(
                f"moment accumulator {name} has invalid shape"
            )
    for name in (
        "event_count",
        "inverse_flux_weight",
        "weighted_slip",
        "weighted_relative_speed2",
    ):
        if getattr(wall, name).shape != (cfg.nx,):
            raise NTCCheckpointCorruptionError(
                f"wall accumulator {name} has invalid shape"
            )
    if set(temporal_sums) != set(temporal_sums2):
        raise NTCCheckpointCorruptionError("temporal accumulator keys differ")
    for key in temporal_sums:
        first = temporal_sums[key]
        second = temporal_sums2[key]
        if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
            raise NTCCheckpointCorruptionError(
                "temporal accumulator values must be arrays"
            )
        if first.shape != second.shape:
            raise NTCCheckpointCorruptionError(
                f"temporal accumulator shape mismatch for {key!r}"
            )


def load_ntc_checkpoint(
    path: str | Path,
    cfg: PhysicalCavityConfig,
) -> NTCCheckpoint:
    """Load and fully validate an NTC checkpoint before exposing its state."""
    source = Path(path)
    manifest, arrays = _load_archive(source)
    _validate_manifest(manifest, arrays, cfg)
    payload = _decode_tree(manifest.get("payload"), arrays)
    payload = _required_mapping(payload, "root")

    state_data = _required_mapping(payload.get("state"), "state")
    state = PhysicalParticleState(
        pos=_required_array(state_data, "pos"),
        vel=_required_array(state_data, "vel"),
        weight=_required_array(state_data, "weight"),
    )
    moment_data = _required_mapping(payload.get("moments"), "moments")
    moments = PhysicalMomentAccumulator(cfg)
    samples = moment_data.get("samples")
    if not isinstance(samples, int) or samples < 0:
        raise NTCCheckpointCorruptionError("invalid moment sample count")
    moments.samples = samples
    for name in (
        "simulated_count",
        "m0",
        "m1",
        "m2",
        "energy",
        "energy_velocity",
    ):
        setattr(moments, name, _required_array(moment_data, name))

    wall_data = _required_mapping(payload.get("wall_events"), "wall_events")
    wall = LidWallEventAccumulator(cfg)
    for name in (
        "event_count",
        "inverse_flux_weight",
        "weighted_slip",
        "weighted_relative_speed2",
    ):
        setattr(wall, name, _required_array(wall_data, name))

    rng_state = _required_mapping(payload.get("rng_state"), "rng_state")
    diagnostics = _required_mapping(payload.get("diagnostics"), "diagnostics")
    temporal_sums = _required_mapping(
        payload.get("temporal_sums"), "temporal_sums"
    )
    temporal_sums2 = _required_mapping(
        payload.get("temporal_sums2"), "temporal_sums2"
    )
    step_index = payload.get("step_index")
    temporal_nsamples = payload.get("temporal_nsamples")
    if not isinstance(step_index, int) or step_index < 0:
        raise NTCCheckpointCorruptionError("invalid checkpoint step index")
    if not isinstance(temporal_nsamples, int) or temporal_nsamples < 0:
        raise NTCCheckpointCorruptionError("invalid temporal sample count")
    block_accumulators = payload.get("block_accumulators")
    if block_accumulators is not None and not isinstance(
        block_accumulators, dict
    ):
        raise NTCCheckpointCorruptionError(
            "block_accumulators must be a mapping or null"
        )
    if not all(
        isinstance(key, str)
        and isinstance(value, (bool, int, float))
        and not isinstance(value, complex)
        for key, value in diagnostics.items()
    ):
        raise NTCCheckpointCorruptionError(
            "diagnostics must map strings to numeric scalars"
        )
    _validate_loaded_shapes(
        state,
        moments,
        wall,
        temporal_sums,
        temporal_sums2,
        cfg,
    )
    result = NTCCheckpoint(
        state=state,
        rng_state=rng_state,
        step_index=step_index,
        diagnostics=diagnostics,
        moments=moments,
        wall_events=wall,
        temporal_sums=temporal_sums,
        temporal_sums2=temporal_sums2,
        temporal_nsamples=temporal_nsamples,
        block_accumulators=block_accumulators,
        metadata=manifest.get("metadata", {}),
        manifest=manifest,
    )
    _validate_runtime_checkpoint(result, cfg)
    return result
