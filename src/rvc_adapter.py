"""
AICoverGen RVC adapter for the uziproj/rvc package.

This module preserves AICoverGen's original RVC entry points
(`Config`, `load_hubert`, `get_vc`, `rvc_infer`) but delegates all the
heavy lifting to the modern, pip-installed `rvc` package from
https://github.com/uziproj/rvc.

The legacy in-repo RVC stack (`vc_infer_pipeline.py`, `infer_pack/`,
`rmvpe.py`, `configs/*.json`, `trainset_preprocess_pipeline_print.py`)
has been moved to `src/_legacy_rvc/` and is no longer imported at
runtime. It is kept only as a reference for the historical
implementation.

The uziproj/rvc package:
  * loads its own embedder (contentvec / hubert) inside `Config`
  * loads its own RMVPE predictor inside `Config`
  * exposes a clean `RVClass(config, pth_path).run(...)` API
  * supports 20+ F0 methods (pm, dio, harvest, yin, pyin, swipe,
    rmvpe, rmvpe-legacy, fcpe, fcpe-legacy, djcm, crepe-*,
    mangio-crepe-*, hybrid[...])

This adapter:
  * Translates AICoverGen's `Config(device, is_half)` call signature
    into uziproj/rvc's `Config(cpu_mode=..., is_half=...)` signature.
  * Provides a no-op `load_hubert` for backwards compatibility
    (AICoverGen's `main.py` still calls it at module load time).
  * `get_vc` returns a `(cpt, version, net_g, tgt_sr, rvc_instance)`
    tuple where `rvc_instance` is a `RVClass` ready to call `.run()`
    on. The first four tuple members are kept only for backwards
    compatibility with `main.py`'s `voice_change()` and are NOT used
    by `rvc_infer` below.
  * `rvc_infer` calls `RVClass.run()` with the same parameter set
    AICoverGen historically exposed (pitch, f0_method, index_rate,
    filter_radius, rms_mix_rate, protect, crepe_hop_length). The
    `steps` parameter preserves the multi-pass behaviour of the
    original implementation: each pass reuses the previous pass's
    output as input.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Tuple

import torch

# Import the modern RVC package.
# `rvc` is installed via `pip install git+https://github.com/uziproj/rvc.git`
# (or `pip install -e .` from a local clone).
from rvc import Config as _RVCConfig, RVClass as _RVClass
from rvc import F0_METHODS as _F0_METHODS

BASE_DIR = Path(__file__).resolve().parent.parent

# AICoverGen historically stored embedder + rmvpe under `rvc_models/`,
# while the uziproj/rvc package looks for them under `assets/models/`.
# `download_models.py` (updated in this same adaptation) is responsible
# for provisioning `assets/models/` — either by downloading directly
# or by symlinking the files AICoverGen already downloads.
ASSETS_MODELS_DIR = BASE_DIR / "assets" / "models"
RVC_MODELS_DIR = BASE_DIR / "rvc_models"


# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------
__all__ = [
    "Config",
    "load_hubert",
    "get_vc",
    "rvc_infer",
    "RVClass",
    "F0_METHODS",
]


class Config:
    """Adapter that maps AICoverGen's `Config(device, is_half)` call
    signature onto uziproj/rvc's `Config(cpu_mode=..., is_half=...)`.

    AICoverGen historically determined the device itself ("cuda:0" /
    "cpu" / "mps") and passed it positionally. The uziproj/rvc package
    figures out the device internally; we translate by setting
    `cpu_mode=True` whenever the requested device is "cpu" or whenever
    CUDA is unavailable.

    The underlying uziproj/rvc `Config` is a singleton, so multiple
    instantiations of this adapter with the same args will return
    the same internal state.
    """

    def __init__(
        self,
        device: str = "cpu",
        is_half: bool = False,
        f0_method: str = "rmvpe",
        embedder_model: str = "hubert_base",
        log_level: str = "info",
    ):
        # Translate the device string. uziproj/rvc's Config determines
        # the device itself; we just tell it whether to force CPU mode.
        requested_cpu = str(device).lower() in ("cpu", "mps")
        cuda_unavailable = not torch.cuda.is_available()
        cpu_mode = requested_cpu or cuda_unavailable

        # If the user wants CUDA but it's unavailable, warn but continue.
        if not cpu_mode and cuda_unavailable:
            print(
                "[AICoverGen/rvc-adapter] CUDA requested but unavailable; "
                "falling back to CPU."
            )
            cpu_mode = True

        # Delegate to uziproj/rvc Config (singleton — first call wins).
        self._rvc_config: Any = _RVCConfig(
            cpu_mode=cpu_mode,
            is_half=is_half and not cpu_mode,  # half-precision only makes
                                               # sense on GPU
            embedder_model=embedder_model,
            f0_method=f0_method,
            log_level=log_level,
        )

        # Expose the attributes AICoverGen's main.py reads.
        self.device = self._rvc_config.device
        self.is_half = self._rvc_config.is_half
        self.gpu_name = getattr(self._rvc_config, "gpu_name", None)
        self.gpu_mem = getattr(self._rvc_config, "gpu_mem", None)
        # The old Config exposed x_pad/x_query/x_center/x_max — these
        # are no longer public on the new Config (they live inside the
        # Pipeline class). Expose them via device_config() for any
        # downstream code that still reads them.
        try:
            (
                self.x_pad,
                self.x_query,
                self.x_center,
                self.x_max,
            ) = self._rvc_config.device_config()
        except Exception:
            # Fall back to half-precision defaults.
            self.x_pad, self.x_query, self.x_center, self.x_max = (
                (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
            )

    # -- helpers ----------------------------------------------------------
    @property
    def rvc_config(self) -> Any:
        """Return the underlying uziproj/rvc Config instance."""
        return self._rvc_config

    def set_log_level(self, level: str) -> None:
        """Change the log level at runtime."""
        self._rvc_config.set_log_level(level)


def load_hubert(
    device: str = "cpu",
    is_half: bool = False,
    model_path: str | None = None,
):
    """Backwards-compatible no-op.

    AICoverGen's `main.py` calls `load_hubert(...)` at module load
    time and stores the result in a module-level `hubert_model`
    global. The uziproj/rvc package loads its own embedder
    (contentvec / hubert) inside `Config.__init__`, so this function
    is no longer required — it returns `None` and the embedder is
    fetched lazily from the `Config` singleton when `RVClass` is
    instantiated.

    The `model_path` argument is ignored: uziproj/rvc pulls the
    embedder from `assets/models/{embedder_model}.pt` (set on
    `Config`), so the AICoverGen-downloaded `rvc_models/hubert_base.pt`
    is no longer used directly. `download_models.py` (updated in
    this adaptation) takes care of provisioning `assets/models/`.
    """
    return None


def get_vc(
    device: str,
    is_half: bool,
    config: Config | Any,
    model_path: str,
) -> Tuple[Any, str, Any, int, _RVClass]:
    """Load an RVC voice model.

    Returns the 5-tuple `(cpt, version, net_g, tgt_sr, rvc_instance)`
    that AICoverGen's `main.py::voice_change()` expects. The first
    four members are kept only for backwards compatibility — they
    are pulled from the underlying `VoiceConverter` so that any code
    that still inspects them works — but `rvc_infer` below no longer
    uses them; it operates purely on the returned `RVClass` instance.

    Args:
        device: device string ("cuda:0" / "cpu"). Used only for
            logging — actual device selection happens inside the
            uziproj/rvc Config singleton.
        is_half: whether to use half-precision. Used only for
            logging — actual half-precision selection happens inside
            the uziproj/rvc Config singleton.
        config: a `Config` adapter instance (or a raw uziproj/rvc
            `Config` instance — both are supported).
        model_path: absolute path to the `.pth` voice model file.
    """
    # Resolve the underlying uziproj/rvc Config from the adapter.
    rvc_config = (
        config.rvc_config if isinstance(config, Config) else config
    )

    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[AICoverGen/rvc-adapter] Voice model not found: {model_path!r}"
        )

    # RVClass loads the model once and reuses it across multiple
    # .run() calls — exactly what AICoverGen's voice_change() wants.
    rvc_instance = _RVClass(config=rvc_config, pth_path=model_path)

    # Pull metadata for backwards compatibility with main.py. None of
    # these are used by rvc_infer below; they exist so that any code
    # reading `cpt`, `version`, `net_g`, or `tgt_sr` after get_vc()
    # still works.
    converter = rvc_instance._converter  # the VoiceConverter instance
    cpt = converter.cpt
    version = converter.version or "v2"
    net_g = converter.net_g
    tgt_sr = converter.tgt_sr or 48000

    return cpt, version, net_g, tgt_sr, rvc_instance


def rvc_infer(
    index_path: str,
    index_rate: float,
    input_path: str,
    output_path: str,
    pitch_change: int,
    f0_method: str,
    cpt: Any,
    version: str,
    net_g: Any,
    filter_radius: int,
    tgt_sr: int,
    rms_mix_rate: float,
    protect: float,
    crepe_hop_length: int,
    vc: _RVClass,
    hubert_model: Any,
    steps: int,
) -> None:
    """Run RVC inference.

    Args mirror AICoverGen's historical signature. The following
    positional args are accepted for backwards compatibility but
    are NOT used by this adapter (they live inside the `RVClass`
    instance `vc`):
        cpt, version, net_g, tgt_sr, hubert_model

    The actually-used args are:
        index_path: path to the `.index` file (may be "" → ignored)
        index_rate: feature retrieval ratio (0.0-1.0)
        input_path: input audio file (wav/mp3/flac/...)
        output_path: output audio file (.wav)
        pitch_change: pitch shift in semitones
        f0_method: F0 extraction method (see F0_METHODS)
        filter_radius: median filter radius for pitch (0-7)
        rms_mix_rate: how much of the original vocal's loudness to
            preserve (0.0 = full fixed, 1.0 = full original)
        protect: protect voiceless consonants (0.0-0.5, 0.5 disables)
        crepe_hop_length: hop length for crepe/mangio-crepe methods
        vc: a RVClass instance returned by `get_vc`
        steps: number of inference passes (multi-pass behaviour —
            each pass uses the previous pass's output as input)
    """
    if not isinstance(vc, _RVClass):
        raise TypeError(
            "rvc_infer() expects `vc` to be a RVClass instance as "
            "returned by get_vc(). Got: "
            f"{type(vc).__name__}"
        )

    # Normalise the index path: empty string → None.
    index_path_arg = index_path.strip() if isinstance(index_path, str) else None
    if not index_path_arg:
        index_path_arg = None

    # Coerce empty/None f0_method → "rmvpe" (the modern default, also
    # the historical AICoverGen default). Then translate legacy names
    # like 'rmvpe+', 'crepe', 'mangio-crepe' onto the uziproj/rvc
    # package's canonical names.
    f0_method_arg = _normalize_f0_method(f0_method or "rmvpe")

    # Multi-pass behaviour: each pass reuses the previous pass's
    # output as input. Matches the historical rvc_infer() semantics.
    working_path = input_path
    for _step in range(max(1, int(steps))):
        vc.run(
            input_path=working_path,
            output_path=output_path,
            pitch=pitch_change,
            f0_method=f0_method_arg,
            filter_radius=filter_radius,
            index_rate=index_rate,
            index_path=index_path_arg,
            # rms_mix_rate in AICoverGen == volume_envelope in uziproj/rvc
            volume_envelope=rms_mix_rate,
            protect=protect,
            # crepe_hop_length in AICoverGen == hop_length in uziproj/rvc
            hop_length=crepe_hop_length,
            # Match AICoverGen's historical output: 16k → tgt_sr wav
            export_format="wav",
            # Don't resample — let RVClass use the model's native tgt_sr
            resample_sr=0,
        )
        working_path = output_path

    # Be polite about GPU memory — matches the historical `del cpt`
    # in main.py::voice_change().
    del vc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Re-export the most useful symbols for downstream code that prefers
# the modern API directly.
RVClass = _RVClass
F0_METHODS = _F0_METHODS


# ---------------------------------------------------------------------------
# F0 method name normalization
# ---------------------------------------------------------------------------
# AICoverGen historically exposed three f0 methods in its webui:
#   * 'rmvpe+'   — RMVPE with pitch-based (min/max-clamped) inference.
#                  The uziproj/rvc package does not have a direct
#                  equivalent of the '+' variant; we map it to plain
#                  'rmvpe' (which is still the high-quality RMVPE
#                  U-Net predictor).
#   * 'rmvpe'    — supported directly by uziproj/rvc.
#   * 'mangio-crepe' — CREPE (Mangio fork) without a size suffix.
#                  uziproj/rvc requires a size suffix
#                  (crepe-tiny / -small / -medium / -large / -full);
#                  we map the bare 'mangio-crepe' to 'mangio-crepe-large'
#                  to match the historical behaviour (the original
#                  torchcrepe.predict() default was the 'full' model,
#                  but 'large' is the closest equivalent in the new
#                  package's naming).
#
# Other historical method names that work directly with the new
# package: 'pm', 'dio', 'harvest', 'crepe-tiny', 'mangio-crepe-tiny',
# 'rmvpe', 'hybrid[...]'.
#
# New methods the new package adds that AICoverGen's webui doesn't
# (yet) expose: 'yin', 'pyin', 'swipe', 'fcpe', 'fcpe-legacy',
# 'djcm', 'crepe-small', 'crepe-medium', 'crepe-large', 'crepe-full',
# 'mangio-crepe-small', 'mangio-crepe-medium', 'mangio-crepe-full'.
_F0_METHOD_ALIASES = {
    "rmvpe+": "rmvpe",
    "crepe": "crepe-large",
    "mangio-crepe": "mangio-crepe-large",
}


def _normalize_f0_method(method: str) -> str:
    """Translate AICoverGen's historical f0 method names onto the
    names expected by the uziproj/rvc package.

    Unknown / already-canonical names are returned unchanged so that
    users who switch to the new naming (e.g. 'fcpe', 'crepe-full',
    'hybrid[rmvpe+fcpe]') work without translation.
    """
    if not method:
        return "rmvpe"
    return _F0_METHOD_ALIASES.get(method, method)


if __name__ == "__main__":
    # Quick self-test: print the adapter's surface.
    print("[AICoverGen/rvc_adapter] Public symbols:")
    print("  Config      :", Config)
    print("  load_hubert :", load_hubert)
    print("  get_vc      :", get_vc)
    print("  rvc_infer   :", rvc_infer)
    print("  RVClass     :", RVClass)
    print("  F0_METHODS  :", F0_METHODS)
