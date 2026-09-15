# Adaptation: AICoverGen RVC → uziproj/rvc

This document describes the changes made to AICoverGen to adapt its
RVC inference method to use the modern **uziproj/rvc** package
(<https://github.com/uziproj/rvc>) instead of the legacy in-repo
RVC stack.

## Summary of changes

| Area                | Before                                              | After                                                                                  |
|---------------------|-----------------------------------------------------|----------------------------------------------------------------------------------------|
| RVC inference stack | In-repo: `vc_infer_pipeline.py`, `infer_pack/`, `rmvpe.py`, `configs/*.json`, `trainset_preprocess_pipeline_print.py`, `my_utils.py` | Backed up under `src/_legacy_rvc/` (no longer imported at runtime)               |
| RVC entry point     | `src/rvc.py` exposed `Config`, `load_hubert`, `get_vc`, `rvc_infer` | `src/rvc_adapter.py` exposes the same 4 names but delegates all heavy lifting to `uziproj/rvc` |
| `main.py` import    | `from rvc import Config, load_hubert, get_vc, rvc_infer` | `from rvc_adapter import Config, load_hubert, get_vc, rvc_infer` (one-line change) |
| `download_models.py` | Downloaded `hubert_base.pt` and `rmvpe.pt` into `rvc_models/` only | Same downloads + provisions `assets/models/` (where the uziproj/rvc package looks for embedders + predictors). Existing files in `rvc_models/` are reused (copied) — no duplicate downloads. |
| `requirements.txt`  | Pinned to old `torch==2.5.1`, `fairseq==0.12.2`, `librosa==0.9.1`, `numpy==1.23.5`, etc. | Adds `rvc @ git+https://github.com/uziproj/rvc.git` (which pulls its own modern torch / faiss / librosa / praat-parselmouth / soundfile / numpy / scipy / transformers). Removed the now-redundant RVC-specific pins. AICoverGen-specific tools (gradio, pedalboard, pydub, sox) remain pinned. |

## File-by-file changes

### `src/rvc_adapter.py` (NEW — replaces `src/rvc.py`)

The new adapter exposes the same four public names AICoverGen's
`main.py` historically imported:

- **`Config(device, is_half, ...)`** — translates AICoverGen's
  historical call signature (positional `device`, `is_half`) onto
  the uziproj/rvc `Config(cpu_mode=..., is_half=..., f0_method=...,
  embedder_model=..., log_level=...)` signature. The underlying
  uziproj/rvc `Config` is a singleton — first call wins, subsequent
  calls return the same internal state. The adapter also exposes
  `.device`, `.is_half`, `.gpu_name`, `.gpu_mem`, and the historical
  `.x_pad / .x_query / .x_center / .x_max` tuple (pulled from
  `Config.device_config()`).

- **`load_hubert(device, is_half, model_path)`** — backwards-
  compatible no-op. Returns `None`. AICoverGen's `main.py` calls this
  at module load and stores the result in a module-level
  `hubert_model` global; the uziproj/rvc package loads its own
  embedder inside `Config.__init__`, so this function is no longer
  required. `rvc_infer` ignores the `hubert_model` argument it
  receives.

- **`get_vc(device, is_half, config, model_path)`** — returns a
  5-tuple `(cpt, version, net_g, tgt_sr, rvc_instance)` matching
  AICoverGen's historical contract. The first four members are kept
  only for backwards compatibility (pulled from the underlying
  `VoiceConverter`); `rvc_infer` doesn't use them. `rvc_instance`
  is a `RVClass` ready to call `.run()` on.

- **`rvc_infer(...)`** — calls `RVClass.run()` with the same
  parameter set AICoverGen historically exposed. The `steps`
  parameter preserves the multi-pass behaviour (each pass reuses
  the previous pass's output as input).

Additionally:

- **`RVClass`** and **`F0_METHODS`** are re-exported for downstream
  code that prefers the modern API directly.
- **`_normalize_f0_method(method)`** translates AICoverGen's
  legacy f0 method names onto the uziproj/rvc package's canonical
  names:

  | AICoverGen legacy | uziproj/rvc canonical | Reason                                   |
  |-------------------|-----------------------|------------------------------------------|
  | `rmvpe+`          | `rmvpe`               | No direct equivalent of the '+' variant; `rmvpe` is the same high-quality RMVPE U-Net predictor. |
  | `crepe`           | `crepe-large`         | The new package requires a size suffix; `large` matches the historical torchcrepe default. |
  | `mangio-crepe`    | `mangio-crepe-large`  | Same — requires a size suffix. |

  Other historical names (`pm`, `dio`, `harvest`, `crepe-tiny`,
  `mangio-crepe-tiny`, `rmvpe`, `hybrid[...]`) work directly
  without translation.

### `src/main.py` (one-line change)

```diff
-from rvc import Config, load_hubert, get_vc, rvc_infer
+from rvc_adapter import Config, load_hubert, get_vc, rvc_infer
```

That is the **only** change to `main.py`. The `voice_change()`
function continues to work unchanged because the adapter preserves
the exact API contract:

```python
cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)
rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method,
          cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect,
          crepe_hop_length, vc, hubert_model, steps)
```

### `src/download_models.py` (extended)

In addition to the historical downloads (MDX models into
`mdxnet_models/`; `hubert_base.pt` and `rmvpe.pt` into
`rvc_models/`), the updated script:

1. Creates `assets/models/` (where the uziproj/rvc package looks
   for embedders + predictors).
2. Calls a new `provision_assets_model(model_name)` helper for each
   of `hubert_base.pt` and `rmvpe.pt` — if the file already exists
   in `rvc_models/`, it is **copied** (not re-downloaded) into
   `assets/models/`. Only if neither location has the file does it
   fall back to a fresh download.

### `requirements.txt` (rewritten)

- **Added**: `rvc @ git+https://github.com/uziproj/rvc.git` (pulls
  modern torch / torchaudio / faiss-cpu / librosa /
  praat-parselmouth / soundfile / numpy / scipy / transformers /
  onnxruntime / einops / omegaconf / tqdm / matplotlib / requests /
  ffmpy / ffmpeg-python / yt-dlp / gradio / aiohttp / cuda-bindings
  / safetensors / tokenizers).
- **Removed**: explicit pins for `torch`, `torchvision`,
  `torchaudio`, `fairseq`, `faiss-cpu`, `librosa`, `numpy`,
  `onnxruntime-gpu`, `praat-parselmouth`, `pyworld`, `scipy`,
  `soundfile`, `torchcrepe` (all now provided transitively by the
  `rvc` package).
- **Kept**: AICoverGen-specific tools (`deemix`, `ffmpeg-python`,
  `gradio==5.44.0`, `lib==4.0.0`, `pedalboard==0.7.7`,
  `pydub==0.25.1`, `Requests>=2.31.0`, `sox==1.4.1`, `noisereduce`,
  `spaces`, `matplotlib-inline`, `yt_dlp`, `tqdm>=4.65.0`).

### `src/_legacy_rvc/` (NEW directory)

The following legacy files were moved here (no longer imported at
runtime; kept only as historical reference):

- `vc_infer_pipeline.py` (the old `VC` class with `pipeline`/`vc`/
  `get_f0` methods — ~660 lines)
- `rmvpe.py` (the old RMVPE predictor wrapper)
- `trainset_preprocess_pipeline_print.py` (training-set
  pre-processing — not used by inference, but kept here since
  AICoverGen used to ship it)
- `my_utils.py` (the old `load_audio` helper using ffmpeg
  subprocess — replaced by uziproj/rvc's `rvc.utils.load_audio`
  which uses soundfile + librosa)
- `infer_pack/` (the old RVC v1/v2 model definitions —
  `SynthesizerTrnMs256NSFsid`, `SynthesizerTrnMs768NSFsid`, etc.
  Replaced by uziproj/rvc's `rvc.lib.algorithm.synthesizers.
  Synthesizer`)
- `configs/` (the old `32k.json`, `40k.json`, `48k.json`,
  `48k_v2.json`, `32k_v2.json` config files — uziproj/rvc reads
  model config directly from the `.pth` checkpoint, so these are
  no longer needed)

## Verifying the adaptation

A smoke test is provided at
`/home/z/my-project/scripts/verify_adaptation.py` that verifies:

1. The adapter imports cleanly (the uziproj/rvc package must be
   installed).
2. The four public symbols AICoverGen's `main.py` imports are
   present and have the right types.
3. The `Config` adapter accepts AICoverGen's historical call
   signature `Config(device, is_half)`.
4. `load_hubert` is a no-op returning `None`.
5. Legacy f0 method aliases (`rmvpe+`, `mangio-crepe`, `crepe`)
   normalize correctly to canonical uziproj/rvc names.
6. `get_vc` raises a clean `FileNotFoundError` on a missing model.

Run it with:

```bash
cd /home/z/my-project/AICoverGen
python /home/z/my-project/scripts/verify_adaptation.py
```

## Running a real conversion

After the smoke test passes:

1. **Provision the models** (downloads MDX + RVC models and
   provisions `assets/models/` for the uziproj/rvc package):

   ```bash
   cd /home/z/my-project/AICoverGen
   python src/download_models.py
   ```

2. **Place a voice model** (a `.pth` file, optionally with an
   `.index` file) under `rvc_models/<voice_model_name>/`.

3. **CLI conversion**:

   ```bash
   python src/main.py -i song.mp3 -dir <voice_model_name> -p 0
   ```

4. **WebUI**:

   ```bash
   python app.py
   ```

## Notes on behavioural differences

- **`rmvpe+` is no longer a distinct mode**. The historical `+`
  variant used RMVPE with `pitch_based_audio_inference()` (min/max-
  clamped). The uziproj/rvc package's `rmvpe` always uses the
  standard RMVPE inference; the clamping is handled internally via
  the `proposal_pitch` feature. If you need the historical `+`
  behaviour, pass `-f0 rmvpe` and the conversion quality should be
  equivalent.

- **`crepe` / `mangio-crepe` without a size suffix** are normalized
  to `crepe-large` / `mangio-crepe-large`. If you want a different
  size, switch to the new explicit naming (`crepe-tiny`,
  `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full` — and
  the same set with the `mangio-` prefix). The webui dropdown can be
  extended to expose these without further code changes.

- **Multi-pass (`steps`) behaviour is preserved**: each pass
  reuses the previous pass's output as input, exactly as the
  historical `rvc_infer` did.

- **`hubert_base.pt` (AICoverGen's default embedder) is used
  directly** by the adapter (it sets
  `embedder_model="hubert_base"` in the underlying Config). If you
  want to switch to `contentvec_base` (a slightly higher-quality
  embedder supported by uziproj/rvc), the package will download
  `contentvec_base.pt` into `assets/models/` on first use.
