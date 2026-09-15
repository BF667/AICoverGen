"""
AICoverGen model downloader.

Downloads:
  * MDX separation models (UVR-MDX-NET-*) into `mdxnet_models/`
  * Hubert embedder (`hubert_base.pt`) and RMVPE predictor
    (`rmvpe.pt`) into `rvc_models/` (kept for backwards compat)
    AND into `assets/models/` (where the modern uziproj/rvc
    package looks for them)

If a model already exists in either location, the download is
skipped. If `rvc_models/{hubert_base,rmvpe}.pt` already exists but
`assets/models/{hubert_base,rmvpe}.pt` does not, the file is copied
(rather than re-downloaded) to avoid duplicate network traffic.
"""
from pathlib import Path
import shutil
import requests

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'
rvc_models_dir = BASE_DIR / 'rvc_models'
# uziproj/rvc looks for embedders + predictors under {cwd}/assets/models
assets_models_dir = BASE_DIR / 'assets' / 'models'

mdxnet_models_dir.mkdir(parents=True, exist_ok=True)
rvc_models_dir.mkdir(parents=True, exist_ok=True)
assets_models_dir.mkdir(parents=True, exist_ok=True)


def dl_model(link, model_name, dir_name):
    model_path = dir_name / model_name
    if model_path.exists():
        # print(f"{model_name} already exists, skipping download.")
        return

    print(f"Downloading {model_name}...")
    with requests.get(f'{link}{model_name}', stream=True) as r:
        r.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def provision_assets_model(model_name: str) -> None:
    """Make sure `assets/models/{model_name}` exists.

    The uziproj/rvc package looks for embedders + predictors under
    `assets/models/`. AICoverGen historically downloaded
    `hubert_base.pt` and `rmvpe.pt` to `rvc_models/`. To avoid a
    duplicate download, we copy from `rvc_models/` to `assets/models/`
    if the source file is present and the destination is missing.

    If neither location has the file, `dl_model()` is called to
    download it fresh into `assets/models/`.
    """
    dest = assets_models_dir / model_name
    if dest.exists():
        return

    src = rvc_models_dir / model_name
    if src.exists():
        print(f"Copying {model_name} from rvc_models/ to assets/models/")
        shutil.copy2(src, dest)
    else:
        # Fall back to a fresh download from the same HuggingFace
        # source AICoverGen historically used.
        dl_model(RVC_DOWNLOAD_LINK, model_name, assets_models_dir)


if __name__ == '__main__':
    mdx_model_names = ['UVR-MDX-NET-Inst_HQ_4.onnx', 'UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    rvc_model_names = ['hubert_base.pt', 'rmvpe.pt']
    # First download into rvc_models/ (kept for backwards compatibility
    # with any code that still references the old location).
    for model in rvc_model_names:
        dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)

    # Then provision assets/models/ for the uziproj/rvc package by
    # copying from rvc_models/ (or downloading fresh if missing).
    for model in rvc_model_names:
        provision_assets_model(model)

    # NOTE: `hubert_base.pt` and `contentvec_base.pt` are DIFFERENT
    # models in the uziproj/rvc package's view. We do NOT alias one
    # to the other. The adapter in `src/rvc.py` defaults to
    # `embedder_model="hubert_base"` so that the file AICoverGen
    # already downloads is the one that gets used. If the user
    # switches to `contentvec_base` (a different, slightly higher-
    # quality embedder), the uziproj/rvc package will download
    # `contentvec_base.pt` itself on first use from its own
    # HuggingFace source.

    print('All models ready!')
