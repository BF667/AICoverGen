
from pathlib import Path
import shutil
import requests

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'

BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'

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

    
