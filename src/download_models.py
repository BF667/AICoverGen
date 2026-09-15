
from pathlib import Path
import shutil
import requests

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'

BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'

mdxnet_models_dir.mkdir(parents=True, exist_ok=True)


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



if __name__ == '__main__':
    mdx_model_names = ['UVR-MDX-NET-Inst_HQ_4.onnx', 'UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    
