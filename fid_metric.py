import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Encoders import ClipModel
from utils.seed import set_seed
from utils.train import parallel_load_images
from utils.image_utils import list_image_files


def name_path(pair):
    name, path = pair.split(',')
    return name, Path(path)


@torch.inference_mode()
def compute_fid_datasets(datasets, target='celeba', device=torch.device('cuda'), CLIP=False, seed=3407):
    set_seed(seed)
    result = {}

    if CLIP:
        fid = FrechetInceptionDistance(feature=ClipModel(), reset_real_features=False, normalize=False)
    else:
        fid = FrechetInceptionDistance(reset_real_features=False, normalize=False)
    fid.to(device).eval()

    real_dataloader = DataLoader(TensorDataset(datasets[target]), batch_size=32)
    for batch in tqdm(real_dataloader):
        batch = batch[0].to(device)
        fid.update(batch, real=True)

    for key, tensor in datasets.items():
        if key == target:
            continue
        fid.reset()

        fake_dataloader = DataLoader(TensorDataset(tensor), batch_size=32)
        for batch in tqdm(fake_dataloader):
            batch = batch[0].to(device)
            fid.update(batch, real=False)
        result[key] = fid.compute().item()
    return result


def main():
    datasets = {}
    print('start')
    # source = 'datasets/FFHQ_TrueScale'
    datasets['FFHQ_TrueScale'] = parallel_load_images('datasets/FFHQ_TrueScale', list_image_files('datasets/FFHQ_TrueScale'))

    # for method, path_dataset in args.methods_dataset:
    datasets['Same_Scale'] = parallel_load_images('test_outputs', list_image_files('test_outputs'))
    datasets['Hair_DownScale'] = parallel_load_images('test_outputs_dif_hair', list_image_files('test_outputs_dif_hair'))
    datasets['Face_DownScale'] = parallel_load_images('test_outputs_dif_face', list_image_files('test_outputs_dif_face'))
    datasets['Hair_Resized'] = parallel_load_images('test_outputs_FaceScale', list_image_files('test_outputs_FaceScale'))

    print('Loading')

    FIDs = compute_fid_datasets(datasets, target='FFHQ_TrueScale', CLIP=False)
    df_fid = pd.DataFrame.from_dict(FIDs, orient='index', columns=['FID'])

    FIDs_CLIP = compute_fid_datasets(datasets, target='FFHQ_TrueScale', CLIP=True)
    df_clip = pd.DataFrame.from_dict(FIDs_CLIP, orient='index', columns=['FID_CLIP'])

    df_result = pd.concat([df_fid, df_clip], axis=1).round(2)
    print(df_result)

    # os.makedirs('log', exist_ok=True)
    # df_result.to_csv('metric.csv', index=True)


if __name__ == '__main__':
    
    # exit()
    parser = argparse.ArgumentParser(description='Compute metrics')
    parser.add_argument('--source_dataset', default=None, type=Path, help='Dataset with real faces')
    parser.add_argument('--methods_dataset', default=None, type=name_path, nargs='+', help='Datasets after applying the method')
    parser.add_argument('--output', type=Path, default='logs/metric.csv', help='Folder for saving logs')
    args = parser.parse_args()

    main()
