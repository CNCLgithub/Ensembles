import os
import json
import torch
import numpy as np
# from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets.folder import pil_loader
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.fields import RGBImageField, NDArrayField
from ffcv.pipeline.operation import Operation
from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
from ffcv.transforms import (Convert, NormalizeImage, ToTensor, ToTorchImage,
    ToDevice)

from . pytypes import *

class ObjectDataset(Dataset):
    # inheriting pytorch dataset; return vector of object and gstate
    def __init__(self, src: str):
        with open(src + '_manifest.json', 'r') as f:
            manifest = json.load(f)
        self.src = src
        self.manifest = manifest
        self.embedding_order = ["radius", "mass", "pos", "vel", "gstate"]
    #length function w parameter specified in manifest file
    def __len__(self):
        return self.manifest['trials'] * self.manifest['k'] * \
            self.manifest['n_dots']
    #loads a trial
    def __getitem__(self, idx):
        scene = np.floor(idx/len(self))
        time = np.floor(idx % len(self)/self.manifest['n_dots'])
        item = idx % self.manifest['n_dots']
        obj_path = os.path.join(self.src, str(idx+1),"serialized",
        f"{time}_{item}.json")
        img_path = os.path.join(self.src, str(idx+1),"images",
        f"{time}_{item}.png")
        image = pil_loader(img_path)
        results = []
        with open(obj_path, 'r') as f:
            obj = json.load(f)
            for i in self.embedding_order:
                results.append(obj[i])
        embedding = np.flatten(results)
        return embedding, image

def write_ffcv_data(d: OGVAEDataset,
                    path: str,
                    img_kwargs: dict,
                    og_kwargs: dict,
                    w_kwargs: dict) -> None:
    writer = DatasetWriter(path,
                           { 'image': RGBImageField(**img_kwargs),
                             'og': NDArrayField(**og_kwargs)},
                           **w_kwargs)
    writer.from_indexed_dataset(d)

def img_pipeline(mu, sd) -> List[Operation]OGVAE:
    return [SimpleRGBImageDecoder(),
            # NormalizeImage(mu, sd, np.float16),
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float32),
            ]

def og_pipeline() -> List[Operation]:
    return [NDArrayDecoder(),
            Convert(torch.float32),
            ToTensor()]

def ogvae_loader(path: str, device,  **kwargs) -> Loader:
    with open(path + '_manifest.json', 'r') as f:
        manifest = json.load(f)

    l =  Loader(path + '.beton',
                pipelines= {'image' : img_pipeline(manifest['img_mu'],
                                                   manifest['img_sd']) +
                                      [ToDevice(device)],
                            'og': None},
                **kwargs)
    return l

def ogdecoder_loader(path: str , **kwargs) -> Loader:
    with open(path + '_manifest.json', 'r') as f:
        manifest = json.load(f)

    l =  Loader(path + '.beton',
                pipelines= {'image' : img_pipeline(manifest['img_mu'],
                                                   manifest['img_sd']),
                            'og'    : og_pipeline()},
                **kwargs)
    return l