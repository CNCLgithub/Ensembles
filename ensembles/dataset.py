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
        n = len(self)
        scene = np.floor(idx/n).astype(int) + 1
        time = np.floor(idx % len(self)/self.manifest['n_dots']).astype(int) + 1
        item = idx % self.manifest['n_dots'] + 1
        obj_path = f"{self.src}/{scene}/serialized/{time}_{item}.json"
        img_path = f"{self.src}/{scene}/images/{time}_{item}.png"
        image = pil_loader(img_path)
        results = []
        with open(obj_path, 'r') as f:
            obj = json.load(f)
            k = 6 + len(obj['gstate'])
            result = np.zeros(k)
            result[0] = obj['radius']
            result[1] = obj['mass']
            result[2:4] = obj['pos']
            result[4:6] = obj['vel']
            gstate = np.asarray(obj['gstate']).flatten()
            #embedding = np.ndarray.flatten(results).astype(np.float32)
            result[6:] = gstate


        # results = np.asarray(results)
        # embedding = np.ndarray.flatten(results).astype(np.float32)
        #
        return result, image

def write_ffcv_data(d: ObjectDataset,
                    path: str,
                    emb_kwargs: dict,
                    img_kwargs: dict,
                    w_kwargs: dict) -> None:
    writer = DatasetWriter(path,
                           {'emb': NDArrayField(**emb_kwargs),
                            'image': RGBImageField(**img_kwargs)},
                           **w_kwargs)
    writer.from_indexed_dataset(d)

def img_pipeline(mu, sd) -> List[Operation]:
    return [SimpleRGBImageDecoder(),
            NormalizeImage(mu, sd, np.float32),
            ToTensor(),
            ToTorchImage(convert_back_int16 =False),
            Convert(torch.float32),
            ]

def object_pipeline() -> List[Operation]:
    return [NDArrayDecoder(),
            Convert(torch.float32),
            ToTensor()]

def object_loader(path: str, device,  **kwargs) -> Loader:
    with open(path + '_manifest.json', 'r') as f:
        manifest = json.load(f)
    mu = np.zeros(3)
    sd = np.array([255, 255, 255])
    l =  Loader(path + '.beton',
                pipelines= {'emb' : object_pipeline() +
                                      [ToDevice(device)],
                            'image': None},
                **kwargs)
    return l
