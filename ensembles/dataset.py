import os
import json
import torch
import numpy as np
from PIL import Image
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
        nk = self.manifest['k'] * self.manifest['n_dots']
        time = np.floor((idx % nk)/self.manifest['n_dots']).astype(int) + 1
        item = idx % self.manifest['n_dots'] + 1
        obj_path = f"{self.src}/{scene}/serialized/{time}_{item}.json"
        img_path = f"{self.src}/{scene}/images/{time}_{item}.png"
        # gstate = np.asarray(pil_loader(img_path))
        gstate = np.asarray(Image.open(img_path).convert('L')).astype('float32')/255.0
        #print(np.squeeze(gstate, axis=2).shape)
        with open(obj_path, 'r') as f:
            obj = json.load(f)

        #k = 2 (size and mass) + (4(position, velocity) * 5)
        k = 20
        dk_embedding = []
        #dk_embedding[0] = obj['radius']
        #dk_embedding[1] = obj['mass']
        for i in range(time-5,time):
            obj_path = f"{self.src}/{scene}/serialized/{i+1}_{item}.json"
            if i<0:
                obj_i = obj
            else:
                with open(obj_path, 'r') as f:
                    obj_i = json.load(f)

            dk_embedding.append(np.asarray(obj_i['pos'])/300.0)
            dk_embedding.append(np.asarray(obj_i['vel'])/100.0)


        #for loop time -5 : time, if time <0, just copy time t
        #dk_embedding[2:4] = np.asarray(obj['pos'])/300.0
        #dk_embedding[4:6] = np.asarray(obj['vel'])/100.0
            #old_gstate = np.array(obj['gstate']).astype(np.float32)
        dk_embedding = np.asarray(dk_embedding)
        dk_embedding = np.ndarray.flatten(dk_embedding).astype(np.float32)
        return dk_embedding, gstate

    #    return dk_embedding, old_gstate, gstate

def write_ffcv_data(d: ObjectDataset,
                    path: str,
                    dk_kwargs: dict,
                    gs_kwargs: dict,
                    w_kwargs: dict) -> None:
    writer = DatasetWriter(path,
                           {'dk': NDArrayField(**dk_kwargs),
                           #'ogs' : NDArrayField(**gs_kwargs),
                            'gs': NDArrayField(**gs_kwargs)},
                           **w_kwargs)
    writer.from_indexed_dataset(d)

# def img_pipeline(mu, sd) -> List[Operation]:
#     return [SimpleRGBImageDecoder(),
#             NormalizeImage(mu, sd, np.float32),
#             ToTensor(),
#             ToTorchImage(convert_back_int16 =False),
#             Convert(torch.float32),
#             ]

def object_pipeline() -> List[Operation]:
    return [NDArrayDecoder(),
            ToTensor(),
            Convert(torch.float32)]

def object_loader(path: str, device,  **kwargs) -> Loader:
    with open(path + '_manifest.json', 'r') as f:
        manifest = json.load(f)
    l =  Loader(path + '.beton',
                pipelines= {'dk' : object_pipeline() +
                                      [ToDevice(device)],
                            #'ogs' : object_pipeline() +
                                      #[ToDevice(device)],
                            'gs': None},
                **kwargs)
    return l
