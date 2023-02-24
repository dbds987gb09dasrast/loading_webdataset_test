import os
import sys
from glob import glob
from datetime import datetime

import numpy as np
import cv2
import webdataset as wds

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, Normalize, ToTensor


def identity(x):
    return x


def main():
    data_root= "/path/to/laion-400m"
    data_index= [int(os.path.splitext(os.path.basename(fl))[0]) for fl in glob(os.path.join(data_root, "*.tar"))]
    data_index.sort()
    print(len(data_index))
    print(data_index[:5])

    batch_path= os.path.join(data_root, "{:05d}.tar".format(data_index[1010]))

    # start1= datetime.now()
    # ds= wds.WebDataset(batch_path)
    # ds= ds.decode("rgb").to_tuple("webp", "json")
    # start2= datetime.now()
    # img, js= next(iter(ds))
    # print(np.max(img))
    # cv2.imwrite("prev.png", (img*255).astype(np.uint8))
    # print(img.shape)
    # print(js)

    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    preproc = Compose([
        RandomResizedCrop(384),
        ToTensor(),
        normalize])
    
    start1= datetime.now()
    ds = (
        wds.WebDataset(batch_path)
        .decode("pil")
        .to_tuple("webp", "json")
        .map_tuple(preproc, identity))
    
    start2= datetime.now()
    batch_size = 128
    dl = DataLoader(ds.batched(batch_size), num_workers=8, batch_size=None, pin_memory=True)
    start3= datetime.now()
    for img, trg in dl:
        # print(img.shape)
        pass
    # tend= datetime.now()
    # print(start2-start1)
    # print(tend-start2)
    allend= datetime.now()

    print(start2-start1)
    print(start3-start2)
    print(allend-start3)
    # SATA
    #bs= 64: worker=2~0:00:18.701800,  worker=4~0:00:19.007674, worker=8~0:00:18.954796
    #bs=128: worker=2~0:00:18.952625,  worker=4~0:00:18.693086, worker=8~0:00:19.020247

    # SSD
    #bs= 64: worker=2~0:00:18.991907,  worker=4~0:00:18.950253, worker=8~0:00:18.980862
    #bs=128: worker=2~0:00:18.697496,  worker=4~0:00:18.702886, worker=8~0:00:18.980204





if __name__=="__main__":
    main()
