import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image

from FlexEditMetric.GroundedSAM import GroundedSAM, get_sam_embeddings

from tqdm import tqdm

import matplotlib.pyplot as plt

def print_result(scores):
    for key, value in scores.items():
        if key != 'SC' and key != 'PQ':
            print(f"{key}: {np.mean(value):.2f}+-{np.std(value):.2f}")
        else:
            value0 = np.array(value, dtype=np.float32)[:, 0]
            value1 = np.array(value, dtype=np.float32)[:, 1]
            print(f"{key}: {value0.mean():.2f}+-{value0.std():.2f}; {value1.mean():.2f}+-{value1.std():.2f}")


if __name__ == "__main__":
    
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    from datasets import load_dataset
    datapath = '/network/scratch/r/rabiul.awal/escher/AURORA/data/'
    import os
    list_files = os.listdir(datapath)
    dataset = load_dataset('parquet', data_files=[datapath+filename for filename in list_files]).shuffle()
    num_samples = 101

    scores = {'PSNR': [], 'SSIM': [], 'LPIPS': [], 'SAM': []}
    GS = GroundedSAM()

    for data_i, data in tqdm(enumerate(dataset['train'])):
        
        prompt = data['instruction']
        image_a = data['input']
        image_b = data['output']

        if image_a.size != image_b.size:
            import warnings
            warnings.warn(f"Image sizes are different: {image_a.size} and {image_b.size}")
            image_b = image_b.resize(image_a.size)
       
        results = GS.compute_metrics(image_a, image_b, prompt)
        scores['PSNR'].append(results['PSNR'])
        scores['SSIM'].append(results['SSIM'])
        scores['LPIPS'].append(results['LPIPS'])
        scores['SAM'].append(results['SAM'])
        
        print_result(results)

        num_samples -= 1
        # except IndexError as e:
        #     print(e)
        #     continue
        # except KeyboardInterrupt:
        #     print_result(scores)
        #     break
        # except Exception as e:
        #     print(e)
        #     break
            
        if num_samples <= 0:
            print("Finished sampling")
            print_result(scores)
            break