from model import Uformer
import torch
# from utils import load_checkpoint
import os
from utils import load_img
import numpy as np
from collections import OrderedDict
import cv2

model_restoration = Uformer(img_size=128,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
    depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=3)  


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model

model_restoration  = load_checkpoint(model_restoration,'/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/cvip/code/Uformer/weights/model_best.pth')
import tqdm
ds_path = '/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/cvip/code/Uformer/test_ds/input'
input_list = os.listdir(ds_path)
with torch.no_grad():
    model_restoration.eval()
    for im_name in tqdm.tqdm(input_list):
        im = load_img(os.path.join(ds_path,im_name))
        name = im_name.split('_')[0]
        im = torch.from_numpy(np.float32(im)).permute(2,0,1).unsqueeze(dim=0)
        with torch.cuda.amp.autocast():
            out = model_restoration(im)
        out = torch.clamp(out,0,1)[0,...]
        out = out.detach().cpu().permute(1,2,0).numpy()*255
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/cvip/code/Uformer/test_ds/pred/{name}_gt.png',out)


