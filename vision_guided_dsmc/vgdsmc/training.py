from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, numpy as np
from .model import build_unet
@dataclass(frozen=True)
class TrainConfig:
    epochs:int=20; learning_rate:float=1e-3; batch_size:int=4; seed:int=7

def load_cases(paths):
    xs=[]; ys=[]
    for path in paths:
        with np.load(path) as data: xs.append(data['x'].astype(np.float32)); ys.append(data['label'].astype(np.int64))
    if not xs: raise ValueError('At least one case file is required')
    return np.stack(xs),np.stack(ys)

def normalize_channels(x):
    mean=x.mean(axis=(0,2,3),keepdims=True); std=np.maximum(x.std(axis=(0,2,3),keepdims=True),1e-6); return (x-mean)/std,mean,std

def train_model(case_paths,output_dir,cfg=TrainConfig()):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); x,y=load_cases(case_paths); x,mean,std=normalize_channels(x); dataset=TensorDataset(torch.from_numpy(x),torch.from_numpy(y)); loader=DataLoader(dataset,batch_size=min(cfg.batch_size,len(dataset)),shuffle=True)
    model=build_unet(); optimizer=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate); counts=np.bincount(y.ravel(),minlength=3).astype(np.float32); weights=counts.sum()/np.maximum(counts,1); weights/=weights.mean(); loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights)); history=[]; model.train()
    for _ in range(cfg.epochs):
        total=0
        for xb,yb in loader:
            optimizer.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); optimizer.step(); total+=float(loss.detach())*len(xb)
        history.append(total/len(dataset))
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); model_path=output/'model.pt'; torch.save({'state_dict':model.state_dict(),'mean':mean,'std':std,'history':history},model_path); (output/'metrics.json').write_text(json.dumps({'loss':history,'class_counts':counts.tolist()},indent=2)); return model_path

def predict_labels(model_path,x):
    import torch
    checkpoint=torch.load(model_path, map_location="cpu", weights_only=False); model=build_unet(); model.load_state_dict(checkpoint['state_dict']); model.eval(); normalized=(x[None].astype(np.float32)-checkpoint['mean'])/checkpoint['std']
    with torch.no_grad(): return model(torch.from_numpy(normalized)).argmax(dim=1).squeeze(0).numpy()
