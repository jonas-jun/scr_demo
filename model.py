import torch.nn as nn
import clip
from joblib import load

class ScoringNet(nn.Module):
    def __init__(self, pretrain_name="RN50x16", svr_path=None):
        super(ScoringNet, self).__init__()
        model, preprocess = clip.load(pretrain_name)
        self.clip = model.visual
        print("clip loaded")
        self.svr = load(svr_path)
        print("svr loaded")

    def forward(self, input):
        out = self.clip(input).detach().cpu()
        out = self.svr.predict(out)
        return out  # numpy array type
