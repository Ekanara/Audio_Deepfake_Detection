

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from src.utils.arguments import get_args
import torch.nn as nn
from typing import Any, Optional
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, inception_v3
from src.base_pipeline import BasePipeline


class EfficientNetPipeline(BasePipeline):
    def __init__(self, args=None, device=None, model = None, in_features=None, num_label=4):
        super().__init__(args=args, device=device, model=model, in_features=in_features, num_label=num_label)


    def forward_pipeline(self, input):
        output = self.model(input)
        return output
        
if __name__ == "__main__":
    args = get_args()
    dummy_input = torch.randn(1, 2048, 7, 7)
    model = efficientnet_b1(pretrained=True)
    pipeline = BasePipeline(args=args, model=model, in_features=3)
    output = pipeline.forward_pipeline(dummy_input)
    print(output.shape)
    


