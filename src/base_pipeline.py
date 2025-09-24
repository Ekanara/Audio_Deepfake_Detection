

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from utils.arguments import get_args
import torch.nn as nn
from typing import Any, Optional
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, inception_v3


class BasePipeline(nn.Module):
    def __init__(self, args=None, device=None, model = None, in_features=None, num_label=4):
        super().__init__()
        self.model = model
        self.device = device
        self.args = args
        self.in_features = in_features

        model_input_dim = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(model_input_dim, num_label)
    

    
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels = in_features, out_channels = 1024, kernel_size=4, stride=2, padding=1),  # 7->14
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size=4, stride=2, padding=1),          # 14->28
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size=4, stride=2, padding=1),           # 28->56
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size=4, stride=2, padding=1),            # 56->112
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels = 64, out_channels = 3, kernel_size=4, stride=2, padding=1)    # 112->224
        # )

    def forward_pipeline(self, input):
        #self.check_inputs()
        batch_size = input.shape[0]
        #input = self.decoder(input)
        #print(f"afer_decoder: {input.shape}")
        output = self.model(input)
        return output
    
    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pass


        
if __name__ == "__main__":
    args = get_args()
    dummy_input = torch.randn(1,3, 224, 224)
    model = efficientnet_b1(pretrained=True)
    pipeline = BasePipeline(args=args, model=model, in_features=3)
    output = pipeline.forward_pipeline(dummy_input)
    print(output.shape)
    


