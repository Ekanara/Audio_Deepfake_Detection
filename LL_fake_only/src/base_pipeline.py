
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, inception_v3


class BasePipeline(nn.Module):
    def __init__(self, args=None, device=None, model = None, in_features=None, num_label=4, stage1 = False, mode=['resnet', 'efficientnet','inception', 'densenet']):
        super().__init__()
        self.model = model
        self.device = device
        self.args = args
        self.in_features = in_features
        self.mode = mode
        self.stage1 = stage1
        self.num_label = num_label
        if self.mode == 'efficientnet':
            model_input_dim = self.model.classifier[1].in_features
            #self.model.classifier[1] = nn.Linear(model_input_dim, num_label)  
            self.model.classifier[1] = nn.Sequential(
                nn.Linear(model_input_dim, 512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.num_label)
            )

        elif self.mode == 'densenet':
            model_input_dim = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(model_input_dim, 512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.num_label)
            )

        elif self.mode in ['resnet', 'inception']:
            model_input_dim = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(model_input_dim, 512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.num_label)
            )

        elif self.mode == 'vgg':
            model_input_dim = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(model_input_dim, 512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.num_label)
            )

        elif self.mode == 'mobilenet':
            model_input_dim = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Sequential(
                nn.Linear(model_input_dim, 512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.num_label)
            )

        elif self.mode == 'convnext':
            model_input_dim = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Sequential(
                nn.Linear(model_input_dim, 512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.num_label)
            )

        elif self.mode == 'xception':
            model_input_dim = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(model_input_dim, 512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.num_label)
            )

        elif self.mode == 'nasnet':
            model_input_dim = self.model.last_linear.in_features
            self.model.last_linear = nn.Sequential(
                nn.Linear(model_input_dim, 512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.num_label)
            )

        
        self.softmax_head = nn.Sequential(
            nn.Linear(model_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 5),
            nn.Softmax(dim=1)
        )

        self.contrastive_head = nn.Sequential(
            nn.Linear(model_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        
        
    def forward_pipeline(self, input):
        #self.check_inputs()
        batch_size = input.shape[0]
        #input = self.decoder(input)
        #print(f"afer_decoder: {input.shape}")
        x_head = self.model(input)
        if self.num_label == 2:
            #softmax = nn.Softmax(dim=1)
            #model_pred = softmax(model_pred)
            return x_head
        else:
            softmax_head = self.softmax_head(x_head)
            constrative_head = self.contrastive_head(x_head)
        return x_head, softmax_head, constrative_head

    
    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pass


        
if __name__ == "__main__":
    try:
        from src.utils.arguments import get_args
    except ModuleNotFoundError:
        from utils.arguments import get_args

    args = get_args()
    dummy_input = torch.randn(1, 3, 224, 224)
    model = efficientnet_b1(pretrained=True)
    pipeline = BasePipeline(args=args, model=model, in_features=3)
    pipeline.eval()
    output, softmax_head, constrative_head = pipeline.forward_pipeline(dummy_input)

    
