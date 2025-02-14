import torch
import torch.nn as nn
from torchvision.models import resnet18
from mmcv.ops import RoIAlignRotated

class ModifiedResNet(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        self.features = nn.Sequential(*list(original_resnet.children())[:-2])  # Remove avgpool and fc
        
    def forward(self, x):
        return self.features(x)

class BBoxDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = ModifiedResNet(resnet18(pretrained=True))
        
        # ROI pooling layer
        self.roi_pool = RoIAlignRotated(output_size=(7, 7), spatial_scale=1/16, sampling_ratio=2)
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 24)  # 8 * 3
        )

    def forward(self, image, boxes):

        batch_size = image.shape[0]
        num_objects = boxes.shape[1]

        # Extract features
        features = self.backbone(image)
        
        batch_ids = torch.zeros(boxes.shape[1], device=boxes.device)
        boxes = boxes.squeeze(0)
        boxes = torch.column_stack((batch_ids, boxes))

        # Pool ROI features for each box
        roi_features = self.roi_pool(features, boxes)
        roi_features = roi_features.view(roi_features.size(0), -1)

        # print("roi feat ", roi_features.shape)

        bbox = self.bbox_regressor(roi_features)
        bbox = bbox.unsqueeze(0).reshape(batch_size, num_objects, 8, 3)
        
        return bbox