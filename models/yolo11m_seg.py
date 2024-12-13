import torch
import torch.nn as nn


class YOLO11MSeg(nn.Module):
    def __init__(self, num_classes):
        super(YOLO11MSeg, self).__init__()
        # Define layers or load pre-trained yolo11m-seg weights.
        self.backbone = torch.hub.load('ultralytics/yolo', 'yolov5')
        self.backbone.classifier = nn.Conv2d(1280, num_classes, 1)

    def forward(self, x):
        return self.backbone(x)
