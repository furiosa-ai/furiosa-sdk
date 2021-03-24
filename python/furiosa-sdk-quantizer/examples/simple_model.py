import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 3, 3)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.conv3(x)
        return x


dummy_input = torch.randn(1, 3, 224, 224)
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(Net(),
                  (dummy_input,),
                  "simple_model.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names)
