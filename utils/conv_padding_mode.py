import jittor as jt
from jittor import nn

class Pad2dMode(nn.Module):
    def __init__(self, padding, padding_mode = "reflect"):
        super().__init__()
        self.padding = padding
        self.padding_mode = padding_mode

    def execute(self, x, ):
        return jt.nn.pad(x, [self.padding] * 4, mode=self.padding_mode)

