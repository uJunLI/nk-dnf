from jittor import nn


class ResidualSwitchBlock(nn.Module):
    def __init__(self, block) -> None:
        super().__init__()
        self.block = block
        
    def execute(self, x, residual_switch):
        return self.block(x) + residual_switch * x