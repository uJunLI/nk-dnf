


from jittor import nn

class SimpleDownsample(nn.Module):
    def __init__(self, dim, *, padding_mode='reflect'):
        super().__init__()
        self.body = nn.Conv2d(dim, dim*2, kernel_size=2, stride=2, padding=0, bias=False)

    def execute(self, x):
        return self.body(x)

class SimpleUpsample(nn.Module):
    def __init__(self, dim, *, padding_mode='reflect'):
        super().__init__()
        self.body = nn.ConvTranspose2d(dim, dim//2, kernel_size=2, stride=2, padding=0, bias=False)

    def execute(self, x):
        return self.body(x)
