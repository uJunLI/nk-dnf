import jittor as jt
from jittor import nn
from utils.conv_padding_mode import Pad2dMode

class PDConvFuse(nn.Module):
    def __init__(self, in_channels=None, f_number=None, feature_num=2, bias=True, **kwargs) -> None:
        super().__init__()
        if in_channels is None:
            assert f_number is not None
            in_channels = f_number
        self.feature_num = feature_num
        self.act = nn.GELU()
        self.pwconv = nn.Conv2d(feature_num * in_channels, in_channels, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias, groups=in_channels)

    def execute(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        return self.dwconv(self.act(self.pwconv(jt.concat(inp_feats, dim=1))))

class GFM(nn.Module):
    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode='reflect', **kwargs) -> None:
        super().__init__()
        self.feature_num = feature_num
        self.act = nn.GELU()
        hidden_features = in_channels * feature_num
        self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, 1, 0, bias=bias)
        self.Pad2dMode = Pad2dMode(1,padding_mode)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 0, bias=bias, groups=hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)
        self.mlp = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)


    def execute(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        shortcut = inp_feats[0]
        x = jt.concat(inp_feats, dim=1)
        x = self.pwconv(x)

        x = self.Pad2dMode(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = self.act(x1) * x2
        x = self.project_out(x)
        return self.mlp(x + shortcut)
