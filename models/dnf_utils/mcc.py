from jittor import nn
import jittor
from einops import rearrange
from ..utils import LayerNorm
from utils.conv_padding_mode import Pad2dMode

class MCC(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
        super().__init__()
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')
        self.Pad2dMode = Pad2dMode(1,padding_mode)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(jittor.ones((num_heads, 1, 1)))
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 0, bias=bias, groups=f_number * 3)
        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
            nn.GELU(),
            self.Pad2dMode,
            nn.Conv2d(f_number, f_number, 3, 1, 0, bias=bias, groups=f_number),
            nn.GELU()
        )

    def execute(self, x):
        attn = self.norm(x)
        _, _, h, w = attn.shape

        qkv = self.dwconv(self.Pad2dMode(self.pwconv(attn)))
        q, k, v = qkv.chunk(3, dim=1)

        # 假设 q, k, v 是 Jittor 张量
        q = q.numpy()  # 将 Jittor 张量转换为 NumPy 数组
        k = k.numpy()
        v = v.numpy()

        # 进行 einops 的 rearrange 操作
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 将 NumPy 数组转换回 Jittor 张量
        q = jittor.Var(q)
        k = jittor.Var(k)
        v = jittor.Var(v)
        
        # 假设 q 和 k 是输入张量
        q = nn.layer_norm(q, normalized_shape=q.shape[-1:], eps=1e-5, elementwise_affine=False)
        k = nn.layer_norm(k, normalized_shape=k.shape[-1:], eps=1e-5, elementwise_affine=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = out.numpy()
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = jittor.Var(out)

        out = self.project_out(out)
        out = self.feedforward(out + x)
        return out
