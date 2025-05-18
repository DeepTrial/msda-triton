import torch
from msda_triton.autograd_function import triton_multiscale_deformable_attention
from msda_triton.torch_frontend import native_multiscale_deformable_attention, MultiscaleDeformableAttention



N = 320000
B, H, C, P = 2, 8, 64, 4
img_shapes = [(400,800)]
L = 1
I = sum(h * w for h, w in img_shapes) # noqa: E741
dtype = torch.float32
padding_mode = "border"
align_corners = True

print(B, I, H, C)
print(B, N, H, L, P, 2)
print(B, N, H, L, P)

img = torch.randn(B, I, H, C, device="cuda", dtype=dtype)
img_shapes = torch.tensor(img_shapes, device="cuda")
sampling_points = torch.rand(B, N, H, L, P, 2, device="cuda", dtype=dtype)
att_weights = torch.softmax(torch.randn(B, N, H, L, P, device="cuda", dtype=dtype), dim=-1)

test = triton_multiscale_deformable_attention(img, img_shapes, sampling_points, att_weights, padding_mode, align_corners)
print(test.shape)