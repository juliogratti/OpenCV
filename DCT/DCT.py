import torch
import torch_dct as dct

x = torch.randn(200)
X = dct.dct(x)   # DCT-II done through the last dimension
y = dct.idct(X)  # scaled DCT-III done through the last dimension
assert (torch.abs(x - y)).sum() < 1e-10  # x == y within numerical tolerance