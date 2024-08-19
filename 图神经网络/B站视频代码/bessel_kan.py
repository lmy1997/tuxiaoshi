import torch
import torch.nn as nn
from typing import List


class BesselKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3):
        super(BesselKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # Initialize Bessel polynomial coefficients
        self.bessel_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.bessel_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape x to (batch_size, input_dim)
        # Normalize x to [-1, 1] using tanh
        x = torch.tanh(x)

        # Initialize Bessel polynomial tensors
        bessel = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            bessel[:, :, 1] = x + 1  # y1(x) = x + 1
        for i in range(2, self.degree + 1):
            bessel[:, :, i] = (2 * i - 1) * x * bessel[:, :, i - 1].clone() + bessel[:, :, i - 2].clone()

        # Bessel interpolation using einsum for batched matrix-vector multiplication
        y = torch.einsum('bid,iod->bo', bessel, self.bessel_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)
        return y


# To avoid gradient vanishing caused by tanh
class BesselKANLayerWithNorm(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(BesselKANLayerWithNorm, self).__init__()
        self.layer = BesselKANLayer(input_dim=input_dim, output_dim=output_dim, degree=degree)
        self.layer_norm = nn.LayerNorm(output_dim) # To avoid gradient vanishing caused by tanh

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x
    
class Bessel_KAN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        degree: int = 3,
        grid_size: int = 8, # placeholder
        spline_order=0. # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            BesselKANLayerWithNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
            )
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x