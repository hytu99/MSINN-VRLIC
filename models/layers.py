import torch
import torch.nn as nn
from torch.nn import functional as F

from compressai.layers import ResidualBlock


class MultistageMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == 'A':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        elif mask_type == 'B':
            self.mask[:, :, 0::2, 0::2] = 1
            self.mask[:, :, 1::2, 1::2] = 1
            self.mask[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 1
        else:
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    def forward(self, x):
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)
    

class ActNorm2d(nn.Module):
    def __init__(self, num_features, scale=1.0):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_features = num_features
        self.scale = scale
        self.register_buffer("inited", torch.tensor(0, dtype=torch.uint8))

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited.fill_(1)

    def forward(self, input, reverse=False):
        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input = input * torch.exp(-self.logs) - self.bias
        else:
            input = (input + self.bias) * torch.exp(self.logs)

        return input

class RandPermute2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.register_buffer("indices", torch.randperm(num_channels))
        self.register_buffer("indices_inverse", torch.argsort(self.indices))

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]
        

class Reverse2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1, -1, -1, dtype=torch.long)
        self.indices_inverse = torch.argsort(self.indices)

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]
        
    
class CrossPermute2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels, dtype=torch.long)
        temp = self.indices.clone()
        self.indices[0::2], self.indices[1::2] = temp[:self.num_channels // 2], temp[self.num_channels // 2:]
        self.indices_inverse = torch.argsort(self.indices)

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]
    

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.linalg.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        if not self.LU_decomposed:
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(self.lower.device)
            self.eye = self.eye.to(self.lower.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous().to(self.upper.device)
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1).to(input.device)

    def forward(self, input, reverse=False):
        weight = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            return z
        else:
            z = F.conv2d(input, weight)
            return z


class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, input):
        output = self.conv(input)
        return output


class InvUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        flow_permutation='invconv',
        flow_coupling='affine',
        use_act_norm=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.flow_coupling = flow_coupling
        self.use_act_norm = use_act_norm

        if use_act_norm:
            self.actnorm = ActNorm2d(in_channels)

        # 2. permute
        if flow_permutation == "invconv":
            self.permute = InvertibleConv1x1(in_channels)
        elif flow_permutation == "shuffle":
            self.permute = RandPermute2d(in_channels)
        elif flow_permutation == "cross":
            self.permute = CrossPermute2d(in_channels)
        elif flow_permutation == "reverse":
            self.permute = Reverse2d(in_channels)

        # 3. coupling
        if flow_coupling == "additive":
            self.block = nn.Sequential(
                 ResidualBlock(in_ch=in_channels // 2, out_ch=hidden_channels),
                #  nn.LeakyReLU(inplace=True),
                 Conv2dZeros(hidden_channels, in_channels // 2, kernel_size=1, stride=1)
            )
        elif flow_coupling == "affine":
            self.block = nn.Sequential(
                 ResidualBlock(in_ch=in_channels // 2, out_ch=hidden_channels),
                #  nn.LeakyReLU(inplace=True),
                 Conv2dZeros(hidden_channels, in_channels, kernel_size=1, stride=1)
            )


    def forward(self, input, reverse=False):
        if not reverse:
            return self.inv_forward(input)
        else:
            return self.inv_reverse(input)

    def inv_forward(self, input):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        if self.use_act_norm:
            input = self.actnorm(input, reverse=False)

        # 2. permute
        z = self.permute(input, False)

        # 3. coupling
        z1, z2 = z.chunk(2, dim=1)
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            scale, shift = torch.chunk(self.block(z1), 2, dim=1)
            scale = torch.exp(torch.sigmoid(scale) * 2 - 1)
            z2 = z2 + shift
            z2 = z2 * scale

        z = torch.cat((z1, z2), dim=1)

        return z

    def inv_reverse(self, input):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = input.chunk(2, dim=1)
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            scale, shift = torch.chunk(self.block(z1), 2, dim=1)
            scale = torch.exp(torch.sigmoid(scale) * 2 - 1)
            z2 = z2 / scale
            z2 = z2 - shift

        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z = self.permute(z, True)

        # 3. actnorm
        if self.use_act_norm:
            z = self.actnorm(z, reverse=True)

        return z
    
