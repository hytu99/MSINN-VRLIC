import math

import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import LowerBound
from compressai.layers import CheckerboardMaskedConv2d

from .model_utils import conv, quantize_ste, Demultiplexer, Multiplexer, Space2Depth, Depth2Space
from .layers import MultistageMaskedConv2d, InvUnit
from .post_process import UNetPost

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class MSINNCompression(nn.Module):
    def __init__(self, N=128, M=192, depths=[4,4,4,4], flow_permutation='invconv', flow_coupling='affine',
                use_act_norm=True, post_process=True, lrp=True, sc_type='multi_ckbd', spatial_context_num=5,
                anchor_num=8):
        super().__init__()

        self.num_iters = 5
        self.M = M
        self.post_process = post_process
        self.lrp = lrp

        self.sc_type = sc_type
        self.spatial_context_num = spatial_context_num

        self.space2depth = Space2Depth(r=2)
        self.depth2space = Depth2Space(r=2)

        input_dim =3 * 4
        self.g_a0 = nn.ModuleList(
            InvUnit(input_dim, N, flow_permutation, flow_coupling, use_act_norm) for _ in range(depths[0])
        )
        input_dim = input_dim * 2
        self.g_a1 = nn.ModuleList(
            InvUnit(input_dim, N, flow_permutation, flow_coupling, use_act_norm) for _ in range(depths[1])
        )
        input_dim = input_dim * 2
        self.g_a2 = nn.ModuleList(
            InvUnit(input_dim, N, flow_permutation, flow_coupling, use_act_norm) for _ in range(depths[2])
        )
        input_dim = input_dim * 2

        self.g_a3 = nn.ModuleList(
            InvUnit(input_dim, M, flow_permutation, flow_coupling, use_act_norm) for _ in range(depths[3])
        )

        if self.post_process:
            self.post_net = UNetPost()

        self.channel_split = [48, 48, 24, 12, 6]

        self.initial_scale = nn.Parameter(torch.ones(1, self.channel_split[0], 1, 1))
        internal_channel = 128

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.channel_split[i], internal_channel, stride=1, kernel_size=3),
                nn.GELU(),
                conv(internal_channel, internal_channel, stride=1, kernel_size=3),
                nn.GELU(),
                conv(internal_channel, internal_channel, stride=1, kernel_size=3),
                nn.GELU(),
                conv(internal_channel, internal_channel, stride=1, kernel_size=3),
                nn.GELU(),
                conv(internal_channel, self.channel_split[i] * 2, stride=1, kernel_size=3),
            ) for i in range(1, self.num_iters)
        )

        if self.sc_type == 'multi_ckbd':
            self.sc_transforms = nn.ModuleList(
                nn.Sequential(
                    MultistageMaskedConv2d(self.channel_split[i], internal_channel, kernel_size=3, padding=1, stride=1, mask_type='A'),
                    nn.GELU(),
                    MultistageMaskedConv2d(internal_channel, internal_channel, kernel_size=3, padding=1, stride=1, mask_type='B'),
                    nn.GELU(),
                    MultistageMaskedConv2d(internal_channel, internal_channel, kernel_size=3, padding=1, stride=1, mask_type='B'),
                    nn.GELU(),
                    MultistageMaskedConv2d(internal_channel, self.channel_split[i] * 2, kernel_size=3, padding=1, stride=1, mask_type='B'),
                ) for i in range(self.num_iters)
            )
        elif 'ckbd' in self.sc_type:
            kernel_size = int(self.sc_type.split('_')[1][1:])
            padding = (kernel_size - 1) // 2
            self.sc_transforms = nn.ModuleList(
                CheckerboardMaskedConv2d(self.channel_split[i], 2 * self.channel_split[i], kernel_size=kernel_size, stride=1, padding=padding)
                for i in range(self.num_iters)
            )

        self.entropy_parameters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.channel_split[i] * 12 // 3, self.channel_split[i] * 10 // 3, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.channel_split[i] * 10 // 3, self.channel_split[i] * 8 // 3, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.channel_split[i] * 8 // 3, self.channel_split[i] * 6 // 3, kernel_size=1),
            ) for i in range(self.num_iters)
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.anchor_num = anchor_num

        self.gain = nn.Parameter(torch.ones((anchor_num, sum(self.channel_split), 1, 1)))
        self.inv_gain = nn.Parameter(torch.ones((anchor_num, sum(self.channel_split), 1, 1)))
        self.scale_gain = nn.Parameter(torch.ones((anchor_num, sum(self.channel_split), 1, 1)))

        self.lower_bound = LowerBound(0.01)


        if self.lrp:
            self.lrp_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(self.channel_split[i] * 2, internal_channel, stride=1, kernel_size=3) if i > 0 else \
                        conv(self.channel_split[i], internal_channel, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(internal_channel, internal_channel, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(internal_channel, internal_channel, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(internal_channel, internal_channel, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(internal_channel, self.channel_split[i], stride=1, kernel_size=3),
                ) for i in range(self.num_iters)
            )


    def forward(self, x, qp_index):

        gain_slices = self.gain.split(self.channel_split, dim=1)
        inv_gain_slices = self.inv_gain.split(self.channel_split, dim=1)
        scale_gain_slices = self.scale_gain.split(self.channel_split, dim=1)

        y = self.g_a(x)

        y_slices = y[::-1]
        y_hat_slices = []

        y_likelihood = []
        for slice_index, y_slice in enumerate(y_slices): 
            gain_vector = self.get_gain_vectors(gain_slices, slice_index, qp_index)   
            inv_gain_vector = self.get_gain_vectors(inv_gain_slices, slice_index, qp_index)   
            scale_gain_vector = self.get_gain_vectors(scale_gain_slices, slice_index, qp_index)   

            if slice_index == 0:
                scales_hat = torch.zeros_like(y_slice)
                scales_hat[:] = self.initial_scale
                means_hat = torch.zeros_like(scales_hat) 
                cc_params = torch.cat((scales_hat, means_hat), dim=1)
            else:
                # support_slices = torch.cat(y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index - 1](support_slices)
                scales_hat, means_hat = cc_params.chunk(2, 1)

            if self.num_iters - slice_index <= self.spatial_context_num:
                sc_params = torch.zeros((y_slice.shape[0], y_slice.shape[1] * 2, y_slice.shape[2], y_slice.shape[3]), device=y_slice.device)
                gaussian_params = self.entropy_parameters[slice_index](
                    torch.cat((cc_params, sc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                
                res = (y_slice - means_hat) * self.lower_bound(gain_vector)
                res_hat = quantize_ste(res)
                res_hat = res_hat * self.lower_bound(inv_gain_vector)
                y_hat_slice = res_hat + means_hat

                if slice_index % 2 == 0:
                    y_half = y_hat_slice.clone()
                    y_half[:, :, 0::2, 0::2] = 0
                    y_half[:, :, 1::2, 1::2] = 0
                else:
                    y_half = y_hat_slice.clone()
                    y_half[:, :, 0::2, 1::2] = 0
                    y_half[:, :, 1::2, 0::2] = 0

                if slice_index % 2 == 0:
                    sc_params = self.sc_transforms[slice_index](y_half)
                    sc_params[:, :, 0::2, 1::2] = 0
                    sc_params[:, :, 1::2, 0::2] = 0
                else:
                    sc_params = self.sc_transforms[slice_index](y_half)
                    sc_params[:, :, 0::2, 0::2] = 0
                    sc_params[:, :, 1::2, 1::2] = 0

                gaussian_params = self.entropy_parameters[slice_index](
                    torch.cat((cc_params, sc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
            
            res = (y_slice - means_hat) * self.lower_bound(gain_vector)
            res_hat = quantize_ste(res)
            res_hat = res_hat * self.lower_bound(inv_gain_vector)
            y_hat_slice = res_hat + means_hat 

            if self.lrp:
                # LRP
                if slice_index == 0:
                    lrp_support = y_hat_slice
                else:
                    lrp_support = torch.cat((y_hat_slice, support_slices), dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice = y_hat_slice + lrp * self.lower_bound(inv_gain_vector)

            # y_hat_slice = y_slice
            y_hat_slices.append(y_hat_slice)

            _, y_slice_likelihood = self.gaussian_conditional(res, scales_hat * self.lower_bound(scale_gain_vector), means=None)
            y_likelihood.append(y_slice_likelihood)

            if slice_index == 0:
                support_slices = y_hat_slice
            elif slice_index == 1:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a3):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)
            elif slice_index == 2:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a2):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)
            elif slice_index == 3:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a1):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)
            elif slice_index == 4:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a0):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)

        # Generate the image reconstruction.
        # x_hat = self.g_s(y_hat)
        x_hat = support_slices

        if self.post_process:
            x_hat = self.post_net(x_hat)
        likelihoods = {f"y{i}": y_likelihood[i] for i in range(len(y_likelihood))}
        return {
            "x_hat": x_hat,
            "y": y,
            "y_hat": y_hat_slices,
            "likelihoods": likelihoods,
        }

    
    def g_a(self, x):
        x = self.space2depth(x)
        for layer in self.g_a0:
            x = layer(x)
        x1, x2 = x.chunk(2, dim=1)

        x2 = self.space2depth(x2)
        for layer in self.g_a1:
            x2 = layer(x2)
        x2, x3 = x2.chunk(2, dim=1)

        x3 = self.space2depth(x3)
        for layer in self.g_a2:
            x3 = layer(x3)
        x3, x4 = x3.chunk(2, dim=1)

        x4 = self.space2depth(x4)
        for layer in self.g_a3:
            x4 = layer(x4)
        x4, x5 = x4.chunk(2, dim=1)

        return (x1, x2, x3, x4, x5)
    
    def g_s(self, y):
        x1, x2, x3, x4, x5 = y

        x4 = torch.cat((x4, x5), dim=1)
        for layer in reversed(self.g_a3):
            x4 = layer(x4, reverse=True)
        x4 = self.depth2space(x4)

        x3 = torch.cat((x3, x4), dim=1)
        for layer in reversed(self.g_a2):
            x3 = layer(x3, reverse=True)
        x3 = self.depth2space(x3)

        x2 = torch.cat((x2, x3), dim=1)
        for layer in reversed(self.g_a1):
            x2 = layer(x2, reverse=True)
        x2 = self.depth2space(x2)

        x = torch.cat((x1, x2), dim=1)
        for layer in reversed(self.g_a0):
            x = layer(x, reverse=True)
        x = self.depth2space(x)

        return x

        
    def get_gain_vectors(self, slices, slice_index, qp_index):
        if isinstance(qp_index, list):
            return slices[slice_index][qp_index]
        
        if isinstance(qp_index, float):
            index_floor, index_ceil = int(qp_index), int(qp_index) + 1
            left = qp_index - index_floor

            left_vector = slices[slice_index][index_floor:index_floor+1]
            right_vector = slices[slice_index][index_ceil:index_ceil+1]
            return left_vector ** (1 - left) * right_vector ** left
        
        return slices[slice_index][qp_index:qp_index+1]

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv

        return updated
         
    def compress(self, x, qp_index):
        gain_slices = self.gain.split(self.channel_split, dim=1)
        inv_gain_slices = self.inv_gain.split(self.channel_split, dim=1)
        scale_gain_slices = self.scale_gain.split(self.channel_split, dim=1)
        y = self.g_a(x)

        y_slices = y[::-1]
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder() 
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                scales_hat = torch.zeros_like(y_slice)
                scales_hat[:] = self.initial_scale
                means_hat = torch.zeros_like(scales_hat) 
                cc_params = torch.cat((scales_hat, means_hat), dim=1)
            else:
                cc_params = self.cc_transforms[slice_index - 1](support_slices)
                scales_hat, means_hat = cc_params.chunk(2, 1)
            sc_params = torch.zeros((y_slice.shape[0], y_slice.shape[1] * 2, y_slice.shape[2], y_slice.shape[3]), device=y_slice.device)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((cc_params, sc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            if slice_index % 2 == 0:
                y_slice_anchor, y_slice_non_anchor = Demultiplexer(y_slice)  
                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)
            else:
                y_slice_non_anchor, y_slice_anchor = Demultiplexer(y_slice)
                _, scales_hat_anchor = Demultiplexer(scales_hat)
                _, means_hat_anchor = Demultiplexer(means_hat)

            gain_vector = self.get_gain_vectors(gain_slices, slice_index, qp_index)   
            inv_gain_vector = self.get_gain_vectors(inv_gain_slices, slice_index, qp_index)   
            scale_gain_vector = self.get_gain_vectors(scale_gain_slices, slice_index, qp_index)   

            gain_vector_ = gain_vector.repeat(1, 2, 1, 1)
            inv_gain_vector_ = inv_gain_vector.repeat(1, 2, 1, 1)
            scale_gain_vector_ = scale_gain_vector.repeat(1, 2, 1, 1)

            res_anchor = (y_slice_anchor - means_hat_anchor) * self.lower_bound(gain_vector_)

            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor* self.lower_bound(scale_gain_vector_))
            res_q_anchor = self.gaussian_conditional.quantize(res_anchor, "symbols")
            y_hat_slice_anchor = res_q_anchor * self.lower_bound(inv_gain_vector_) + means_hat_anchor

            symbols_list.extend(res_q_anchor.reshape(-1).tolist())
            indexes_list.extend(index_anchor.reshape(-1).tolist())

            if slice_index % 2 == 0:
                y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))

                sc_params = self.sc_transforms[slice_index](y_hat_slice)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
            else:
                y_hat_slice = Multiplexer(torch.zeros_like(y_hat_slice_anchor), y_hat_slice_anchor)

                sc_params = self.sc_transforms[slice_index](y_hat_slice)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0

            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((cc_params, sc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            if slice_index % 2 == 0:
                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)
            else:
                scales_hat_non_anchor, _ = Demultiplexer(scales_hat)
                means_hat_non_anchor, _ = Demultiplexer(means_hat)

            res_non_anchor = (y_slice_non_anchor - means_hat_non_anchor) * self.lower_bound(gain_vector_)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor * self.lower_bound(scale_gain_vector_))
            res_q_non_anchor = self.gaussian_conditional.quantize(res_non_anchor, "symbols")
            y_hat_slice_non_anchor = res_q_non_anchor * self.lower_bound(inv_gain_vector_) + means_hat_non_anchor

            symbols_list.extend(res_q_non_anchor.reshape(-1).tolist())
            indexes_list.extend(index_non_anchor.reshape(-1).tolist())

            if slice_index % 2 == 0:
                y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            else:
                y_hat_slice = Multiplexer(y_hat_slice_non_anchor, y_hat_slice_anchor)

            # LRP
            if slice_index == 0:
                lrp_support = y_hat_slice
            else:
                lrp_support = torch.cat((y_hat_slice, support_slices), dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp * self.lower_bound(inv_gain_vector)

            y_hat_slices.append(y_hat_slice)

            if slice_index == 0:
                support_slices = y_hat_slice
            elif slice_index == 1:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a3):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)
            elif slice_index == 2:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a2):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)
            elif slice_index == 3:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a1):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)
            # elif slice_index == 4:
            #     support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
            #     for layer in reversed(self.g_a0):
            #         support_slices = layer(support_slices, reverse=True)
            #     support_slices = self.depth2space(support_slices)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings], "shape": y_slices[0].size()[-2:]}
    
    def decompress(self, strings, shape, qp_index):
        assert isinstance(strings, list)

        inv_gain_slices = self.inv_gain.split(self.channel_split, dim=1)
        # gain_slices = self.gain.split(self.channel_split, dim=1)
        scale_gain_slices = self.scale_gain.split(self.channel_split, dim=1)

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)
        for slice_index in range(self.num_iters):
            if slice_index == 0:
                slice_shape = shape
                scales_hat = self.initial_scale.expand((1, -1, shape[0], shape[1]))
                means_hat = torch.zeros_like(scales_hat) 
                cc_params = torch.cat((scales_hat, means_hat), dim=1)
            else:
                slice_shape = (shape[0] * 2**(slice_index - 1), shape[1] * 2**(slice_index - 1))
                # support_slices = torch.cat(y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index - 1](support_slices)
                scales_hat, means_hat = cc_params.chunk(2, 1)

            sc_params = torch.zeros_like(cc_params)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((cc_params, sc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            if slice_index % 2 == 0:  
                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)
            else:
                _, scales_hat_anchor = Demultiplexer(scales_hat)
                _, means_hat_anchor = Demultiplexer(means_hat)

            inv_gain_vector = self.get_gain_vectors(inv_gain_slices, slice_index, qp_index)   
            scale_gain_vector = self.get_gain_vectors(scale_gain_slices, slice_index, qp_index)   

            inv_gain_vector_ = inv_gain_vector.repeat(1, 2, 1, 1)
            scale_gain_vector_ = scale_gain_vector.repeat(1, 2, 1, 1)

            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor * self.lower_bound(scale_gain_vector_))
            rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, slice_shape[0] // 2, slice_shape[1] // 2)
            y_hat_slice_anchor = rv.type_as(means_hat_anchor) * self.lower_bound(inv_gain_vector_) + means_hat_anchor

            if slice_index % 2 == 0:
                y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))

                sc_params = self.sc_transforms[slice_index](y_hat_slice)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0

            else:
                y_hat_slice = Multiplexer(torch.zeros_like(y_hat_slice_anchor), y_hat_slice_anchor)

                sc_params = self.sc_transforms[slice_index](y_hat_slice)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0

            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((cc_params, sc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            if slice_index % 2 == 0:
                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)
            else:
                scales_hat_non_anchor, _ = Demultiplexer(scales_hat)
                means_hat_non_anchor, _ = Demultiplexer(means_hat)

            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor * self.lower_bound(scale_gain_vector_))
            rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, slice_shape[0] // 2, slice_shape[1] // 2)
            y_hat_slice_non_anchor = rv.type_as(means_hat_non_anchor) * self.lower_bound(inv_gain_vector_) + means_hat_non_anchor

            if slice_index % 2 == 0:
                y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            else:
                y_hat_slice = Multiplexer(y_hat_slice_non_anchor, y_hat_slice_anchor)

            # LRP
            if slice_index == 0:
                lrp_support = y_hat_slice
            else:
                lrp_support = torch.cat((y_hat_slice, support_slices), dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp * self.lower_bound(inv_gain_vector)

            y_hat_slices.append(y_hat_slice)

            if slice_index == 0:
                support_slices = y_hat_slice
            elif slice_index == 1:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a3):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)
            elif slice_index == 2:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a2):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)
            elif slice_index == 3:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a1):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)
            elif slice_index == 4:
                support_slices = torch.cat((y_hat_slice, support_slices), dim=1)
                for layer in reversed(self.g_a0):
                    support_slices = layer(support_slices, reverse=True)
                support_slices = self.depth2space(support_slices)

        x_hat = support_slices

        if self.post_process:
            x_hat = self.post_net(x_hat)
        x_hat = x_hat.clamp_(0, 1)

        return {"x_hat": x_hat}

