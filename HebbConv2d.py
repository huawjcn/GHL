import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class HebbConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,  # Number of input channels
            out_channels: int,  # Number of output channels
            kernel_size: int,  # Size of the convolutional kernel
            stride: int = 1,  # Stride of the convolution
            padding: int = 0,  # Padding added to all four sides of the input
            dilation: int = 1,  # Spacing between kernel elements (dilated convolution)
            padding_mode: str = 'zeros',
            t_invert: float = 1,  # Parameter for softmax temperature scaling
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Convert kernel_size, stride, dilation, and padding to (height, width) tuples
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])  # left, right, top, bottom
        self.padding_mode = padding_mode  # F.conv2d does not support non-zero padding modes, so manual padding is required in forward
        self.t_invert = t_invert

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        self.bias = None

        # These states require the computation of the Hebbian gradient
        self.compute_hebbian = True
        # Buffer to store the raw Hebbian gradients computed during the forward pass
        self.register_buffer('hebbian_grad', torch.zeros_like(self.weight.data))

    def forward(self, x):

        x = F.pad(x, self.padding, 'constant', 0)

        # Convolution operation
        weighted_input = F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation)

        if self.training:
            # Calculate post-synaptic activations for plasticity updates
            # Ensure Hebbian computation does not affect the main computation graph for backpropagation
            with torch.no_grad():
                # Get the shape of the "input"; here, the input for competitive learning is the result after convolution
                batch_size, out_channels, height_out, width_out = weighted_input.shape

                # --- Competitive learning across the channel dimension ---
                # Move the competitive dimension (channel) to dimension 0 and flatten other non-competitive dimensions -> (Channel(competitive_dim), Batch*Height*Width(non-competitive_dims))
                flat_weighted_inputs = weighted_input.transpose(0, 1).reshape(out_channels, -1)
                flat_wta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
                # Due to the previous flattening, reshape the activations back to the standard shape -> (B, OC, OH, OW). Note that the competitive dimension is moved back.
                wta_activs = flat_wta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)

                # --- Calculate Hebbian gradient ---
                # Using Oja's rule:
                # delta_weight = lr * y_k * (x_i - u_k * w_ik)
                #              = lr * (y_k * x_i - y_k * u_k * w_ik) will be calculated separately later
                #    y_k:  wta_activs      Result after competition (post-synaptic activation)
                #    x_i:  x               Input
                #    u_k:  weighted_input  Result before competition (weighted input)
                #    w_ik: self.weight     Weight

                # Get weight dimension information
                _, _, kernel_h, kernel_w = self.weight.shape

                # Due to weight sharing in CNNs, group the inputs that share weights into L columns. The remaining IC*KH*KW are the non-shared dimensions.
                # Note that padding = 0 here because we have already manually padded the input.
                # (B, IC, H, W) -> (B, IC*KH*KW, L) -> (B, L, IC*KH*KW)
                # L = OW * OH, which is the size of the output feature map, i.e., the number of results obtained after one pass of a kernel over the input. These all share the same weight.
                # Transpose the last two dimensions to facilitate subsequent multiplication.
                unfolded_x = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=0, stride=self.stride).transpose(1, 2)

                # Calculate y_k * x_i, first flatten the shape of y_k, then perform matrix multiplication (not element-wise).
                # The significance of matrix multiplication is to multiply and sum all x_i, y_k pairs that share the same weight (there are L such pairs).
                wta_activs_flat = wta_activs.reshape(batch_size, self.out_channels, -1)  # -> (B, OC, L)
                # This matrix multiplication is the sum of y_k * x_i for all shared weights.
                # (B, OC, L) @ (B, L, IC*KH*KW) -> (B, OC, IC*KH*KW)
                yx_flat = torch.matmul(wta_activs_flat, unfolded_x)

                # Sum over the batch dimension to get the final gradient for this batch, then reshape it to the normal weight shape.
                yx_flat = torch.sum(yx_flat, dim=0)  # (B, OC, IC*KH*KW) -> (OC, IC*KH*KW)
                yx = yx_flat.view(self.out_channels, self.in_channels, kernel_h, kernel_w)

                # Calculate y_k * u_k, their shapes are the same (both are y, just with or without competition), so a direct element-wise multiplication is sufficient.
                yu = torch.mul(wta_activs, weighted_input)  # (B, OC, OH, OW)
                # Calculate y_k * u_k * w_ik
                # Sum over the batch and spatial dimensions and expand dimensions for subsequent broadcasting (B, OC, OH, OW) -> (OC) -> (OC, 1, 1, 1)
                yu = torch.sum(yu, dim=(0, 2, 3)).view(-1, 1, 1, 1)
                yuw = yu * self.weight.data

                # Calculate delta_weight
                delta_weight = yx - yuw

                # Normalize the Hebbian gradient (to prevent explosion)
                # Perform global normalization on the gradient tensor of the entire layer
                delta_weight_norm = torch.abs(delta_weight).amax() + 1e-30
                delta_weight = delta_weight / delta_weight_norm

                # Store the computed and normalized Hebbian gradient in the buffer
                self.hebbian_grad.copy_(delta_weight)

        return weighted_input

    def update_gradients(self):
        if self.training and self.weight.grad is not None:
            with torch.no_grad():
                if hasattr(self, 'hebbian_grad'):
                    hebb_grad = self.hebbian_grad.to(self.weight.grad.device)

                combined_grad = torch.abs(hebb_grad) * torch.sign(self.weight.grad)
                self.weight.grad.copy_(combined_grad)
