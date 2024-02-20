import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

import torchvision.ops as ops

class DeformableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConv, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
        self.deform_conv = ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        return x

class DoubleConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2, self).__init__()
        self.conv1 = DeformableConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = DeformableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


import torchvision


def crop_and_concat(upsampled, bypass, crop = False):
    diffY = bypass.size()[2] - upsampled.size()[2]
    diffX = bypass.size()[3] - upsampled.size()[3]

    bypass = bypass[:, :, diffY // 2: bypass.size()[2] - diffY // 2 - diffY % 2,
                    diffX // 2: bypass.size()[3] - diffX // 2 - diffX % 2]

    return torch.cat((upsampled, bypass), 1)

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x


class DUNet(nn.Module):
    
    def __init__(self, num_classes):
        super(DUNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle=self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, 
                                               kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = DeformableConv2d(in_channels=1024, out_channels=512, 
                                             kernel_size=3, stride=1, padding=1)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, 
                                               kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = DeformableConv2d(in_channels=512, out_channels=256, 
                                             kernel_size=3, stride=1, padding=1)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, 
                                               kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                                               kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        return output_out


# class DeformableConv2d(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=1,
#                  dilation=1,
#                  bias=False):
#         super(DeformableConv2d, self).__init__()

#         assert type(kernel_size) == tuple or type(kernel_size) == int

#         kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
#         self.stride = stride if type(stride) == tuple else (stride, stride)
#         self.padding = padding
#         self.dilation = dilation

#         self.offset_conv = nn.Conv2d(in_channels,
#                                      2 * kernel_size[0] * kernel_size[1],
#                                      kernel_size=kernel_size,
#                                      stride=stride,
#                                      padding=self.padding,
#                                      dilation=self.dilation,
#                                      bias=True)

#         nn.init.constant_(self.offset_conv.weight, 0.)
#         nn.init.constant_(self.offset_conv.bias, 0.)

#         self.modulator_conv = nn.Conv2d(in_channels,
#                                         1 * kernel_size[0] * kernel_size[1],
#                                         kernel_size=kernel_size,
#                                         stride=stride,
#                                         padding=self.padding,
#                                         dilation=self.dilation,
#                                         bias=True)

#         nn.init.constant_(self.modulator_conv.weight, 0.)
#         nn.init.constant_(self.modulator_conv.bias, 0.)

#         self.regular_conv = nn.Conv2d(in_channels=in_channels,
#                                       out_channels=out_channels,
#                                       kernel_size=kernel_size,
#                                       stride=stride,
#                                       padding=self.padding,
#                                       dilation=self.dilation,
#                                       bias=bias)

#     def forward(self, x):
#         # h, w = x.shape[2:]
#         # max_offset = max(h, w)/4.

#         offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
#                 # Print offset values for the first training sample only
#         # print("Offset values for the first training sample:", offset[0])
#         # print("Offset shape:", offset.shape)
#         modulator = 2. * torch.sigmoid(self.modulator_conv(x))
#         # op = (n - (k * d - 1) + 2p / s)
#         x = torchvision.ops.deform_conv2d(input=x,
#                                           offset=offset,
#                                           weight=self.regular_conv.weight,
#                                           bias=self.regular_conv.bias,
#                                           padding=self.padding,
#                                           mask=modulator,
#                                           stride=self.stride,
#                                           dilation=self.dilation)

#         return x


# class DUNet(nn.Module):

#     def __init__(self, num_classes):
#         super(DUNet, self).__init__()
#         self.num_classes = num_classes
#         self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
#         self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
#         self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
#         self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
#         self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.middle = DeformableConv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
#         self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
#                                                 kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
#         self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
#                                                 kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
#         self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
#                                                 kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
#         self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
#                                                 kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
#         self.output = DeformableConv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

#     def conv_block(self, in_channels, out_channels):
#         block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.BatchNorm2d(num_features=out_channels),
#                                     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.BatchNorm2d(num_features=out_channels))
#         return block

#     def forward(self, X):
#         X = X.float()
#         # print(X.shape,"Here we are printing the shape of input to the first conv blcok, contracting_11")
#         contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
#         contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
#         contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
#         contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
#         contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
#         contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
#         contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
#         contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
#         # middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
#         middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]

#         expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
#         expansive_12_out = self.expansive_12(crop_and_concat(expansive_11_out, contracting_41_out)) # Cropping for matching dimensions
#         # expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
#         # expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
#         expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
#         expansive_22_out = self.expansive_22(crop_and_concat(expansive_21_out, contracting_31_out))
#         # expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
#         expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
#         expansive_32_out = self.expansive_32(crop_and_concat(expansive_31_out, contracting_21_out))

#         # expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
#         expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
#         expansive_42_out = self.expansive_42(crop_and_concat(expansive_41_out, contracting_11_out))
#         # expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
#         output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
#         # print("output_out shape: the final outout from the model which is used in the loss fx",output_out.shape)
#         return output_out


 
#resnext50_32x4d, mit_b2, timm-gernet_s, efficientnet-b3, mobilenet_v2, resnet152, vgg13		
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax' 

CLASSES = [ 'background','road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']
# create segmentation model with pretrained encoder
#Decoders= PAN, PSPNet, MAnet, Linknet, FPN, DeepLabV3, DeepLabV3Plus, Unet
model_smp =smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


model_unet = UNET(in_channels=3, out_channels=10, features=[64, 128, 256, 512, 1024])


model_dunet = DUNet(num_classes=len(CLASSES))


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()