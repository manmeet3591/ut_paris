
__all__ = [
    "DiscriminatorForVGG", "SRResNet",
    "discriminator_for_vgg", "srresnet_x2", "srresnet_x4", "srresnet_x8",
]

feature_extractor_net_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(net_cfg_name: str, batch_norm: bool = False) -> nn.Sequential:
    net_cfg = feature_extractor_net_cfgs[net_cfg_name]
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in net_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class _FeatureExtractor(nn.Module):
    def __init__(
            self,
            net_cfg_name: str = "vgg19",
            batch_norm: bool = False,
            num_classes: int = 1000) -> None:
        super(_FeatureExtractor, self).__init__()
        self.features = _make_layers(net_cfg_name, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class SRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcb: int = 16,
            upscale: int = 4,
    ) -> None:
        super(SRResNet, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        # else:
        #     raise NotImplementedError(f"Upscale factor `{upscale}` is not support.")
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x


class DiscriminatorForVGG(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
    ) -> None:
        super(DiscriminatorForVGG, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(channels, channels, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, int(2 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(int(2 * channels), int(2 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(2 * channels), int(4 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(int(4 * channels), int(4 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(4 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(int(8 * channels), int(8 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(8 * channels) * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input image size must equal 96
        assert x.size(2) == 96 and x.size(3) == 96, "Input image size must be is 96x96"

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.rcb(x)

        x = torch.add(x, identity)

        return x


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            net_cfg_name: str,
            batch_norm: bool,
            num_classes: int,
            model_weights_path: str,
            feature_nodes: list,
            feature_normalize_mean: list,
            feature_normalize_std: list,
    ) -> None:
        super(ContentLoss, self).__init__()
        # Define the feature extraction model
        model = _FeatureExtractor(net_cfg_name, batch_norm, num_classes)
        # Load the pre-trained model
        if model_weights_path == "":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif model_weights_path is not None and os.path.exists(model_weights_path):
            checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
            if "state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError("Model weight file not found")
        # Extract the output of the feature extraction layer
        self.feature_extractor = create_feature_extractor(model, feature_nodes)
        # Select the specified layers as the feature extraction layer
        self.feature_extractor_nodes = feature_nodes
        # input normalization
        self.normalize = transforms.Normalize(feature_normalize_mean, feature_normalize_std)
        # Freeze model parameters without derivatives
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> [Tensor]:
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
        device = sr_tensor.device

        losses = []
        # input normalization
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        # Get the output of the feature extraction layer
        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        # Compute feature loss
        for i in range(len(self.feature_extractor_nodes)):
            losses.append(F_torch.mse_loss(sr_feature[self.feature_extractor_nodes[i]],
                                           gt_feature[self.feature_extractor_nodes[i]]))

        losses = torch.Tensor([losses]).to(device)

        return losses


def srresnet_x2(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale=2, **kwargs)

    return model


def srresnet_x4(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale=4, **kwargs)

    return model


def srresnet_x8(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale=8, **kwargs)

    return model


def discriminator_for_vgg(**kwargs) -> DiscriminatorForVGG:
    model = DiscriminatorForVGG(**kwargs)

    return model

import torch
import torch.nn as nn
from typing import Any

# Define a function to test the SRResNet model with a random torch tensor
def test_srresnet(upscale_factor: int = 4):
    # Create a random input tensor with shape (batch_size, channels, height, width)
    batch_size = 1
    in_channels = 24
    out_channels = 124

    height, width = 240, 240  # Adjust height and width as needed

    input_tensor = torch.rand((batch_size, in_channels, height, width))

    # Initialize the SRResNet model with the given upscale factor
    model = SRResNet(in_channels=in_channels, out_channels=out_channels, upscale=upscale_factor)

    # Forward pass through the model
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

# Test the SRResNet model with upscale factor of 4
test_srresnet(upscale_factor=1)

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, p=1):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
			nn.BatchNorm2d(out_channels),
			nn.PReLU(),
        	
        	nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=p),
        	nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.net(x)
        
class UpsampleBLock(nn.Module):
	def __init__(self, in_channels, scaleFactor, k=3, p=1):
		super(UpsampleBLock, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, in_channels * (scaleFactor ** 2), kernel_size=k, padding=p),
			nn.PixelShuffle(scaleFactor),
			nn.PReLU()
		)
	
	def forward(self, x):
		return self.net(x)
        
class Generator(nn.Module):
    def __init__(self, n_residual=8):
        super(Generator, self).__init__()
        self.n_residual = n_residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        for i in range(n_residual):
            self.add_module('residual' + str(i+1), ResidualBlock(64, 64))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
        self.upsample = nn.Sequential(
        	UpsampleBLock(64, 2),
        	UpsampleBLock(64, 2),
        	nn.Conv2d(64, 3, kernel_size=9, padding=4)
        )

    def forward(self, x):
        #print ('G input size :' + str(x.size()))
        y = self.conv1(x)
        cache = y.clone()
        
        for i in range(self.n_residual):
            y = self.__getattr__('residual' + str(i+1))(y)
            
        y = self.conv2(y)
        y = self.upsample(y + cache)
        #print ('G output size :' + str(y.size()))
        return (torch.tanh(y) + 1.0) / 2.0
    
class Discriminator(nn.Module):
	def __init__(self, l=0.2):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(l),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(l),

			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(l),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(l),

			nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(l),

			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(l),

			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(l),

			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(512, 1024, kernel_size=1),
			nn.LeakyReLU(l),
			nn.Conv2d(1024, 1, kernel_size=1)
		)

	def forward(self, x): 
		#print ('D input size :' +  str(x.size()))
		y = self.net(x)
		#print ('D output size :' +  str(y.size()))
		si = torch.sigmoid(y).view(y.size()[0])
		#print ('D output : ' + str(si))
		return si
		
class Discriminator_WGAN(nn.Module):
	def __init__(self, l=0.2):
		super(Discriminator_WGAN, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(l),

			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(512, 1024, kernel_size=1),
			nn.LeakyReLU(l),
			nn.Conv2d(1024, 1, kernel_size=1)
		)

	def forward(self, x): 
		#print ('D input size :' +  str(x.size()))
		y = self.net(x)
		#print ('D output size :' +  str(y.size()))
		return y.view(y.size()[0])

def compute_gradient_penalty(D, real_samples, fake_samples):
	alpha = torch.randn(real_samples.size(0), 1, 1, 1)
	if torch.cuda.is_available():
		alpha = alpha.cuda()
		
	interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
	d_interpolates = D(interpolates)
	fake = torch.ones(d_interpolates.size())
	if torch.cuda.is_available():
		fake = fake.cuda()
		
	gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty		

import numpy as np
import torch

import os
from os import listdir
from os.path import join

from PIL import Image

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision.utils as utils
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def to_image():
    return Compose([
        ToPILImage(),
        ToTensor()
    ])
	
class TrainDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_preprocess = Compose([CenterCrop(384), RandomCrop(crop_size), ToTensor()])
        self.lr_preprocess = Compose([ToPILImage(), Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC), ToTensor()])

    def __getitem__(self, index):
        hr_image = self.hr_preprocess(Image.open(self.image_filenames[index]))
        lr_image = self.lr_preprocess(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)
        
class DevDataset(Dataset):
	def __init__(self, dataset_dir, upscale_factor):
		super(DevDataset, self).__init__()
		self.upscale_factor = upscale_factor
		self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

	def __getitem__(self, index):
		hr_image = Image.open(self.image_filenames[index])
		crop_size = calculate_valid_crop_size(128, self.upscale_factor)
		lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
		hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
		hr_image = CenterCrop(crop_size)(hr_image)
		lr_image = lr_scale(hr_image)
		hr_restore_img = hr_scale(lr_image)
		norm = ToTensor()
		return norm(lr_image), norm(hr_restore_img), norm(hr_image)

	def __len__(self):
		return len(self.image_filenames)

def print_first_parameter(net):	
	for name, param in net.named_parameters():
		if param.requires_grad:
			print (str(name) + ':' + str(param.data[0]))
			return

def check_grads(model, model_name):
	grads = []
	for p in model.parameters():
		if not p.grad is None:
			grads.append(float(p.grad.mean()))

	grads = np.array(grads)
	if grads.any() and grads.mean() > 100:
		print('WARNING!' + model_name + ' gradients mean is over 100.')
		return False
	if grads.any() and grads.max() > 100:
		print('WARNING!' + model_name + ' gradients max is over 100.')
		return False
		
	return True

def get_grads_D(net):
	top = 0
	bottom = 0
	for name, param in net.named_parameters():
		if param.requires_grad:
			# Hardcoded param name, subject to change of the network
			if name == 'net.0.weight':
				top = param.grad.abs().mean()
				#print (name + str(param.grad))
			# Hardcoded param name, subject to change of the network
			if name == 'net.26.weight':
				bottom = param.grad.abs().mean()
				#print (name + str(param.grad))
	return top, bottom
	
def get_grads_D_WAN(net):
	top = 0
	bottom = 0
	for name, param in net.named_parameters():
		if param.requires_grad:
			# Hardcoded param name, subject to change of the network
			if name == 'net.0.weight':
				top = param.grad.abs().mean()
				#print (name + str(param.grad))
			# Hardcoded param name, subject to change of the network
			if name == 'net.19.weight':
				bottom = param.grad.abs().mean()
				#print (name + str(param.grad))
	return top, bottom

def get_grads_G(net):
	top = 0
	bottom = 0
	#torch.set_printoptions(precision=10)
	#torch.set_printoptions(threshold=50000)
	for name, param in net.named_parameters():
		if param.requires_grad:
			# Hardcoded param name, subject to change of the network
			if name == 'conv1.0.weight':
				top = param.grad.abs().mean()
				#print (name + str(param.grad))
			# Hardcoded param name, subject to change of the network
			if name == 'upsample.2.weight':
				bottom = param.grad.abs().mean()
				#print (name + str(param.grad))
	return top, bottom

import torch