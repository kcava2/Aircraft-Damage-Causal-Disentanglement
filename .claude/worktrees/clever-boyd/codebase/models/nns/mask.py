#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software;
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    
def dag_right_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret
    
def dag_left_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = weight.matmul(input)
        if bias is not None:
            output += bias
        ret = output
    return ret



class MaskLayer(nn.Module):
	def __init__(self, z_dim, concept=4, z2_dim=4):
		super().__init__()
		self.z_dim = z_dim
		self.z2_dim = z2_dim
		self.concept = concept
		
		self.elu = nn.ELU()
		self.net1 = nn.Sequential(
			nn.Linear(z2_dim , 32),
			nn.ELU(),
			nn.Linear(32, z2_dim),
		)
		self.net2 = nn.Sequential(
			nn.Linear(z2_dim , 32),
			nn.ELU(),
			nn.Linear(32, z2_dim),
		)
		self.net3 = nn.Sequential(
			nn.Linear(z2_dim , 32),
			nn.ELU(),
		  nn.Linear(32, z2_dim),
		)
		self.net4 = nn.Sequential(
			nn.Linear(z2_dim , 32),
			nn.ELU(),
			nn.Linear(32, z2_dim)
		)
		self.net5 = nn.Sequential(
			nn.Linear(z2_dim , 32),
			nn.ELU(),
			nn.Linear(32, z2_dim)
		)
		self.net = nn.Sequential(
			nn.Linear(z2_dim , 32),
			nn.ELU(),
			nn.Linear(32, z2_dim),
		)
	def masked(self, z):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z
   
	def masked_sep(self, z):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z
   
	def mix(self, z):
		zy = z.view(-1, self.concept*self.z2_dim)
		if self.z2_dim == 1:
			zy = zy.reshape(zy.size()[0],zy.size()[1],1)
			if self.concept ==5:
				zy1, zy2, zy3, zy4, zy5 = zy[:,0], zy[:,1], zy[:,2], zy[:,3], zy[:,4]
			elif self.concept ==4:
				zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
			elif self.concept ==3:
				zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
		else:
			if self.concept == 5:
				zy1, zy2, zy3, zy4, zy5 = torch.split(zy, self.z_dim // self.concept, dim=1)
			elif self.concept == 4:
				zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
			elif self.concept == 3:
				zy1, zy2, zy3 = torch.split(zy, self.z_dim // self.concept, dim=1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		if self.concept == 5:
			rx4 = self.net4(zy4)
			rx5 = self.net5(zy5)
			h = torch.cat((rx1, rx2, rx3, rx4, rx5), dim=1)
		elif self.concept == 4:
			rx4 = self.net4(zy4)
			h = torch.cat((rx1, rx2, rx3, rx4), dim=1)
		elif self.concept == 3:
			h = torch.cat((rx1, rx2, rx3), dim=1)
		#print(h.size())
		return h


class Attention(nn.Module):
  def __init__(self, in_features, bias=False):
    super().__init__()
    self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
    self.sigmd = torch.nn.Sigmoid()
    #self.M =  nn.Parameter(torch.zeros(in_features,in_features))
    #self.A = torch.zeros(in_features,in_features).to(device)
    
  def attention(self, z, e):
    a = z.matmul(self.M).matmul(e.permute(0,2,1))
    a = self.sigmd(a)
    #print(self.M)
    A = torch.softmax(a, dim = 1)
    e = torch.matmul(A,e)
    return e, A
  def __init__(self, in_features, bias=False):
    super().__init__()
    self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
    self.sigmd = torch.nn.Sigmoid()
    #self.M =  nn.Parameter(torch.zeros(in_features,in_features))
    #self.A = torch.zeros(in_features,in_features).to(device)
    
  def attention(self, z, e):
    a = z.matmul(self.M).matmul(e.permute(0,2,1))
    a = self.sigmd(a)
    #print(self.M)
    A = torch.softmax(a, dim = 1)
    e = torch.matmul(A,e)
    return e, A
    
class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features,i = False, bias=False, initial=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features,out_features)
        self.a = self.a
        if initial:
            self.a[0][1], self.a[0][2], self.a[0][3] = 1,1,1
            self.a[1][2], self.a[1][3] = 1,1

        self.A = nn.Parameter(self.a)
        
        self.b = torch.eye(out_features)
        self.b = self.b
        self.B = nn.Parameter(self.b)
        
        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad=False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
    def mask_z(self,x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = torch.matmul(self.B.t(), x)
        return x
        
    def mask_u(self,x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(self.B.t(), x)
        return x
        
    def calculate_dag(self, x, v):
        #print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias) 
        #print(x.size())
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v
      
class ConvEncoder(nn.Module):
	def __init__(self, z_dim, channel):
		super().__init__()
		self.z_dim = z_dim
		self.channel = channel
		self.conv1 = torch.nn.Conv2d(channel, 32, 4, 2, 1)   # 96→48, 32ch
		self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1)         # 48→24, 64ch
		self.conv3 = torch.nn.Conv2d(64, 128, 4, 2, 1)        # 24→12, 128ch  (skip s3)
		self.conv4 = torch.nn.Conv2d(128, 256, 4, 2, 1)       # 12→6,  256ch
		self.LReLU = torch.nn.LeakyReLU(0.2, inplace=True)
		self.bottleneck = torch.nn.Conv2d(256, 32, 1)          # compress channels before linear
		self.mean_layer = nn.Sequential(torch.nn.Linear(32 * 6 * 6, z_dim))
		self.var_layer  = nn.Sequential(torch.nn.Linear(32 * 6 * 6, z_dim))

	def encode(self, x):
		s1 = self.LReLU(self.conv1(x))    # 48×48, 32ch  — skip for decoder level 3
		s2 = self.LReLU(self.conv2(s1))   # 24×24, 64ch  — skip for decoder level 2
		s3 = self.LReLU(self.conv3(s2))   # 12×12, 128ch — skip for decoder level 1
		h  = self.LReLU(self.conv4(s3))   # 6×6,  256ch
		hb = self.LReLU(self.bottleneck(h)).view(-1, 32 * 6 * 6)
		mu  = self.mean_layer(hb)
		var = self.var_layer(hb)
		var = F.softplus(var) + 1e-8
		return mu, var, (s1, s2, s3)


class ConvDecoder(nn.Module):
	def __init__(self, z_dim, channel):
		super().__init__()
		self.z_dim = z_dim
		self.channel = channel

		self.net6 = nn.Sequential(
				nn.Conv2d(z_dim, 128, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(128, 64, 4),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(64, 64, 4, 2, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(64, 32, 4, 2, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(32, 32, 4, 2, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(32, 32, 4, 2, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(32, channel, 4, 2, 1)
		)

	def decode_sep(self,x):
		x = x.view(-1, self.z_dim, 1, 1)
		return self.net6(x).view(x.size(0), -1), None, None, None, None


class UNetDecoder(nn.Module):
	"""U-Net style decoder with skip connections from the encoder.

	Spatial progression (matching ConvEncoder):
	  latent z  →  6×6  →  12×12  →  24×24  →  48×48  →  96×96
	Skip connections concatenated at 12×12 (s3, 128ch), 24×24 (s2, 64ch),
	and 48×48 (s1, 32ch).  When skips=None (e.g. pure sampling), zero
	tensors are used so the decoder can still run.
	"""
	def __init__(self, z_dim, channel=3):
		super().__init__()
		# 1×1 → 6×6
		self.proj  = nn.ConvTranspose2d(z_dim, 256, 6)
		# 6×6 → 12×12
		self.up1   = nn.ConvTranspose2d(256, 128, 4, 2, 1)
		self.cat1  = nn.Sequential(
			nn.Conv2d(128 + 128, 128, 3, 1, 1),
			nn.BatchNorm2d(128), nn.ReLU(inplace=True))
		# 12×12 → 24×24
		self.up2   = nn.ConvTranspose2d(128, 64, 4, 2, 1)
		self.cat2  = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 3, 1, 1),
			nn.BatchNorm2d(64), nn.ReLU(inplace=True))
		# 24×24 → 48×48
		self.up3   = nn.ConvTranspose2d(64, 32, 4, 2, 1)
		self.cat3  = nn.Sequential(
			nn.Conv2d(32 + 32, 32, 3, 1, 1),
			nn.BatchNorm2d(32), nn.ReLU(inplace=True))
		# 48×48 → 96×96
		self.up4   = nn.ConvTranspose2d(32, 16, 4, 2, 1)
		self.final = nn.Conv2d(16, channel, 1)

	def forward(self, z, skips=None):
		b, dev = z.size(0), z.device
		x = z.view(b, -1, 1, 1)
		x = F.leaky_relu(self.proj(x), 0.2)   # 6×6, 256ch
		x = F.leaky_relu(self.up1(x), 0.2)    # 12×12, 128ch
		s3 = skips[2] if skips is not None else torch.zeros(b, 128, 12, 12, device=dev)
		x = self.cat1(torch.cat([x, s3], dim=1))
		x = F.leaky_relu(self.up2(x), 0.2)    # 24×24, 64ch
		s2 = skips[1] if skips is not None else torch.zeros(b, 64, 24, 24, device=dev)
		x = self.cat2(torch.cat([x, s2], dim=1))
		x = F.leaky_relu(self.up3(x), 0.2)    # 48×48, 32ch
		s1 = skips[0] if skips is not None else torch.zeros(b, 32, 48, 48, device=dev)
		x = self.cat3(torch.cat([x, s1], dim=1))
		x = F.leaky_relu(self.up4(x), 0.2)    # 96×96, 16ch
		return self.final(x)                    # 96×96, channel ch

class Encoder(ConvEncoder):
	def __init__(self, z_dim, channel=3, y_dim=4):
		super().__init__(z_dim, channel)


class Decoder_DAG(nn.Module):
	"""U-Net decoder with skip connections from the paired ConvEncoder.

	Concept disentanglement is enforced at the latent level (DAG + KL losses).
	A single shared UNetDecoder reconstructs from the full z while using
	encoder skip features for high-frequency spatial detail.
	"""
	def __init__(self, z_dim, concept, z1_dim, channel=3, y_dim=0):
		super().__init__()
		self.z_dim   = z_dim
		self.z1_dim  = z1_dim
		self.concept = concept
		self.channel = channel
		self.unet    = UNetDecoder(z_dim, channel)

	def decode_sep(self, z, u, skips=None, y=None):
		z   = z.view(-1, self.z_dim)
		out = self.unet(z, skips)           # (batch, channel, 96, 96)
		out_flat = out.view(out.size(0), -1)
		return out_flat, out_flat, out_flat, out_flat, out_flat

	def decode(self, z, u=None, skips=None, y=None):
		return self.decode_sep(z, u, skips, y)