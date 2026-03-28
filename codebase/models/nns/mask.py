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
		# init 96*96
		self.conv1 = torch.nn.Conv2d(channel, 32, 4, 2, 1) # 48*48
		self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1, bias=False) # 24*24
		self.conv3 = torch.nn.Conv2d(64, 1, 4, 2, 1, bias=False) # 12*12
   
		self.LReLU = torch.nn.LeakyReLU(0.2, inplace=True)
		self.convm = torch.nn.Conv2d(1, 1, 4, 2, 1) # 6*6
		self.convv = torch.nn.Conv2d(1, 1, 4, 2, 1) # 6*6
		self.mean_layer = nn.Sequential(
			torch.nn.Linear(6*6, z_dim)
			)
		self.var_layer = nn.Sequential(
			torch.nn.Linear(6*6, z_dim)
			)
		# self.fc1 = torch.nn.Linear(6*6*128, 512)

	def encode(self, x):
		x = self.LReLU(self.conv1(x))
		x = self.LReLU(self.conv2(x))
		x = self.LReLU(self.conv3(x))
		#x = self.LReLU(self.conv4(x))
		#print(x.size())
		hm = self.convm(x)
		#print(hm.size())
		hm = hm.view(-1, 6*6)
		hv = self.convv(x)
		hv = hv.view(-1, 6*6)
		mu, var = self.mean_layer(hm), self.var_layer(hv)
		var = F.softplus(var) + 1e-8
		#var = torch.reshape(var, [-1, 16, 16])
		#print(mu.size())
		return  mu, var
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

class Encoder(ConvEncoder):
	def __init__(self, z_dim, channel=3, y_dim=4):
		super().__init__(z_dim, channel)
   
   
class Decoder_DAG(nn.Module):
	def __init__(self, z_dim, concept, z1_dim, channel=3, y_dim=0):
		super().__init__()
		self.z_dim = z_dim
		self.z1_dim = z1_dim
		self.concept = concept
		self.y_dim = y_dim
		self.channel = channel
		# Convolutional decoders for each concept
		self.net1 = nn.Sequential(
			nn.Conv2d(z1_dim, 128, 1),
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
		self.net2 = nn.Sequential(
			nn.Conv2d(z1_dim, 128, 1),
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
		self.net3 = nn.Sequential(
			nn.Conv2d(z1_dim, 128, 1),
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
		self.net4 = nn.Sequential(
			nn.Conv2d(z1_dim, 128, 1),
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
		self.net5 = nn.Sequential(
			nn.Conv2d(z1_dim, 128, 1),
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
		# Full decoder for all z_dim
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

	def _reshape_to_conv(self, z, z1_dim):
		"""Reshape latent vector to 4D tensor for conv layers."""
		return z.view(-1, z1_dim, 1, 1)

	def _decode_concept(self, z, decoder_net):
		"""Decode a single concept using its decoder."""
		z_reshaped = self._reshape_to_conv(z, self.z1_dim)
		out = decoder_net(z_reshaped)
		# Resize to target image size (96x96)
		out = F.interpolate(out, size=(96, 96), mode='bilinear', align_corners=False)
		return out

	def decode_sep(self, z, u, y=None):
		"""Decode with concept separation using sum instead of averaging.
		Sum preserves more signal and forces each concept decoder to learn meaningful patterns.
		"""
		z = z.view(-1, self.concept * self.z1_dim)
		zy = z if y is None else torch.cat((z, y), dim=1)
		
		if self.z1_dim == 1:
			zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
			if self.concept == 4:
				zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
			elif self.concept == 3:
				zy1, zy2, zy3 = zy[:, 0], zy[:, 1], zy[:, 2]
		else:
			if self.concept == 4:
				zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
			elif self.concept == 3:
				zy1, zy2, zy3 = torch.split(zy, self.z_dim // self.concept, dim=1)
		
		rx1 = self._decode_concept(zy1, self.net1)
		rx2 = self._decode_concept(zy2, self.net2)
		rx3 = self._decode_concept(zy3, self.net3)
		
		if self.concept == 4:
			rx4 = self._decode_concept(zy4, self.net4)
			# Sum concept decoders and normalize properly
			h = (rx1 + rx2 + rx3 + rx4) / 4.0  # Normalize to average pixel range
		elif self.concept == 3:
			h = (rx1 + rx2 + rx3) / 3.0
		
		# Flatten output for compatibility with VAE loss computation
		h = h.view(h.size(0), -1)
		return h, h, h, h, h

	def decode(self, z, u, y=None):
		z = z.view(-1, self.concept * self.z1_dim)
		z_reshaped = self._reshape_to_conv(z, self.z_dim)
		h = self.net6(z_reshaped)
		# Resize to target image size (96x96)
		h = F.interpolate(h, size=(96, 96), mode='bilinear', align_corners=False)
		h = h.view(h.size(0), -1)
		return h, h, h, h, h