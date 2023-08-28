from PIL import Image
from astropy.time import Time
import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
import astropy.constants as C
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time
from multiprocessing import Process, Manager, Value, Array
import serial
from time import sleep
import decawave_ble
import logging
import time
import tenacity
from astropy.io import ascii
from tqdm import tqdm
import pickle
import os
import sys
from matplotlib import colors,cm
import itertools
import csv
import ast
from serial.tools.list_ports import comports
import glob
import operator
import itertools


def vprint(*args,verbose = True):
	if verbose:
		print(*args)
	else:
		pass

def norm(x):
	return (x - np.min(x))/(np.max(x) - np.min(x))

class Observatory:
	def __init__(self,name="ALMA",latitude =-23.0229*u.deg,longitude = -67.755 * u.deg,ant_pos_EW =[],ant_pos_NS =[],ant_pos_bit_file='./ant_pos.2.txt'):
		""" 
		Observatory class
		=========================
		Description:
		Handles everything related to the telescope setup, such as its position in longitude, latitude, relative positions of antennas.
		Parameters:
		-------------------------
		latitude : float in astropy angle units
			the latitude of the observatory, default is ALMA -23.0229 deg
		longitude : float in astropy angle units
			the longitude of the observatory, default is ALMA 84.6603 deg
		ant_pos_EW : list of float
			antenna position in offsets in the EW direction from a reference antenna, default is empty list, uses meters
		ant_pos_NW : list of float 
			antenna position in offsets in the NS direction from a reference antenna, default is empty list, uses meters
		ant_pos_bit_file : str 
			path to antenna position file where antenna offsets are mapped to bit positions from the usb connection, default is to use ./ant_pos.txt

		Attributes:
		-------------------------
		latitude : float in astropy angle units
			the latitude of the observatory, default is ALMA -23.0229 deg
		longitude : float in astropy angle units
			the longitude of the observatory, default is ALMA -673.755 deg
		ant_pos_EW : list of float
			antenna position in offsets in the EW direction from a reference antenna, default is empty list, uses meters
		ant_pos_NW : list of float 
			antenna position in offsets in the NS direction from a reference antenna, default is empty list, uses meters
		ant_pos_bit_file : str 
			path to antenna position file where antenna offsets are mapped to bit positions from the usb connection, default is to use ./ant_pos.txt
		baselines_BX : 2d numpy array in astropy meters unit 
			square matrix containing the baselines in X cartesian coordinates,
			where BX[i,j] is the distance between antenna i and antenna j
		baselines_BY : 2d numpy array in astropy meters unit 
			square matrix containing the baselines in Y cartesian coordinates,
			where BY[i,j] is the distance between antenna i and antenna j
		baselines_BZ : 2d numpy array in astropy meters unit 
			square matrix containing the baselines in Z cartesian coordinates,
			where BZ[i,j] is the distance between antenna i and antenna j
		"""

		self.latitude = latitude
		self.longitude = longitude
		self.ant_pos_EW = ant_pos_EW
		self.ant_pos_NS = ant_pos_NS
		self.ant_pos_X = -1*self.ant_pos_NS * np.sin(latitude)
		self.ant_pos_Y  = self.ant_pos_EW
		self.ant_pos_Z = self.ant_pos_NS * np.cos(latitude)
		self.ant_pos_bit_file = ant_pos_bit_file
#	def set_ant_pos(self,EW,NS):
#		self.ant_pos_EW = EW
#		self.ant_pos_NS = NS
	def load_ant_pos_bit_file(self,ant_pos_bit_file=None,center_coords = (115./2,115./2),scale_factor=2):
		"""Description
		===============
		Function that constructs a dictionary with keys being bit-positions read from the USB, 
		and values being the offsets from reference antenna, using a file.
		Arguments
		---------------
		ant_post_bit_file : str, default = None
			position file to use, default is None. If None, uses the class attribute of the same name
		center_coords : bool, default = True
			if True, center the coordinates of the antennas by subtracting the mean position on each axis.
		Returns
		---------------
		None
		"""
		if ant_pos_bit_file == None:
			ant_file_data = ascii.read(self.ant_pos_bit_file)
			self._ant_bit_dict = dict(zip(ant_file_data['bit'],
				zip((ant_file_data['posx'] - center_coords[0]) *scale_factor,
					-(ant_file_data['posy'] - center_coords[1]) *scale_factor
					)
				)
				)
		else:
			ant_file_data = ascii.read(ant_pos_bit_file)
			self._ant_bit_dict = dict(zip(ant_file_data['bit'],zip((ant_file_data['posx'] - center_coords[0]) *scale_factor,(ant_file_data['posy'] - center_coords[1])*scale_factor )))
	def set_read_source_ids(self,src_list):
		"""Description
		===============
		Sets the connection names for getting antenna positions. 
		Arguments
		---------------
		src_list : list of strings
			Contains the path to the USB connection for the 
			antenna positions (e.g. '/dev/ttyUSB0') and the
			names of the BLE tags (e.g. 'DW12345')

		"""
		self._read_source_list = src_list
		
	def set_ant_pos(self,measurements_dict):
		pos = [[],[]]
		try:
			for src in self._read_source_list:
				if 'USB' in src:
					bit_array = measurements_dict[src]
					pos = np.hstack([pos,np.array([np.array(self._ant_bit_dict[bit]) for bit in bit_array]).T])
				if 'DW' in src:
					try:
						pos = np.hstack([pos,self.apply_transform(measurements_dict[src][:-1])[np.newaxis].T])
					except Exception as e:
						print("set_ant_pos",e)
						pass
		except Exception as e:
			print("error",e)
			pass
		self.ant_pos_EW,self.ant_pos_NS = pos
		self.ant_pos_X = -1*self.ant_pos_NS * np.sin(self.latitude)
		self.ant_pos_Y  = self.ant_pos_EW
		self.ant_pos_Z = self.ant_pos_NS * np.cos(self.latitude)

	def set_3pos_tag_single(self,measurements_dict,tag_id):
		X_vecs = []
		print("Aligning tags with ALMA")
		for ii in ['red','orange','green']:
			input("Place tag "+tag_id+" at the "+ (ii) +" pad and press Enter")
			temp_vec = []
			data_in = None
			with tqdm(total = 40,desc="Collecting position data", unit="tag positions") as pbar:
				while len(temp_vec) < 40:
					sleep(0.2)
					try:
						data_in = measurements_dict[tag_id][:-1]
						if data_in is not None:
							#print(data_in)
							temp_vec.append(data_in)
							pbar.update(1)
					except KeyError as e:
						pass
						
			temp_vec = np.median(temp_vec,axis=0)
			X_vecs.append(temp_vec)
		print(X_vecs)
		self.X_vecs = X_vecs
	def set_3pos_tag_all(self,measurements_dict,tag_ids):
		X_vecs = []
		print("Aligning tags with ALMA")
		input("Place tag "+str(tag_ids)+" at the red, orange, and green pads, respectively, and press Enter")
		temp_vec = []
		data_in = None
		with tqdm(total = 40,desc="Collecting position data", unit="tag positions") as pbar:
			while len(temp_vec) < 40:
				sleep(0.5)
				try:	
					data_in = np.array([measurements_dict[tag_id][:-1] for tag_id in tag_ids])
					if data_in is not None and len(data_in) == 3:
						print(data_in)
						temp_vec.append(data_in)
						pbar.update(1)
				except KeyError as e:
					pass
					
		temp_vec = np.median(temp_vec,axis=0)
		print("shape of vec",temp_vec.shape)
		X_vecs = temp_vec
		print(X_vecs)
		self.X_vecs = X_vecs

	
	def set_transform(self,X=None):
		if X==None:
			X = self.X_vecs
			x1,x2,x3 = X
		else:
			x1,x2,x3 = X
		try:
			X12 = np.array([x1,x2]).T
			X23 = np.array([x2,x3]).T
			Y12 = np.array([[-52.5,-32.0],[53.0,-48.5]]).T
			Y23 = np.array([[53,-48.5],[3.5,10.0]]).T
			A = np.matmul((Y12-Y23),np.linalg.inv(X12-X23))
			B = Y12 - np.matmul(A,X12)
			self.A=A
			self.b=B.T[0]
		except Exception as e:
			print("can't get matrix transform:",e)
			
	def transform_query(self,measurements_dict,tag_id):
		ans = input("Perform tag to ALMA position calibration?(y) load from disk? (l) don't use tags! (n) quit (q)")
		if ans == "y" and type(tag_id) == str:
			self.set_3pos_tag_single(measurements_dict,tag_id)
			self.set_transform()
			with open("transM.pickle",'wb') as fout:
				pickle.dump([self.A,self.b],fout)
			self.vlbi_mode = True
		if ans == "y" and type(tag_id) == list:
			self.set_3pos_tag_all(measurements_dict,tag_id)
			self.set_transform()
			with open("transM.pickle",'wb') as fout:
				pickle.dump([self.A,self.b],fout)
			self.vlbi_mode = True

		elif ans == "l":
			try: 
				with open("transM.pickle",'rb') as fin:
					self.A,self.b = pickle.load(fin)
				self.vlbi_mode = True
			except Exception as e:
				print(e,"File not loaded!")
				self.vlbi_mode = False
		elif ans == "n":
			self.vlbi_mode = False
		elif ans == "q":
			self.vlbi_mode = False
			#quit()
	def apply_transform(self,x,A=None,b=None):
		if A == None:
			A = self.A
		if b == None:
			b = self.b
		return A@x + b


	def set_raw_ant_pos_from_ble(self,ble_data):
		self.ant_pos_ble = np.array([pos for pos in ble_data.values()]).T
	#def merge_ant_pos(self):
	#	self.ant_pos = np.hstack([self.ant_pos_ser,self.ant_pos_ble])
	#	self.ant_pos_EW = self.ant_pos[0]
	#	self.ant_pos_NS = self.ant_pos[1]
		
	def make_baselines(self):
		nant = len(self.ant_pos_X)	
		res = np.zeros((3,nant,nant))
		for i in range(nant):
			for j in range(nant):
				res[0,i,j] = self.ant_pos_X[i] - self.ant_pos_X[j]
				res[1,i,j] = self.ant_pos_Y[i] - self.ant_pos_Y[j]
				res[2,i,j] = self.ant_pos_Z[i] - self.ant_pos_Z[j]
		self.BX,self.BY,self.BZ = res * u.m

		
	def get_baselines(self):
		try:
			return self.BX,self.BY,self.BZ
		except Exception as e:
			print(e)
			return None

class SkyImage:
	def __init__(self,path="./models/",filename="galaxy_lobes.png",declination = -90 * u.deg,pixel_size = 0.50 * u.arcsec):
		self.path = path
		self.filename = filename
		self.pixel_size = pixel_size
		self.declination = declination
	def load_image(self,path = None,filename = None,new_size = (480,480)):
		"""Load image data, for ease of use with real fourier transforms we make the sizes of the images, in pixels, odd.
		"""
		if path is None: path = self.path
		if filename is None: filename = self.filename
		try:
			im = Image.open(path+filename)
			im.thumbnail(new_size)
			self.data = np.flipud(np.asarray(im.convert("L")))
			#make odd sizes by padding with zeros
			if self.data.shape[1] % 2 == 0:
				self.data=np.hstack([self.data,[[0],]*self.data.shape[0]])
			if self.data.shape[0] % 2 == 0:
				self.data=np.vstack([self.data,[0,]*self.data.shape[1]])
				
			self.imsize_Y,self.imsize_X = self.data.shape
		except Exception as e:
			print(e,self.path,self.filename,path,filename,"imageerror.")
	def make_invert(self):
		"""Fourier transforms of the image using the real fft"""

		self.fft_data_r = np.fft.rfft2(self.data)
		self.fft_data = np.zeros(self.data.shape)+0.j
		#since we use the real fft, we only need expand the data to build the full fft,
		#knowing that the fft or a real valued array is Hermitian-symmetric
		#this is done so we can plot it nicely when needed
		self.fft_data[:,:self.fft_data_r.shape[1]] = self.fft_data_r
		self.fft_data[:,self.fft_data_r.shape[1]:] = np.conjugate(np.roll(self.fft_data_r[::-1,:0:-1],1,axis=0)) 
		self.fft_data = np.fft.fft2(self.data)
		self.fftsize_X,self.fftsize_Y = self.fft_data.shape
	def select_image(self,path = None,filename = None,new_size = (480,480)):
		self.load_image(path = path,filename = filename,new_size = new_size)
		self.make_invert()
class Observation:
	def __init__(self,Observatory,SkyImage,Control,obs_frequency = 180*u.GHz,HA_START=Angle(-4,u.hour),HA_END=Angle(4,u.hour),sample_freq=0.05*u.Hz,EL_LIMIT = 10 * u.deg,var_dic = None):
		self.obs_frequency = obs_frequency
		self.obs_lam = C.c / obs_frequency
		self._buf_obs_frequency = obs_frequency
		self.HA_START = HA_START
		self.HA_END = HA_END
		self._buf_HA_START = HA_START
		self._buf_HA_END = HA_END
			
		self._buf_declination = 90*u.deg
		self.sample_freq = sample_freq
		self._buf_sample_freq = sample_freq
		self._buf_image_filename = None
		self.Observatory = Observatory
		self.SkyImage = SkyImage
		self.SkyImage.make_invert()
		self.N_samples = int(((self.HA_END - self.HA_START).hour * u.hour * sample_freq ).decompose() ) + 1
		self.EL_LIMIT = EL_LIMIT
		self.Control = Control
		self.var_dic = var_dic
	def set_read_source_ids(self,src_list):
		"""Set which USB connection will be used for reading the commands (control) for the observation parameters
		"""
		self.Control.ctrl_id = src_list[0]
	
	
	def set_var_dic(self,var_dic):
		self.var_dic = var_dic
	def _return_val_button(self,but_pos,var_dic):
		r0,r1,r2,s1,s2,s3 = tuple([but_pos[_bid].get_state() for _bid in self.Control.button_ids])
		varname = var_dic[r0][0][r1]
		value = var_dic[r0][1][r1][0](var_dic[r0][1][r1][1],var_dic[r0][2][r1][r2])
		print("ret val",r0,r1,r2,varname,value)
		return varname, value
	
	def update_obs_from_control(self,measurements_dict):
		try:
			but_pos = self.Control.get_state(last_measurements)
			print("but pos", but_pos,tuple([but_pos[_bid].get_state() for _bid in self.Control.button_ids])
)
			varname,value = self._return_val_button(but_pos,self.var_dic)
			print("setting: ",varname,value)
		except Exception as e:
			print(e,"Error in button readout, skipping.")
			return 0
		if varname == "HA":
			self._buf_HA_END =  Angle(value, u.hour) + self._buf_HA_END - self._buf_HA_START
			self._buf_HA_START = Angle(value, u.hour)
			if but_pos['S0'].get_state() == 1:
				self.HA_END = self._buf_HA_END
				self.HA_START = self._buf_HA_START

		if varname == "Int":
			if value >= 12:
				self._buf_HA_START =  Angle(-6.,u.hour)
				self._buf_HA_END = Angle(+6,u.hour)
			else:
				self._buf_HA_END = self._buf_HA_START + Angle(value, u.hour)
			if but_pos['S0'].get_state() == 1:
				self.HA_END = self._buf_HA_END
				self.HA_START = self._buf_HA_START
		
		if varname == "Dec":
			self._buf_declination = value * u.deg
			if but_pos['S0'].get_state() == 1:
				self.SkyImage.declination = self._buf_declination
		if varname == "obs_frequency":
			self._buf_obs_frequency = value * u.GHz
			if but_pos['S0'].get_state() == 1:
				self.obs_frequency = self._buf_obs_frequency
				self.obs_lam = C.c/self.obs_frequency
		if varname == "Fname":
			self._buf_image_filename = value
			if but_pos['S0'].get_state() == 1:
				self.SkyImage.select_image(filename = self._buf_image_filename)
		if but_pos['S1'].get_state() == 1:
			self.HA_END = self._buf_HA_END
			self.HA_START = self._buf_HA_START
			self.SkyImage.declination = self._buf_declination
			self.obs_frequency = self._buf_obs_frequency
			self.obs_lam = C.c/self.obs_frequency
			self.SkyImage.select_image(filename = self._buf_image_filename)
		
	def update_obs(self,measurements_dict = None,obs_frequency = 180*u.GHz,HA_START=Angle(-4,u.hour),HA_END=Angle(4,u.hour),sample_freq=0.05*u.Hz,EL_LIMIT = 10 * u.deg):
		self.obs_frequency = obs_frequency
		self.obs_lam = C.c / obs_frequency
		self.sample_freq = sample_freq
		self.HA_START = HA_START
		self.HA_END = HA_END
		self.N_samples = int(((self.HA_END - self.HA_START).hour * u.hour * sample_freq ).decompose() ) + 1
		self.EL_LIMIT = EL_LIMIT
		#select some other config based on measurements dict
		if measurements_dict is not None:
			self._set_ctrl_state(measurements_dict[self._read_source_id])


	def calc_el_curve(self):
		"""Calculate elevation above the horizon, given Observatory position and SkyImage (i.e. object) declimnation. 
		Store in array HA_arr only the HA values to which correspond elevation values, above the lower elevation limit EL_LIMIT in degrees"""
		self.HA_arr = np.linspace(self.HA_START,self.HA_END,self.N_samples)
		elev = np.degrees((np.sin(self.Observatory.latitude) * np.sin(self.SkyImage.declination) + np.cos(self.Observatory.latitude) * np.cos(self.SkyImage.declination) * np.cos(self.HA_arr))*u.radian) 
		self.HA_arr = self.HA_arr[elev.to(u.deg) > self.EL_LIMIT]
		self.cos_ha = np.cos(self.HA_arr)
		self.sin_ha = np.sin(self.HA_arr)
		self.sin_cos_ha = np.array([self.sin_ha,self.cos_ha]).T
		#print(elev)

	def make_uv_coverage(self):
		"""Calculate the UV coverage using Observatory coordinates, baselines BX and BY, SkyImage (i.e. object) declination, HA values during observation.
		This function calculates U and V visibility values as skew-symmetric matrices. Element u_ij = - u_ji and v_ij = - v_ji, where i and j represent antenna i and j, repsectively.	
		This is the result of introducing the 180 degrees (pi) phase shift for each antenna pair, as in a correlating interferometer.

		The returned values are 3d arrays which are the U and V matrices at each time t (corresponding to a single HA value from HA_arr). 
		"""
		self.U = (np.array([self.Observatory.BX.value * sinha + self.Observatory.BY.value * cosha for sinha,cosha in self.sin_cos_ha]) * self.Observatory.BX.unit /  self.obs_lam).decompose() *1./u.radian

		self.V = (np.array([-self.Observatory.BX.value * np.sin(self.SkyImage.declination) * cosha + self.Observatory.BY.value * np.sin(self.SkyImage.declination) * sinha + self.Observatory.BZ.value*np.cos(self.SkyImage.declination) for sinha,cosha in self.sin_cos_ha]) * self.Observatory.BX.unit / self.obs_lam).decompose() *1./u.radian

		#self.U = (-np.tril(self.U) + np.triu(self.U))*1./u.radian
		#self.V = (-np.tril(self.V) + np.triu(self.V))*1./u.radian
		#print(self.V[0],self.SkyImage.declination)
	def grid_uv_coverage(self):
		"""Knowing the pixelsize in radians and the image size in pixels, grid the UV coverage, returning integer values which correspond to indices (positions) in the
		fourier transformed image of the sky.
		"""
		self.u = np.int16(((self.U + 1./self.SkyImage.pixel_size.to(u.radian))/2 * self.SkyImage.pixel_size.to(u.radian) * self.SkyImage.imsize_X ).decompose().value)
		self.v = np.int16(((self.V + 1./self.SkyImage.pixel_size.to(u.radian))/2 * self.SkyImage.pixel_size.to(u.radian) * self.SkyImage.imsize_Y ).decompose().value)
		#print(self.v[0],self.SkyImage.declination)
	def make_masked_arr(self,weights = "natural"):
		"""Mask the Fourier transform of the sky. Only the values at indices corresponding to positions in the UV plane, given by u, v, are kept, the 
		rest are set to zero. Since the sky is real valued, its fourier transform is a hermitian-symmetric array, so we use the real fourier transform function, with
		the negative index values in u and v discarded, since they correspond to hermitian-symmetric values. Index values outside the Fourier transform array are also discarded. 
		"""

		self.UVC = np.zeros(self.SkyImage.fft_data.shape)
		#get the pairs of u,v samples and count how many times they appear in a cell (weights)
		uv_unique = np.unique((self.u[(self.u >0) & (self.v > 0) & (self.u < self.UVC.shape[0]) & (self.v < self.UVC.shape[1])]+self.v[(self.u >0) & (self.v > 0) & (self.u < self.UVC.shape[0]) & (self.v < self.UVC.shape[1])]*1.j),return_counts= True)
		self.uv_u = uv_unique
		if weights == "uniform": #for uniform weights they all have the same value
			self.UVC[np.real(uv_unique[0]).astype(int),np.imag(uv_unique[0]).astype(int)] = 1
		if weights == "natural": #for natural weights they are proportional to the sampling density
			self.UVC[np.real(uv_unique[0]).astype(int),np.imag(uv_unique[0]).astype(int)] = uv_unique[1] 

		#self.UVC[self.u[(self.u >0) & (self.v > 0) & (self.u < self.UVC.shape[0]) & (self.v < self.UVC.shape[1])],self.v[(self.u >0) & (self.v > 0) & (self.u < self.UVC.shape[0]) & (self.v < self.UVC.shape[1])]]=1
		
			



		#mask zero freq value 
		self.UVC[int(self.UVC.shape[0]/2),int(self.UVC.shape[1]/2)]=0
		#normalize weights
		self.UVC = (self.UVC - np.min(self.UVC))/(np.max(self.UVC) - np.min(self.UVC))

		self.UVC = self.UVC.T
	def make_dirty_arr(self):
		"""Make the dirty image based on the uv coverage UVC
		"""
		#print(self.u,self.v)
		uvc = self.UVC[:,:self.SkyImage.fft_data_r.shape[1]]
		self.uv_fft = uvc * self.SkyImage.fft_data_r
		self.uv_fft_sampled = self.UVC * self.SkyImage.fft_data
		#self.uv_fft_i = self.UVC * self.SkyImage.fft_data_i
		#self.dirty_beam = np.fft.ifftshift(np.fft.irfft2(uvc))
		self.dirty_beam = np.fft.ifftshift(np.fft.ifft2(self.UVC))
		self.dirty_image = np.fft.ifft2(self.uv_fft)
		#self.dirty_i = np.fft.ifft2(self.uv_fft_i)

class Reader:
	def __init__(self):
		self.ser_conn_dict = dict()
		self.ble_conn_dict = dict()


	def create_ble_conn(self,net_id=0x66ce):
		#devices = decawave_ble.scan_for_decawave_devices()
		#devices_tag = dict()
		#peripherals_tag = dict()
		#for key,value in devices.items():
		#	decawave_peripheral = decawave_ble.get_decawave_peripheral(value)
		#	operation_mode_data = decawave_ble.get_operation_mode_data_from_peripheral(decawave_peripheral)
		#	if operation_mode_data['device_type_name'] == 'Tag':
		#		devices_tag[key] = value
		#		peripherals_tag[key] = decawave_peripheral

		print("connecting to ble")
		devices = decawave_ble.scan_for_decawave_devices()
		devices_anchor, devices_tag, peripherals_anchor, peripherals_tag = dict(), dict(), dict(), dict()
		decawave_ble.retry_initial_wait = 0.1  # seconds
		decawave_ble.retry_num_attempts = 6

		for key, value in devices.items():
			decawave_peripheral = decawave_ble.get_decawave_peripheral(value)
			operation_mode_data = decawave_ble.get_operation_mode_data_from_peripheral(decawave_peripheral)
			network_id = decawave_ble.get_network_id_from_peripheral(decawave_peripheral)
			# ToDo: a hard-coded network ID is not a good idea! We need a tiny application that can re-assign
			#  devices with any deviceID to a specific network ID to retrieve one that went astray or was hijacked.
			if network_id == net_id:
				if operation_mode_data['device_type_name'] == 'Tag':
					devices_tag[key] = value
					peripherals_tag[key] = decawave_peripheral
			#	elif operation_mode_data['device_type_name'] == 'Anchor':
			#		devices_anchor[key] = value
			#		peripherals_anchor[key] = decawave_peripheral
			#else:
			#	logging.warning("Decawave devices found from network ID: {}, being disregarded".format(network_id))
		print(devices,devices_tag,peripherals_tag)
			

		self.ble_conn_dict={'networdk_id':net_id,'device_list':devices_tag,'peripherals_tag':peripherals_tag}
	def scan_ble_conn(self):
		
		result = {}
		network_id,devices_tag,peripherals_tag = self.ble_conn_dict.values()
		#print(devices_tag)

		for key,decawave_peripheral in peripherals_tag.items():
			try:
				location_data = decawave_ble.get_location_data_from_peripheral(decawave_peripheral)
				x = location_data["position_data"]['x_position']
				y = location_data["position_data"]['y_position']
				z = location_data["position_data"]['z_position']
				result[key] = np.array([x,y,z])
			except tenacity.RetryError:
				decawave_peripheral = decawave_ble.get_decawave_peripheral(devices_tag[key])
				peripherals_tag[key] = decawave_peripheral
		#print(result)
		return result
	def disconnect_ble(self):
		peripherals_tag = self.ble_conn_dict['peripherals_tag']
		for periheral in peripherals_tag.items():
			peripheral.disconnect()
		self.ble_conn_dict = {}



	def create_ser_conn(self,path='/dev/ttyUSB0',baud=19200,timeout=1,parity=serial.PARITY_NONE,rtscts=False):
		print("Creating connection to " + path)
		print("Existing ports:")
		for port in comports():
			print(port)
		print("Looking in /dev/:")
		print(glob.glob("/dev/*"))
		self.ser_conn_dict = {'info':[path,baud,timeout,parity,rtscts],'connection':serial.Serial(path,baud,timeout = timeout, parity = parity, rtscts = rtscts)}
		
		#print(self.conn_dict[name])
	#def open_ser_conn(self):
	#	path,baud,timeout,parity,rtscts = self.ser_conn_dict.values()
	#	self.ser_conn_ob_dict = serial.Serial(path,baud,timeout = timeout, parity = parity, rtscts = rtscts)
	def scan_ser_conn(self,return_raw_string = False,print_raw_string = False):
		self.ser_conn_dict['connection'].write('b\r'.encode())
		linein=self.ser_conn_dict['connection'].readline()
		#print(self.ser_conn_dict)
		if print_raw_string:
			print(linein.decode('utf8'))
		if return_raw_string == False:
			bit_pos = np.where(np.array([bit=='1' for bit in linein.decode('utf8')]))[0]
			#print(bit_pos)
			return bit_pos
		elif return_raw_string == True:
			return linein.decode('utf8')
	def disconnect_ser(self):
		self.ser_conn_dict['connection'].close()
		self.ser_conn_dict = {}


class read_controller:
	def __init__(self,sleep_time = 0.05):
		self.sleep_time = sleep_time
		pass
	def _loop_read(self,shared_var,conn_details,connection_type,sleep_time = None,verbose = False):
		if sleep_time is None:
			sleep_time = self.sleep_time
		if connection_type == 'ser':

			reader = Reader()
			reader.create_ser_conn(**conn_details)
			while True:
				sleep(sleep_time)
				try:
					val = reader.scan_ser_conn()
					shared_var[reader.ser_conn_dict['info'][0]] = val #reader.scan_ser_conn(print_raw_string = True)
					if verbose:
						print(val,shared_var,'#')
				except Exception as e:
					print("ser ctrl error!",conn_details,e)
					pass
			reader.disconnect_ser()
		if connection_type == 'ble':
			reader = Reader()
			reader.create_ble_conn(conn_details)
			while True:
				sleep(sleep_time)
				try:
					for key,value in reader.scan_ble_conn().items():
						shared_var[key]=value 
					if verbose:
						print(shared_var)
				except Exception as e:
					print("ble ctrl error!",e)
					pass


class Control:
	def __init__(self,ctrl_id = 'USB1',but_dict_file = None,separating_bit = 35,swi_bit_list = [],button_ids = ["R0","R1","R2","S0","S1","S2"]):
		self.ctrl_id = ctrl_id
		self.separating_bit = separating_bit
		self.but_dict_file = but_dict_file
		self.button_dict = dict()
		self.swi_bit_list = swi_bit_list
		self.button_ids = button_ids
	
	def add_button(self, button_id):
		self.button_dict[button_id] = Button(identifier = button_id)
	def add_buttons(self,button_ids = None):
		if button_ids is None:
			button_ids = self.button_ids
		for _bid in self.button_ids:
			self.add_button(_bid)
		return 1
	def set_button(self,button_id,state):
		if button_id in self.button_dict:
			self.button_dict[button_id].set_state(state)
			return 1
		else:
			print("error",button_id," not in button dict")
			return 0
	def set_buttons(self,button_ids = None,button_states=None):
		if button_ids is None:
			button_ids = self.button_ids
		if button_states is None:
			button_states = [0] * len(button_ids)
		if len(button_ids) != len(button_states):
			print("error, number button ids {} must equal number of button states {}".format(button_ids,button_states))
			return 0
		for _bid,_bst in zip(button_ids,button_states):
			self.set_button(_bid,_bst)	
		return 1
	def remove_button(self,button_id):
		try:
			del self.button_dict[button_id]
		except Exception as e:
			print("error",button_id," not in button dict")
			return 0

	def _read_button_bits(self,last_measurements):
		print(last_measurements[self.ctrl_id])
		rot_bits =tuple(last_measurements[self.ctrl_id][last_measurements[self.ctrl_id] < self.separating_bit])
		#print(rot_bits)
		swi_bits = tuple((last_measurements[self.ctrl_id][last_measurements[self.ctrl_id] > self.separating_bit])[:-1])
		return tuple(rot_bits),tuple(swi_bits)

	def get_all_pos(self,last_measurements):
		rot_bits, swi_bits = self._read_button_bits(last_measurements)
		rot_pos = self.but_bit_dict[rot_bits]
		swi_pos = list(np.array([_b in swi_bits for _b in self.swi_bit_list])*1)
		print(self.swi_bit_list,swi_bits,swi_pos)
		return tuple(rot_pos),tuple(swi_pos)
	
	def set_all_buttons(self,last_measurements):
		all_states = operator.add(*self.get_all_pos(last_measurements))
		print("setting all states",all_states)
		self.set_buttons(button_states = all_states)
		
	def get_state(self,last_measurements):
		self.set_all_buttons(last_measurements)
		return self.button_dict

		
		
	
	def _read_switch_part(self,last_measuerements):
		swi_part = tuple((last_measurements[self.ctrl_id][last_measurements[self.ctrl_id] > self.separating_bit])[:-1])
		return swi_part
	def set_swi_bit_list(self,swi_bit_list):
		self.swi_bit_list = swi_bit_list

	def set_but_bit_dict_file(self,fname):
		self.but_dict_file = fname
	
	def load_but_bit_dict(self,fname = None):
		if fname is None:
			fname = self.but_dict_file

		with open(fname,'rb') as fin:
			self.but_bit_dict = pickle.load(fin)


class Button:
	def __init__(self,identifier = "",init_state = 0):
		self.indentifier = identifier
		self.state = init_state
	def get_state(self):
		return self.state
	def set_state(self,state):
		self.state = state
	def get_id(self):
		return self.identifier
	def set_id(self,identifier):
		self.identifier = identifier


def get_ctrl_ant_usb(measurements_dict, use_last_bit = True, discriminant = 6):
	"""Description
	===============
	Function that selects which USB serial connection contains antenna position information
	and which one contains commands from the controls (buttons or magnets etc.). The last bit of
	the command (control) string should be set to 1. The USB connection that returns a string 
	where the last bit (bit nr. 63) is 1, becomes the control connection, while the other becomes the antenna
	position connection. If not set to use last bit, it asks the user to put at least N (default 6) antennas on the board
	and then uses the connection with at least N active bits as the antenna position connection.
	Arguments
	---------------
	use_last_bit : bool
		If true, connection where bit nr. 63 is 1 becomes control connection, the other one becomes antenna position connection
	discriminant: int
		If use_last_bit is false, ask the user to place at least this amount of antennas on the board, and select the connection which returns
		at least this many 1 bits as the antenna connection, the other becoming the control connection
 
	Returns
	---------------
	tuple: (str,str)
	(The name of the USB connection for antenna position, The name of the USB connection for control )
	"""
	if use_last_bit:
		print("Selecting controllers based on last bit value.")
		print([[key,result] for key,result in measurements_dict.items()])
		ctrl_usb_src = [key for key,result in measurements_dict.items() if 'USB' in key and 63 in result]
		ant_usb_src = [key for key,result in measurements_dict.items() if 'USB' in key and 63 not in result]
		if ant_usb_src == ctrl_usb_src:
			print("error, cannot differentiate between USB antenna and USB ctrl connection")
			return False
		else:
			print("Found antenna data connections:",ant_usb_src)
			print("Found telescope control connection:",ctrl_usb_src)
			return ant_usb_src[0],ctrl_usb_src[0]

	else:
		ans = input("Selecting USB connections for antennas and for the controls. \
				Place at least"+str(int(discriminant))+ "antennas on the table, then press Enter.")
		print(measurements_dict)
		ant_usb_src = [key for key,result in measurements_dict.items() if 'USB' in key and len(result) >= discriminant]
		ctrl_usb_src = [key for key,result in measurements_dict.items() if 'USB' in key and len(result) < discriminant]
		if ant_usb_src == ctrl_usb_src:
			print("error, cannot differentiate between USB antenna and USB ctrl connection")
			return False
		else:
			print("Found antenna data connections:",ant_usb_src)
			print("Found telescope control connection:",ctrl_usb_src)
			return ant_usb_src[0],ctrl_usb_src[0]



class BlitManager:
	def __init__(self, canvas, animated_artists=()):
		"""
		Parameters
		----------
		canvas : FigureCanvasAgg
			The canvas to work with, this only works for subclasses of the Agg
			canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
			`~FigureCanvasAgg.restore_region` methods.

		animated_artists : Iterable[Artist]
			List of the artists to manage
		"""
		self.canvas = canvas
		self._bg = None
		self._artists = []
		for a in animated_artists:
			self.add_artist(a)
		# grab the background on every draw
		self.cid = canvas.mpl_connect("draw_event", self.on_draw)

	def on_draw(self, event):
		"""Callback to register with 'draw_event'."""
		cv = self.canvas
		if event is not None:
			if event.canvas != cv:
				raise RuntimeError
		self._bg = cv.copy_from_bbox(cv.figure.bbox)
		self._draw_animated()

	def add_artist(self, art):
		"""
		Add an artist to be managed.

		Parameters
		----------
		art : Artist

			The artist to be added.  Will be set to 'animated' (just
			to be safe).  *art* must be in the figure associated with
			the canvas this class is managing.

		"""
		if art.figure != self.canvas.figure:
			raise RuntimeError
		art.set_animated(True)
		self._artists.append(art)

	def _draw_animated(self):
		"""Draw all of the animated artists."""
		fig = self.canvas.figure
		for a in self._artists:
			fig.draw_artist(a)

	def update(self):
		"""Update the screen with animated artists."""
		cv = self.canvas
		fig = cv.figure
				
		# paranoia in case we missed the draw event,
		if self._bg is None:
			self.on_draw(None)
		else:
			# restore the background
			cv.restore_region(self._bg)
			# draw all of the animated artists
			self._draw_animated()
			# update the GUI state
			cv.blit(fig.bbox)
		# let the GUI event loop process anything it has to do
		cv.flush_events()

class DisplayManager:
	def __init__(self):
		self._blit_manager_list = []
		self._axes = []
		self._ln_arr = []
		self.maxoffset = 250
		self._ind_text = []
		self._txt = ""
		self._ind_status_text = ""
	def setup_main_figure(self,nrows = 2,ncols = 4,show_button_indicator = True):
		self._fig = plt.figure(figsize=(12, 9)) 
		self._nrows = nrows
		self._ncols = ncols
		#if show_button_indicator == True:
		#	self._nrows += 1 
		self._axes_shape = (nrows,ncols)
		#self._axes = plt.subplots(nrows,ncols)
		plot_counter = 0
		for _row in range(self._nrows):
			#if show_button_indicator == True and plot_counter >= self._nrows * self._ncols:
			#	break
			for _col in range(self._ncols):
				self._axes.append(
						plt.subplot2grid(
							(self._nrows,self._ncols),
							(_row,_col),
							colspan = 1,
							rowspan = 1,
							fig = self._fig))

			#	plot_counter +=1
		if show_button_indicator == True:
			#self._axes.append(
			#		plt.subplot2grid(
			#			(self._nrows,self._ncols),
			#			(self._nrows-1,0),
			#			colspan = self._ncols,
			#			rowspan = 1,
			#			fig = self._fig))
			self._axes.append(self._fig.add_axes((0,0.475,1,0.1)))
					#self._axes[-1].set_box_aspect(1./8)
		plt.subplots_adjust(left = 0,right=1,bottom=0,top=1,wspace=0, hspace=0.1)	
		self._ln_arr = [None] * nrows * ncols
		

	def init_ant_plot(self, maxoffset = 250, row = 0, col = 0):
		self.maxoffset = maxoffset
		ii = np.ravel_multi_index((row,col),(self._nrows,self._ncols))
		(ln,) = self._axes[ii].plot([-10,10],[-10,10],color='white',marker = "o",linestyle='None',animated = True)
		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._axes[ii].set_xlim(-maxoffset*1.1,maxoffset*1.1)
		self._axes[ii].set_ylim(-maxoffset*1.1,maxoffset*1.1)
		self._axes[ii].set_aspect('equal')
		#print(maxoffset)
		self._ln_arr[ii] = ln

		self._txt = self._axes[ii].annotate( "0",(0, 1),xycoords="axes fraction",xytext=(10, -10),textcoords="offset points",ha="left",va="top",animated=True)
	
	def init_ant_proj_plot(self, maxoffset = 250, row = 1, col = 0):
		self.maxoffset = maxoffset
		ii = np.ravel_multi_index((row,col),(self._nrows,self._ncols))
		(ln,) = self._axes[ii].plot([-10,10],[-10,10],color='white',marker = "o",linestyle='None',animated = True)
		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._axes[ii].set_xlim(-maxoffset*1.1,maxoffset*1.1)
		self._axes[ii].set_ylim(-maxoffset*1.1,maxoffset*1.1)
		self._axes[ii].set_aspect('equal')
		#print(maxoffset)	
		self._ln_arr[ii] = ln
	

	
	def init_img_plot(self,size=480, row = 0, col = 1):
		dummy_img = np.zeros((size,size))
		ii = np.ravel_multi_index((row,col),(self._nrows,self._ncols))
		im = self._axes[ii].imshow(dummy_img,vmin = 0,vmax = 255,cmap="hot")

		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._ln_arr[ii] = im
		#print(ii,self._ln_arr[ii],"ini img")

	def init_fft_plot(self,size=480, row = 1, col = 1):
		dummy_img = np.zeros((size+1,size+1))
		ii = np.ravel_multi_index((row,col),(self._nrows,self._ncols))
		im = self._axes[ii].imshow(dummy_img,vmin = 0,vmax = 1.,cmap="nipy_spectral")

		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._ln_arr[ii] = im

	def init_uvc_plot(self,size=480, row = 1, col = 2):
		dummy_img = np.zeros((size+1,size+1))
		ii = np.ravel_multi_index((row,col),(self._nrows,self._ncols))
		im = self._axes[ii].imshow(dummy_img,vmin = 0,vmax = 1.,cmap="gray")

		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._ln_arr[ii] = im

	def init_dbe_plot(self,size=480,row = 0, col = 2):
		dummy_img = np.zeros((size+1,size))
		ii = np.ravel_multi_index((row,col),(self._nrows,self._ncols))
		#im = self._axes[ii].imshow(dummy_img,vmin = 0.,vmax = 1.,cmap="rainbow")
		im = self._axes[ii].imshow(dummy_img,vmin = 0.,vmax = 1.,cmap="nipy_spectral")
		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._ln_arr[ii] = im

	def init_dim_plot(self,size=480,row= 0,col = 3):
		dummy_img = np.zeros((size+1,size))
		ii = np.ravel_multi_index((row,col),(self._nrows,self._ncols))
		#im = self._axes[ii].imshow(dummy_img,vmin = 0.,vmax = 1.,cmap="rainbow")
		im = self._axes[ii].imshow(dummy_img,vmin = 0.,vmax = 1.,cmap="hot")
		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._ln_arr[ii] = im
	def init_mft_plot(self,size=480,row= 1,col = 3):
		dummy_img = np.zeros((size+1,size))
		ii = np.ravel_multi_index((row,col),(self._nrows,self._ncols))
		#im = self._axes[ii].imshow(dummy_img,vmin = 0.,vmax = 1.,cmap="rainbow")
		im = self._axes[ii].imshow(dummy_img,vmin = 0.,vmax = 1.,cmap="nipy_spectral")#"hot")
		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._ln_arr[ii] = im
	def init_ind_plot(self,var_dic,Observatory):
		r0,r1,r2,s1,s2,s3 = tuple([Observatory.Control.button_dict[_bid].get_state() for _bid in Observatory.Control.button_ids])
		upper_text = ['0','1','2','3']
		op_to_text = {operator.add : '+', operator.mul : 'x'}
		mid_text = [var_dic[r0][0][_ir1]+" "+ var_dic[r0][3][_ir1]+" "+ str(var_dic[r0][1][_ir1][1]) for _ir1 in range(len(var_dic[r0][0]))]
		low_text = [op_to_text[var_dic[r0][1][r1][0]] + " " + str(var_dic[r0][2][r1][_ir2]) for _ir2 in  range(len(var_dic[r0][0]))]
		self.rot_text_menu = [upper_text,mid_text,low_text]

		
		ii = -1
		(ln,) = self._axes[ii].plot([],[],color='black',linestyle='None',animated = True)
		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._axes[ii].axes.set_axis_off()
		for _yy in range(len(self.rot_text_menu)):
			for _xx in range(len(self.rot_text_menu[0])):
				self._ind_text.append(self._axes[ii].text(0.3 + 0.65/len(self.rot_text_menu[0])*_xx,0.05 + 0.75/len(self.rot_text_menu) * _yy, self.rot_text_menu[_yy][_xx],transform=self._axes[ii].transAxes,color="white",ha="left", va="center",bbox=dict(boxstyle="square,pad=0.3",fc="black", ec="steelblue", lw=2)))
		status_text = "HA Start: {} \nHA End: {} \nDec: {} \nInt time: {} \nFrequency: {} \nImage file: {}".format(Observatory.HA_START,
				Observatory.HA_END,
				Observatory.SkyImage.declination,
				abs(Observatory.HA_END-Observatory.HA_START),
				Observatory.obs_frequency,
				Observatory.SkyImage.filename) 		

		self._ind_status_text = self._axes[ii].text(0.0,0.3,status_text,transform = self._axes[ii].transAxes,color="white",ha="left",va="center",bbox=dict(boxstyle="square,pad=0.1",fc="black", ec="steelblue", lw=2))
			
	#	var_dic = \
	#		{0:[['HA']*4,list(zip([operator.add]*4,d00)),[d01]*4],
	#		 1:[['Dec']*4,list(zip([operator.add]*4,d10)),[d11]*4],
	#		 2:[['Int']*2+ ['obs_frequency']*2,list(zip([operator.mul]*2+[operator.add]*2,d20)),[d210]*2+[d211]*2],
	#		 3:[['Fname']*4,list(zip([operator.add]*4,d30)),[d31]*4]}
		


	def setup_blit_manager(self):
		artist_list = [_ for _ in self._ln_arr if _ is not None] + [self._txt] + self._ind_text + [self._ind_status_text]
		self._blit_manager = BlitManager(self._fig.canvas,artist_list) 
		print(artist_list)
		plt.show(block = False)
		plt.pause(0.2)
	def update_blit_manager(self):
		self._blit_manager.update()
	def update_ant_plot(self,data,row = 0, col = 0):
		maxoffset = self.maxoffset
		ln_ind = np.ravel_multi_index((row,col),self._axes_shape)
		data = np.array(data)
		#print((-maxoffset < data[0])&(data[0] < maxoffset),(-maxoffset < data[1])&(data[1] < maxoffset),data,maxoffset,"arrayloc")
		if ((-maxoffset < data[0])&(data[0] < maxoffset)).all() and ((-maxoffset < data[1])&(data[1] < maxoffset)).all():
			scale_plot_factor = 1
			xx = data[0]
			yy = data[1]
			#print("normal",xx,yy)
		else:
			xx = data[0] - np.mean(data[0])
			yy = data[1] - np.mean(data[1])

			#print("scaled",xx,yy)
			scale_plot_factor = 2*maxoffset / np.max((np.abs(np.max(data[0]) - np.min(data[0])),np.abs(np.max(data[1]) - np.min(data[1]))))
		self._ln_arr[ln_ind].set_xdata(xx*scale_plot_factor)
		self._ln_arr[ln_ind].set_ydata(yy*scale_plot_factor)
		self._txt.set_text(str(data)+" "+str(scale_plot_factor))  
		#self._blit_manager_list[0].update()
	def update_ant_proj_plot(self,data,row = 1, col = 0,observatory_latitude = -23.03 * u.deg, hrangle = 0 *u.deg,dec = -40 * u.deg):
		maxoffset = self.maxoffset
		ln_ind = np.ravel_multi_index((row,col),self._axes_shape)
		#print(data[0])
		if ((-maxoffset < data[0])&(data[0] < maxoffset)).all() and ((-maxoffset < data[1])&(data[1] < maxoffset)).all():
			scale_plot_factor = 1
			xx = data[0]
			yy = data[1]
			#print("normal",xx,yy)
		else:
			xx = data[0] #- np.mean(data[0])
			yy = data[1] #- np.mean(data[1])

			#print("scaled",xx,yy)
			scale_plot_factor = maxoffset / np.max(np.abs(data))#(np.abs(np.max(data[0]) - np.min(data[0])),np.abs(np.max(data[1]) - np.min(data[1]))))
		xx_earth_cen = - yy*np.sin(observatory_latitude).value
		yy_earth_cen = xx
		zz_earth_cen = yy * np.cos(observatory_latitude).value
		xx_antpos_proj = - xx_earth_cen * np.sin(hrangle).value \
				 + yy_earth_cen * np.cos(hrangle).value
		yy_antpos_proj = - xx_earth_cen * np.sin(dec).value*np.cos(hrangle).value  \
				 + yy_earth_cen * np.sin(dec).value*np.sin(hrangle).value \
				 + zz_earth_cen * np.cos(dec).value


		#print(scale_plot_factor)	
		if hrangle > 0:
			self._ln_arr[ln_ind].set_xdata(yy_antpos_proj * scale_plot_factor)
			self._ln_arr[ln_ind].set_ydata(xx_antpos_proj * scale_plot_factor)
		if hrangle < 0:
			self._ln_arr[ln_ind].set_xdata( - yy_antpos_proj * scale_plot_factor)
			self._ln_arr[ln_ind].set_ydata( - xx_antpos_proj * scale_plot_factor)
		if hrangle == 0:
			self._ln_arr[ln_ind].set_xdata(xx_antpos_proj * scale_plot_factor)
			self._ln_arr[ln_ind].set_ydata(yy_antpos_proj * scale_plot_factor)

		#self._blit_manager_list[ln_ind].update()

	def update_img_plot(self,data,row = 0, col = 1):
		ln_ind = np.ravel_multi_index((row,col),self._axes_shape)
		self._ln_arr[ln_ind].set_array(data)
		#self._blit_manager_list[ln_ind].update()
		#print(ln_ind,self._blit_manager_list[ln_ind],"upd img")
	
	def update_fft_plot(self,data,row = 1, col = 1):
		ln_ind = np.ravel_multi_index((row,col),self._axes_shape)
		plt_data = np.log10(np.abs(np.fft.fftshift(data)))
		plt_data = (plt_data - np.min(plt_data))/(np.max(plt_data) - np.min(plt_data))

		self._ln_arr[ln_ind].set_array(plt_data)

		#print(np.max(np.abs(data)/np.max(np.abs(data))),data.shape)
		#self._blit_manager_list[ln_ind].update()
	
	def update_uvc_plot(self,data,row = 1, col = 2):
		ln_ind = np.ravel_multi_index((row,col),self._axes_shape)
		#print(np.max(np.abs(data)))
		self._ln_arr[ln_ind].set_array(np.abs(data))
		#self._blit_manager_list[ln_ind].update()

	def update_dbe_plot(self,data,row = 0, col = 2):
		ln_ind = np.ravel_multi_index((row,col),self._axes_shape)
		plt_data =np.abs(data)#np.fft.fftshift(data)
		plt_data = (plt_data - np.min(plt_data)) / (np.max(plt_data) - np.min(plt_data)) #* 2 - 1
		#plt_data = plt_data - np.mean(plt_data)
		self._ln_arr[ln_ind].set_array(plt_data)
		#self._blit_manager_list[ln_ind].update()
	def update_dim_plot(self,data,row = 0, col = 3):
		ln_ind = np.ravel_multi_index((row,col),self._axes_shape)
		plt_data = np.abs(data)
		plt_data = (plt_data - np.min(plt_data)) / (np.max(plt_data) - np.min(plt_data)) #* 2 - 1
		#plt_data = plt_data - np.mean(plt_data)
		self._ln_arr[ln_ind].set_array(plt_data)
		#self._blit_manager_list[ln_ind].update()
	def update_mft_plot(self,data,row = 1, col = 3):
		ln_ind = np.ravel_multi_index((row,col),self._axes_shape)
		plt_data = np.abs(data)
		nz = plt_data.nonzero()
		plt_data[nz] = (plt_data[nz] - np.min(plt_data[nz])) / (np.max(plt_data[nz]) - np.min(plt_data[nz])) #* 2 - 1
		time_list = []
		#plt_data = plt_data - np.mean(plt_data)
		self._ln_arr[ln_ind].set_array(plt_data)
		#self._blit_manager_list[ln_ind].update()
	def update_ind_plot(self,var_dic,Observatory):
		r0,r1,r2,s1,s2,s3 = tuple([Observatory.Control.button_dict[_bid].get_state() for _bid in Observatory.Control.button_ids])
		upper_text = ['0','1','2','3']
		op_to_text = {operator.add : '+', operator.mul : 'x'}
		mid_text = [var_dic[r0][0][_ir1]+" "+ var_dic[r0][3][_ir1]+" "+ str(var_dic[r0][1][_ir1][1]) for _ir1 in range(len(var_dic[r0][0]))]
		low_text = [op_to_text[var_dic[r0][1][r1][0]] + " " + str(var_dic[r0][2][r1][_ir2]) for _ir2 in  range(len(var_dic[r0][0]))]
		self.rot_text_menu = [upper_text,mid_text,low_text]
		flat_text_menu = list(itertools.chain.from_iterable(self.rot_text_menu))
		_it = 0
		for _yy in range(len(self.rot_text_menu)):
			for _xx in range(len(self.rot_text_menu[0])):
				self._ind_text[_it].set_text(flat_text_menu[_it])
				if (_yy == 0 and _xx == r0) or (_yy == 1 and _xx == r1) or (_yy == 2 and _xx == r2):
					self._ind_text[_it].set_bbox(dict(facecolor="red"))
				else:
					self._ind_text[_it].set_bbox(dict(facecolor="black"))
				_it += 1
		status_text = "HA Start: {} \nHA End: {} \nDec: {} \nInt Time: {} \nFrequency: {} \nImage file: {}".format(Observatory.HA_START,
				Observatory.HA_END,
				Observatory.SkyImage.declination,
				abs(Observatory.HA_END-Observatory.HA_START),
				Observatory.obs_frequency,
				Observatory.SkyImage.filename) 		
		if s1 == 1:
			self._ind_status_text.set_text(status_text)
			self._ind_status_text.set_bbox(dict(facecolor="red"))
		else:
			self._ind_status_text.set_bbox(dict(facecolor="black"))


if __name__ == "__main__":
	test_conns = True
	with Manager() as manager:
		ser_conn_0 = {'path':'/dev/ttyUSB2','baud':19200,'timeout':1,'parity':serial.PARITY_NONE,'rtscts':False}
		ser_conn_1 = {'path':'/dev/ttyUSB3','baud':19200,'timeout':1,'parity':serial.PARITY_NONE,'rtscts':False}
		ble_conn = 0x66ce
#		fig,ax = plt.subplots()
#		(ln,) = ax.plot([-100,100],[-100,100],color='blue',marker = "o",linestyle='None',animated = True)
#		current_ranges = np.array([ax.get_xlim(),ax.get_ylim()])
#		print(current_ranges)
#		bm = BlitManager(fig.canvas, [ln,])
		d00 = [-6,-3,0,+3]
		d01 = [0,1,2,3]
		d10 = [-90,-50,-10,30]
		d11 = [0,10,20,30]
		d20 = [0.2,1,100,300]
		d210 = [1,2,3,12]
		d211 = [0,50,100,150]
		d30 = ['m','m1','m2','m3']
		d31 = ['1.jpg','2.jpg','3.jpg','4.jpg']
		var_dic = \
			{0:[['HA']*4,list(zip([operator.add]*4,d00)),[d01]*4,["hr"]*4],
			 1:[['Dec']*4,list(zip([operator.add]*4,d10)),[d11]*4,["deg"]*4],
			 2:[['Int']*2+ ['obs_frequency']*2,list(zip([operator.mul]*2+[operator.add]*2,d20)),[d210]*2+[d211]*2,["hr"]*2+["GHz"]*2],
			 3:[['Fname']*4,list(zip([operator.add]*4,d30)),[d31]*4,[" "]*4]}
		
		swi_bits = [52,44,36]


		plt.style.use('dark_background')
		sleep_time = 0.0
		ser0_controller = read_controller(sleep_time = sleep_time)
		ser1_controller = read_controller(sleep_time = sleep_time)
		#ble_controller = read_controller()

		last_measurements = manager.dict()
		#ser0_controller._loop_read(last_measurements,ser_conn_0,'ser')
		p0 = Process(target = ser0_controller._loop_read,args = (last_measurements,ser_conn_0,'ser'))
		p1 = Process(target = ser1_controller._loop_read,args = (last_measurements,ser_conn_1,'ser'))
		#p2 = Process(target = ble_controller._loop_read,args=(last_measurements,ble_conn,'ble'))
		p0.start()
		p1.start()
		time.sleep(3)
		#p2.start()
		#print("Loading BLE, please wait...")
		alma = Observatory(ant_pos_bit_file='./ant_pos.2.txt')
		alma.load_ant_pos_bit_file()
		print(last_measurements)
		ant_usb_id,ctrl_usb_id = get_ctrl_ant_usb(last_measurements)
		src_list = [ant_usb_id]#,'DW4F0B','DWD900']#,'DW4F0B','DW5293','DW912D','DWD900']
		ctrl_list = [ctrl_usb_id]
		alma.set_read_source_ids(src_list)
		#alma.transform_query(last_measurements,['DW5293','DW4F0B','DW912D'])
		#alma.transform_query(last_measurements,'DW4F0B')
		#if alma.vlbi_mode == False:
			#	p2.terminate()
			
		alma.set_ant_pos(last_measurements)
		alma.make_baselines()
		imgobj = SkyImage(path = "./models/",filename = "galaxy_lobes.png")
		imgobj.load_image()
		imgobj.make_invert()
		cont = Control(ctrl_id = ctrl_usb_id)
		cont.add_buttons()
		cont.set_but_bit_dict_file("but_dict.pickle")
		cont.load_but_bit_dict()
		cont.set_swi_bit_list(swi_bits)
		cont.get_state(last_measurements)	
		obs = Observation(alma,imgobj,cont,obs_frequency = 200*u.GHz,var_dic = var_dic)

		#obs.set_read_source_ids(ctrl_usb_id)
		
		obs.calc_el_curve()
		obs.make_uv_coverage()
		obs.grid_uv_coverage()
		obs.make_masked_arr()
		obs.make_dirty_arr()
		#print("A")	
		dm = DisplayManager()
		dm.setup_main_figure()
		dm.init_ant_plot(maxoffset = 120)
		dm.init_ant_proj_plot(maxoffset = 120)
		dm.init_img_plot()
		dm.init_fft_plot()
		dm.init_uvc_plot()
		dm.init_dbe_plot()
		dm.init_dim_plot()
		dm.init_mft_plot()
		dm.init_ind_plot(var_dic,obs)
		dm.setup_blit_manager()
		#dm.update_ant_plot((alma.ant_pos_EW,alma.ant_pos_NS))
		#dm.update_ant_proj_plot((alma.ant_pos_EW,alma.ant_pos_NS))
		#dm.update_img_plot(imgobj.data)
		#frequency_list = [1,10,100,200,400,500,1000]
		#frequency_list = [_*u.GHz for _ in frequency_list]
		#freq_cycle = itertools.cycle(frequency_list)
		dec_list = itertools.cycle([_*u.deg for _ in[-50,-30,20]])
		sleep(0.5)

		time_list = []
		print("A")	
		while True:
			try:
				#plt.pause(0.1)
				if len(time_list) > 0:
					print((np.asarray(time_list) - np.roll(np.asarray(time_list),1))[1:])
				
				time_list = []
				time_list.append(time.time())

				#print(last_measurements)
				#print(alma.ant_pos_EW,alma.ant_pos_NS)
				#print(imgobj.declination)
				obs.Observatory.set_ant_pos(last_measurements)
				obs.Observatory.make_baselines()
				#time.sleep(0.5)	
				#print("B")	
				obs.update_obs_from_control(last_measurements)
				time_list.append(time.time())
				#obs.update_obs()#measurements_dict = last_measurements)#,obs_frequency = next(freq_cycle))
				obs.calc_el_curve()
				time_list.append(time.time())
				obs.make_uv_coverage()
				time_list.append(time.time())
				obs.grid_uv_coverage()
				time_list.append(time.time())
				obs.make_masked_arr(weights="uniform")
				time_list.append(time.time())
				obs.make_dirty_arr()
				time_list.append(time.time())
				#print(np.max(obs.dirty_beam),np.min(obs.dirty_beam),np.mean(obs.dirty_beam),np.percentile(obs.dirty_beam,[5,15,50,65,95]))
				dm.update_ant_plot((obs.Observatory.ant_pos_EW,obs.Observatory.ant_pos_NS))
				dm.update_ant_proj_plot((obs.Observatory.ant_pos_EW,obs.Observatory.ant_pos_NS))

				dm.update_img_plot(obs.SkyImage.data)
				dm.update_fft_plot(obs.SkyImage.fft_data)

				dm.update_uvc_plot(obs.UVC)
				dm.update_dbe_plot(obs.dirty_beam)

				dm.update_dim_plot(obs.dirty_image)
				dm.update_mft_plot(obs.uv_fft_sampled)
				dm.update_ind_plot(var_dic,obs)
				dm.update_blit_manager()
				time_list.append(time.time())
				#plt.scatter(alma.ant_pos_EW,alma.ant_pos_NS,c='blue')
				#obs = Observation(almaobj,imgobj)
				#obs.calc_el_curve()
				#plt.show()
	
			except KeyboardInterrupt:
				p0.terminate()
				p1.terminate()
				#p2.terminate()
				plt.close()
				sys.exit()
				try:
					sys.exit(130)
				except SystemExit:
					os._exit(130)
			except Exception as e:
				print("error! wtf?",e)
				p0.terminate()
				p1.terminate()
				#p2.terminate()
				plt.close()
				sys.exit()
				try:
					sys.exit(130)
				except SystemExit:
					os._exit(130)
		p0.terminate()
		p1.terminate()










#Button 3 rotates, Button 2 at position 1:
#
#{'/dev/ttyUSB1': '0000011100000000000000000000000000000000000000000000000000000001\n'}  position 1
#{'/dev/ttyUSB1': '0000011000000011000000000000000000000000000000000000000000000001\n'}  position 2
#Correct should be:
#{'/dev/ttyUSB1': '0000011000000001000000000000000000000000000000000000000000000001\n'} position 2
#{'/dev/ttyUSB1': '0000011000000000000000010000000000000000000000000000000000000001\n'} position 3
#{'/dev/ttyUSB1': '0000011000000000000000000000000100000000000000000000000000000001\n'} position 4
#
#Button 2 rotates, Button 3 at position 1:
#
#{'/dev/ttyUSB1': '0000011100000000000000000000000000000000000000000000000000000001\n'} position 1
#{'/dev/ttyUSB1': '0000010100000011000000000000000000000000000000000000000000000001\n'} position 2
#Correct should be:
#{'/dev/ttyUSB1': '0000010100000010000000000000000000000000000000000000000000000001\n'} position 2
#{'/dev/ttyUSB1': '0000010100000000000000100000000000000000000000000000000000000001\n'} position 3
#{'/dev/ttyUSB1': '0000010100000000000000000000001000000000000000000000000000000001\n'} position 4
#
#In bit numbers:
#
#{'/dev/ttyUSB1': array([ 5,  6,  7, 63])}
#{'/dev/ttyUSB1': array([ 5,  7, 14, 15, 63])} #Button 2 in position 2
#Correct should be:
#{'/dev/ttyUSB1': array([ 5,  7, 14, 63])} #Button 2 in position 2
#
#{'/dev/ttyUSB1': array([ 5,  6, 14, 15, 63])} #Button 3 in position 2
#Correct should be:
#{'/dev/ttyUSB1': array([ 5,  6, 15, 63])} #Button 3 in position 2



