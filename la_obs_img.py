import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
import astropy.constants as C
from PIL import Image
from astropy.io import ascii
import traceback
import logging
from scipy.ndimage import gaussian_filter
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
		
	def set_ant_pos(self,last_measurements):
		"""Description
		==============
		Sets the antenna positions using the hardware data. Updates
		the antenna positions in units of length along the East West (EW), North South (NS) axes,
		and in cartesian coordinates.
		Arguments
		---------------
		last_measurements : dict 
			Contains the measurements from the USB and Bluetooth channels (antenna positions from 
			pads on the table and antenna positions from bluetooth devices)
		"""
		pos = [[],[]]
		try:
			for src in self._read_source_list:
				if 'USB' in src:
					bit_array = last_measurements[src]
					#if only one antenna don't bother
					if len(bit_array)<1:
						bit_array = [0]
					#print(bit_array)
					#print(np.array([np.array(self._ant_bit_dict[bit]) for bit in bit_array]).T,"##################################")
					pos = np.hstack([pos,np.array([np.array(self._ant_bit_dict[bit]) for bit in bit_array]).T])
				if 'DW' in src:
					try:
						pos = np.hstack([pos,self.apply_transform(last_measurements[src][:-1])[np.newaxis].T])
					except Exception as e:
						print("set_ant_pos",e)
						pass
		except Exception as e:
			print("error",e,traceback.format_exc())
			pass
		self.ant_pos_EW,self.ant_pos_NS = pos
		self.ant_pos_X = -1*self.ant_pos_NS * np.sin(self.latitude)
		self.ant_pos_Y  = self.ant_pos_EW
		self.ant_pos_Z = self.ant_pos_NS * np.cos(self.latitude)

	def set_3pos_tag_single(self,last_measurements,tag_id):
		"""Description
		==============
		Measures the time averaged position of one BLE tag in three different places, used 
		later to match the BLE coordinate system to that of the table.
		Arguments
		--------------
		last_measurements : dict
			the measurements dict
		tag_id : str
			the ID of the tag whose position is to be measured
		Returns
		--------------
		List of shape length 2 : float
			a 2D vector repr the pposition of the tag
		"""
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
						data_in = last_measurements[tag_id][:-1]
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
	def set_3pos_tag_all(self,last_measurements,tag_ids):
		"""Description
		==============
		Measures the time averaged position of three BLE tags in three different places, used 
		later to match the BLE coordinate system to that of the table.
		Arguments
		--------------
		last_measurements : dict
			the measurements dict
		tag_ids : list 
			the list of tag IDs (strings) of the tags whose position is to be measured

		sets X_vecs parameter to List of shape (3,2) : float
			a list of 3 2D vectors repr. the positions of three antennas 

		
		Returns : 
		-------------
		X_vecs
			"""

		X_vecs = []
		print("Aligning tags with ALMA")
		input("Place tag "+str(tag_ids)+" at the red, orange, and green pads, respectively, and press Enter")
		temp_vec = []
		data_in = None
		with tqdm(total = 40,desc="Collecting position data", unit="tag positions") as pbar:
			while len(temp_vec) < 40:
				sleep(0.5)
				try:	
					data_in = np.array([last_measurements[tag_id][:-1] for tag_id in tag_ids])
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
		return X_vecs

	
	def set_transform(self,X=None):
		"""
		Description
		===============
		Calculate the matrix transform needed to convert the position
		Arguments
		---------------
		X : list
			a list of three 2D position vectors, on default use X_vecs attribute
		Returns
		---------------
		1 : int
			on success, sets the A and b parameters to a 2D matrix representing a linear
			map and b representing a bias, which together take the coordinates of the BLE
			tags into the loaded alma coordinates using an equation of the type Y = A X + B
		"""
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
			return 1
		except Exception as e:
			print("can't get matrix transform:",e)
			return 0
			
	def transform_query(self,last_measurements,tag_id):
		"""Description
		==============
		Function which interacts with the users and either starts the position calibration process
		to get the transform between the BLE coordinates and ALMA coordinates, or loads a saved transform from file.

		Arguments
		---------
		last_measurements : dict
			measurements dict
		tag_id : list
			list of tag ids
		Returns
		---------

		"""
		ans = input("Perform tag to ALMA position calibration?(y) load from disk? (l) don't use tags! (n) quit (q)")
		if ans == "y" and type(tag_id) == str:
			self.set_3pos_tag_single(last_measurements,tag_id)
			self.set_transform()
			with open("transM.pickle",'wb') as fout:
				pickle.dump([self.A,self.b],fout)
			self.vlbi_mode = True
		if ans == "y" and type(tag_id) == list:
			self.set_3pos_tag_all(last_measurements,tag_id)
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
		"""Description
		==============
		Performs the transformation from BLE tag coordinates into ALMA
		coordinates.
		Arguments
		---------
		x : Array (2) floats
			position of a tag
		A : Array (2,2) floats
			linear map
		b : Array (2,) floats
			bias vector
		Returns
		--------
		y : Array (2) floats
			the position of the BLE tag in ALMA coordinates vector
		
		"""
		if A == None:
			A = self.A
		if b == None:
			b = self.b
		y = A@x + b
		return y 


	def set_raw_ant_pos_from_ble(self,ble_data):
		#obsolete
		self.ant_pos_ble = np.array([pos for pos in ble_data.values()]).T
	#def merge_ant_pos(self):
	#	self.ant_pos = np.hstack([self.ant_pos_ser,self.ant_pos_ble])
	#	self.ant_pos_EW = self.ant_pos[0]
	#	self.ant_pos_NS = self.ant_pos[1]
		
	def make_baselines(self):
		"""Description
		==============
		Sets (Calculates) baselines based on the positions of the Cartesian coordinates of the antennas. The results are in meters.
		The baselines are stored in the BX, BY, BZ attributes.
		Arguments: None
		Returns: None
		"""
		nant = len(self.ant_pos_X)	
		res = np.zeros((3,nant,nant))
		#if only one antenna don't bother... 
		if nant > 1:
			for i in range(nant):
				for j in range(nant):
					res[0,i,j] = self.ant_pos_X[i] - self.ant_pos_X[j]
					res[1,i,j] = self.ant_pos_Y[i] - self.ant_pos_Y[j]
					res[2,i,j] = self.ant_pos_Z[i] - self.ant_pos_Z[j]
			self.BX,self.BY,self.BZ = res * u.m
		else:
			self.BX,self.BY,self.BZ = res * u.m

		
	def get_baselines(self):
		"""Description
		==============
		Gets the baseline values that have been calculated already
		Arguments: 
		----------
		None
		Returns:
		----------
		Array (3,N,N) : floats (meters)
			where N is the number of antennas active
		"""
		try:
			return self.BX,self.BY,self.BZ
		except Exception as e:
			print(e)
			return None
global_frame = 0
class SkyImage:
	def __init__(self,path="./models/",filename="galaxy_lobes.png",declination = -90 * u.deg,pixel_size = 0.3 * u.arcsec):
		"""Description
		==============
		Initializes the SkyImage object, the image representation of an astronomical target. 
		Arguments
		---------
		path : str
			path to the folder with images
		filename : str
			name of the file containing the image
		declination : float (u.deg)
			declination in degrees with units
		pixel_size : float (u.arcsec)
			size of a pixel in arcseconds
		"""
		self.path = path
		self.filename = filename
		self.pixel_size = pixel_size
		self.declination = declination
		#used for webcam mode and to keep track if an image has changed
		self._webcam_mode = False
		self._loaded_img = False
	def load_image(self,path = None,filename = None,new_size = (480,480),video_stream = None):
		"""Description
		==============
		Load and process image data into memory. Sets the data attribute to an image value, 
		or to a video stream. Sets the size of the image, imsize_Y,imsize_X attributes
		Arguments
		-----------
		path : str
		filename : str
		new_size : (float,float) 
			sets the size in pixels of the image in memory
		video_stream : video stream from cv2, default none
			the video stream from cv2 when using webcam
		"""
		#set this flag to true, we've changed to a new image or image source
		self._loaded_img = True
		if path is None: 
			path = self.path
		else:
			self.path = path
		if filename is None: 
			filename = self.filename
		else:
			self.filename = filename
		if "Misc4.jpg" in filename:
			#if this above image is selected, we go into webcam mode, and the 
			#data parameter is set to the video stream, the webcam mode attribute
			#is set to true.

			self._webcam_mode = True
			#global global_frame
			#print(global_frame,"load_image")
			#global_frame+=1
			try:
				self.data = video_stream
				#some fuckery for the old fourier transform mode, it works but 
				#probablyt not needed anymore
				if self.data.shape[1] % 2 == 0:
					self.data=np.hstack([self.data,[[0],]*self.data.shape[0]])
				if self.data.shape[0] % 2 == 0:
					self.data=np.vstack([self.data,[0,]*self.data.shape[1]])
				
				self.imsize_Y,self.imsize_X = self.data.shape
			except Exception as e:
				logging.exception(e,"antpos"+str(traceback.format_exc()))
			
		else:
			#if not the above image name, we're not in webcam mode
			#image is opened and loaded from disk, converted to monochrome

			self._webcam_mode = False
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
	def make_invert(self,video_stream = None):
		"""Description
		==============
		Does the Fourier transform of the image, i.e. the data attribute, can be static or video source
		and sets it as the fft_data. Also sets the sizes of the fft image as fftsize_X and fftsize_Y
		Only if the image that was loaded is new.
		"""
		if self._loaded_img is True:
			#if we've just loaded a new image then perform the fft on it.
			

			#global global_frame
			#print(global_frame,"inver_image")
			#global_frame+=1
			
			self.fft_data = np.fft.ifftshift(np.fft.fft2(self.data))
			self.fftsize_X,self.fftsize_Y = self.fft_data.shape
			
			#now that the image has been loaded, set the _loaded_img flag to false. Calling this funciton
			#again on the same image will not perform the costly fft.
			self._loaded_img = False
		else:
			#if a new image has not been loaded
			#but we're in webcam mode, then the image is changing anyway
			#since it's a video stream, so do the ffft on the video_stream
			#for some reason it's loading it again. Need to check if this is right
			#(but it works) Might explain the smoother run when webcam is selected...
			if self._webcam_mode is True:
				self.load_image(video_stream = video_stream)
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
		self.rot_varname = None
		self.rot_value = None
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
		#print("ret val",r0,r1,r2,varname,value)
		self.rot_varname = varname
		self.rot_value = value
		return varname, value
	
	def update_obs_from_control(self,last_measurements,video_stream = None):
		try:
			but_pos = self.Control.get_state(last_measurements)
			#print("but pos", but_pos,tuple([but_pos[_bid].get_state() for _bid in self.Control.button_ids])

			varname,value = self._return_val_button(but_pos,self.var_dic)
			#print("setting: ",varname,value)
		except Exception as e:
			print(e,"Error in button readout, skipping.")
			return 0
		if varname == "hr_angle":
			self._buf_HA_END =  Angle(value, u.hour) + self._buf_HA_END - self._buf_HA_START
			self._buf_HA_START = Angle(value, u.hour)
			if but_pos['S0'].get_state() == 1:
				self.HA_END = self._buf_HA_END
				self.HA_START = self._buf_HA_START

		if varname == "int_time":
			if value >= 12:
				self._buf_HA_START =  Angle(-6.,u.hour)
				self._buf_HA_END = Angle(+6,u.hour)
			else:
				self._buf_HA_END = self._buf_HA_START + Angle(value, u.hour)
			if but_pos['S0'].get_state() == 1:
				self.HA_END = self._buf_HA_END
				self.HA_START = self._buf_HA_START
		
		if varname == "obj_dec":
			self._buf_declination = value * u.deg
			if but_pos['S0'].get_state() == 1:
				self.SkyImage.declination = self._buf_declination
		if varname == "obs_freq":
			self._buf_obs_frequency = value * u.GHz
			if but_pos['S0'].get_state() == 1:
				self.obs_frequency = self._buf_obs_frequency
				self.obs_lam = C.c/self.obs_frequency
		if varname == "img_file":
			self._buf_image_filename = value
			if but_pos['S0'].get_state() == 1:
				self.SkyImage.load_image(filename = self._buf_image_filename,video_stream = video_stream)
		#if but_pos['S1'].get_state() == 1:
		#	self.HA_END = self._buf_HA_END
		#	self.HA_START = self._buf_HA_START
		#	self.SkyImage.declination = self._buf_declination
		#	self.obs_frequency = self._buf_obs_frequency
		#	self.obs_lam = C.c/self.obs_frequency
		#	self.SkyImage.select_image(filename = self._buf_image_filename)
		self.SkyImage.make_invert(video_stream = video_stream)	
		#print("loaded: ",self.SkyImage._loaded_img,"webmode: ",self.SkyImage._webcam_mode)

	def update_obs(self,last_measurements = None,obs_frequency = 180*u.GHz,HA_START=Angle(-4,u.hour),HA_END=Angle(4,u.hour),sample_freq=0.05*u.Hz,EL_LIMIT = 10 * u.deg):
		#obsolete
		self.obs_frequency = obs_frequency
		self.obs_lam = C.c / obs_frequency
		self.sample_freq = sample_freq
		self.HA_START = HA_START
		self.HA_END = HA_END
		self.N_samples = int(((self.HA_END - self.HA_START).hour * u.hour * sample_freq ).decompose() ) + 1
		self.EL_LIMIT = EL_LIMIT
		#select some other config based on measurements dict
		if last_measurements is not None:
			self._set_ctrl_state(last_measurements[self._read_source_id])


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
		if len(self.Observatory.ant_pos_X) > 1:
			self.U = (np.array([self.Observatory.BX.value * sinha + self.Observatory.BY.value * cosha for sinha,cosha in self.sin_cos_ha]) * self.Observatory.BX.unit /  self.obs_lam).decompose() *1./u.radian

			self.V = (np.array([-self.Observatory.BX.value * np.sin(self.SkyImage.declination) * cosha + self.Observatory.BY.value * np.sin(self.SkyImage.declination) * sinha + self.Observatory.BZ.value*np.cos(self.SkyImage.declination) for sinha,cosha in self.sin_cos_ha]) * self.Observatory.BX.unit / self.obs_lam).decompose() *1./u.radian
		else:
			self.U = np.array([0])
			self.V = np.array([0])
		#self.U = (-np.tril(self.U) + np.triu(self.U))*1./u.radian
		#self.V = (-np.tril(self.V) + np.triu(self.V))*1./u.radian
		#print(self.V[0],self.SkyImage.declination)
	def grid_uv_coverage(self):
		"""Knowing the pixelsize in radians and the image size in pixels, grid the UV coverage, returning integer values which correspond to indices (positions) in the
		fourier transformed image of the sky.
		"""

		if len(self.Observatory.ant_pos_X) > 1:
			self.u = np.int16(((self.U + 1./self.SkyImage.pixel_size.to(u.radian))/2 * self.SkyImage.pixel_size.to(u.radian) * self.SkyImage.imsize_X ).decompose().value)
			self.v = np.int16(((self.V + 1./self.SkyImage.pixel_size.to(u.radian))/2 * self.SkyImage.pixel_size.to(u.radian) * self.SkyImage.imsize_Y ).decompose().value)
		else:
			self.u = np.int16([0])
			self.v = np.int16([0])
		#print(self.v[0],self.SkyImage.declination)
	def make_masked_arr(self,weights = "natural"):
		"""Mask the Fourier transform of the sky. Only the values at indices corresponding to positions in the UV plane, given by u, v, are kept, the 
		rest are set to zero. Since the sky is real valued, its fourier transform is a hermitian-symmetric array, so we use the real fourier transform function, with
		the negative index values in u and v discarded, since they correspond to hermitian-symmetric values. Index values outside the Fourier transform array are also discarded. 
		"""

		self.UVC = np.zeros(self.SkyImage.fft_data.shape)
		#get the pairs of u,v samples and count how many times they appear in a cell (weights)

		if len(self.Observatory.ant_pos_X) > 1:
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
		#uvc = self.UVC[:,:self.SkyImage.fft_data_r.shape[1]]
		#self.uv_fft = uvc * self.SkyImage.fft_data_r
		if len(self.Observatory.ant_pos_X) > 1:
			self.uv_fft_sampled = self.UVC * self.SkyImage.fft_data.T
			#self.uv_fft_i = self.UVC * self.SkyImage.fft_data_i
			#self.dirty_beam = np.fft.ifftshift(np.fft.irfft2(uvc))
			self.dirty_beam = np.fft.ifftshift(np.fft.ifft2(self.UVC))
			self.dirty_image = np.fft.ifft2(np.fft.ifftshift(self.uv_fft_sampled.T))
			#self.dirty_i = np.fft.ifft2(self.uv_fft_i)
		else:
			self.uv_fft_sampled = 0 * self.SkyImage.fft_data.T + 1
			self.dirty_beam = 0 * self.UVC + 1
			self.dirty_image = self.SkyImage.data
			#we add a factor of 2 to make it more crappy looking with single dish.
			fwhm = 2 * 1.02 * (self.obs_lam / (12. * u.m)).decompose() * (1 * u.radian).to(u.arcsec) / self.SkyImage.pixel_size
			sigma = fwhm.decompose().value / 2.35
			self.dirty_image = gaussian_filter(self.SkyImage.data,sigma = sigma,mode = 'nearest')

