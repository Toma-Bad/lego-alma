import serial
from time import sleep
import decawave_ble
import glob
from serial.tools.list_ports import comports
import numpy as np
import cv2


class VideoReadController:
	def __init__(self,resolution = (480,480)):
		self.resolution = resolution
	def read_vid(self,shared_var):
		VidObj = cv2.VideoCapture(0)
		VidObj.set(4,self.resolution[1])
		ret,frame = VidObj.read()
		#print(frame.shape)
		frame = np.asarray(frame)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#print(frame.shape,self.resolution)
		img_shape = frame.shape
		minx = np.max((0,int((img_shape[0] - self.resolution[0])/2)))
		maxx = np.min((img_shape[0],int((self.resolution[0] + img_shape[0])/2)))
		miny = np.max((0,int((img_shape[1] - self.resolution[1])/2)))
		maxy = np.min((img_shape[1],int((self.resolution[1] + img_shape[1])/2)))
		#print("A",minx,maxx,miny,maxy)	

		while True:
			ret,frame = VidObj.read()
			frame = np.asarray(frame)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#print(frame.shape)
			shared_var['val'] = frame[minx:maxx,miny:maxy]
		VidObj.release()
		
		
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



	def create_ser_conn(self,path='/dev/ttyUSB0',baud=19200,timeout=1,parity=serial.PARITY_NONE,rtscts=False,auto_search_free = True):
		print("Creating connection to " + path)
		print("Existing ports:")
		port_list = comports()
		for port in port_list:
			print(port)
		if path in [_p.device for _p in port_list]:
			self.ser_conn_dict = {'info':[path,baud,timeout,parity,rtscts],'connection':serial.Serial(path,baud,timeout = timeout, parity = parity, rtscts = rtscts,exclusive = True)}
			return 1
		elif auto_search_free is True:
			print("Path "+path+" not found, looking for next")
			for i in range(10):
				try:
					path = '/dev/ttyUSB'+str(i)
					self.ser_conn_dict = {'info':[path,baud,timeout,parity,rtscts],'connection':serial.Serial(path,baud,timeout = timeout, parity = parity, rtscts = rtscts,exclusive = True)}
					return i
				except Exception as e:
					print(i,e,"can't connect to this port, trying next")
			print("no ports found")
			return 0
		
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


class ReadController:
	def __init__(self,sleep_time = 0.05):
		self.sleep_time = sleep_time
	
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


