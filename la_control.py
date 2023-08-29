import pickle
import numpy as np
import operator

class Control:
	def __init__(self,ctrl_id = 'USB1',but_dict_file = None,separating_bit = 35,swi_bit_list = [],rot_bit_list = [],button_ids = ["R0","R1","R2","S0","S1","S2"]):
		self.ctrl_id = ctrl_id
		self.separating_bit = separating_bit
		self.but_dict_file = but_dict_file
		self.button_dict = dict()
		self.swi_bit_list = swi_bit_list
		self.rot_bit_list = rot_bit_list
		self.button_ids = button_ids
		self.all_button_bits = [0]#tuple(rot_bits, swi_bits)
		self.rot_switch_pos =  [0]#tuple(rot_pos),tuple(swi_pos)
		if self.rot_bit_list == []:
			self.rot_bit_list = np.array([[5,13,21,29],[6,14,22,30],[7,15,23,31]])
	def set_rot_bits(self,filename):
		try:
			self.rot_bit_list = np.loadtxt(filename)
			return 1
		except Exception as e:
			print(e,"unable to load rot but bit list")
			return 0

		
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
		#print(last_measurements[self.ctrl_id])
		rot_bits =tuple(last_measurements[self.ctrl_id][last_measurements[self.ctrl_id] < self.separating_bit])
		#print(rot_bits)
		swi_bits = tuple((last_measurements[self.ctrl_id][last_measurements[self.ctrl_id] > self.separating_bit])[:-1])
		return tuple(rot_bits),tuple(swi_bits)

	def get_all_pos(self,last_measurements,use_dict = False):
		rot_bits, swi_bits = self._read_button_bits(last_measurements)
		if use_dict == True:
			rot_pos = self.but_bit_dict[rot_bits]
		else:
			button_bits = self.rot_bit_list#np.array([[5,13,21,29],[6,14,22,30],[7,15,23,31]])
			rot_pos = [0,0,0]
			for _b in rot_bits:
				i,ans = np.where(button_bits == _b)
				rot_pos[i[0]] = ans[0]
			
		swi_pos = list(np.array([_b in swi_bits for _b in self.swi_bit_list])*1)
		self.all_button_bits = tuple(rot_bits), tuple(swi_bits)
		self.rot_switch_pos =  tuple(rot_pos),tuple(swi_pos)
		#print(self.swi_bit_list,swi_bits,swi_pos)
		#print(rot_pos,swi_pos,"POS")
		return tuple(rot_pos),tuple(swi_pos)
		
	def set_all_buttons(self,last_measurements):
		all_states = operator.add(*self.get_all_pos(last_measurements))
		#print("setting all states",all_states)
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


def id_usb_conn(last_measurements, use_last_bit = True, discriminant = 6):
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
		print([[key,result] for key,result in last_measurements.items()])
		ctrl_usb_src = [key for key,result in last_measurements.items() if 'USB' in key and 63 in result]
		ant_usb_src = [key for key,result in last_measurements.items() if 'USB' in key and 63 not in result]
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
		print(last_measurements)
		ant_usb_src = [key for key,result in last_measurements.items() if 'USB' in key and len(result) >= discriminant]
		ctrl_usb_src = [key for key,result in last_measurements.items() if 'USB' in key and len(result) < discriminant]
		if ant_usb_src == ctrl_usb_src:
			print("error, cannot differentiate between USB antenna and USB ctrl connection")
			return False
		else:
			print("Found antenna data connections:",ant_usb_src)
			print("Found telescope control connection:",ctrl_usb_src)
			return ant_usb_src[0],ctrl_usb_src[0]



