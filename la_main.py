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
import traceback
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
from la_control import Control,id_usb_conn
from la_obs_img import Observatory,SkyImage,Observation
from la_disp_man import BlitManager,DisplayManager
from la_reader import Reader,ReadController,VideoReadController
import logging 
import la_config 	

def vprint(*args,verbose = True):
	if verbose:
		print(*args)
	else:
		pass

def norm(x):
	return (x - np.min(x))/(np.max(x) - np.min(x))

if __name__ == "__main__":
	test_conns = True
	logging.basicConfig(filename='lego_alma.log')
	with Manager() as manager:
		#define the connections initialize readers and vars

		plt.style.use('dark_background')
		sleep_time = 0.1
		ser0_controller = ReadController(sleep_time = sleep_time)
		ser1_controller = ReadController(sleep_time = sleep_time)
		ble_controller = ReadController(sleep_time = sleep_time)
		vid_controller = VideoReadController()
		#ble_controller = ReadController()

		last_measurements = manager.dict()
		video_dict = manager.dict()
		ser_conn_0 = {'path':'/dev/ttyUSB0','baud':19200,'timeout':1,'parity':serial.PARITY_NONE,'rtscts': False}
		ser_conn_1 = {'path':'/dev/ttyUSB1','baud':19200,'timeout':1,'parity':serial.PARITY_NONE,'rtscts': False}
		ble_conn = 0x66ce

		#start processes to read the data 
		p0 = Process(target = ser0_controller._loop_read,args = (last_measurements,ser_conn_0,'ser'),kwargs={'verbose': False})
		p1 = Process(target = ser1_controller._loop_read,args = (last_measurements,ser_conn_1,'ser'),kwargs={'verbose': False})
		p2 = Process(target = ble_controller._loop_read,args = (last_measurements,ble_conn,'ble'))
		pv = Process(target = vid_controller.read_vid,args  = (video_dict,))
		p0.start()
		p1.start()
		pv.start()
		p2.start()
		print("Loading BLE, please wait...")
		time.sleep(5)

		#initialize observatory
		alma = Observatory(ant_pos_bit_file='./ant_pos.2.txt')
		alma.load_ant_pos_bit_file()
		ant_usb_id,ctrl_usb_id = id_usb_conn(last_measurements)
		src_list = [ant_usb_id]#,'DW4F0B','DWD900']#,'DW4F0B','DW5293','DW912D','DWD900']
		ctrl_list = [ctrl_usb_id]
		alma.set_read_source_ids(src_list)
		alma.transform_query(last_measurements,['DW5293','DW4F0B','DW912D']) #let's callibrate with these i guess...
			
		alma.set_ant_pos(last_measurements)
		alma.make_baselines()
		
		#initialize the first image
		imgobj = SkyImage(path = "./models/",filename = "galaxy_lobes.png")
		imgobj.load_image()
		imgobj.make_invert()

		#initialize the buttons
		cont = Control(ctrl_id = ctrl_usb_id)
		cont.add_buttons()
		cont.set_but_bit_dict_file("but_dict.pickle")
		cont.load_but_bit_dict()
		cont.set_swi_bit_list(la_config.swi_bits)
		cont.get_state(last_measurements)	

		#initialize the observation
		obs = Observation(alma,imgobj,cont,obs_frequency = 200*u.GHz,var_dic = la_config.var_dic)
		obs.calc_el_curve()
		obs.make_uv_coverage()
		obs.grid_uv_coverage()
		obs.make_masked_arr()
		obs.make_dirty_arr()
		
		#initialize the display manager and plots to start blitting
		dm = DisplayManager()
		dm.setup_main_figure()
		dm.init_ant_plot(maxoffset = 120)
		dm.init_ant_proj_plot()
		pixel_size = obs.SkyImage.pixel_size
		print(pixel_size)
		inv_pixel_size = (1./pixel_size).to(1./u.radian)
		print(inv_pixel_size)
		dm.init_img_plot(pixel_size = pixel_size)
		dm.init_fft_plot()
		dm.init_uvc_plot(inv_pixel_size = inv_pixel_size)
		dm.init_dbe_plot(pixel_size = pixel_size)
		dm.init_dim_plot(pixel_size = pixel_size)
		dm.init_mft_plot(inv_pixel_size = inv_pixel_size)
		dm.init_ind_plot(la_config.var_dic,obs)
		dm.setup_blit_manager()
		#dm.update_ant_plot((alma.ant_pos_EW,alma.ant_pos_NS))
		#dm.update_ant_proj_plot((alma.ant_pos_EW,alma.ant_pos_NS))
		#dm.update_img_plot(imgobj.data)
		#frequency_list = [1,10,100,200,400,500,1000]
		#frequency_list = [_*u.GHz for _ in frequency_list]
		#freq_cycle = itertools.cycle(frequency_list)
		sleep(0.5)


		#start the main loop
		while True:
			try:	
				#do interferometry stuff
				obs.Observatory.set_ant_pos(last_measurements)
				obs.Observatory.make_baselines()
				obs.update_obs_from_control(last_measurements,video_stream = video_dict['val'])
				obs.calc_el_curve()
				obs.make_uv_coverage()
				obs.grid_uv_coverage()
				obs.make_masked_arr(weights="uniform")
				obs.make_dirty_arr()
				pixel_size = obs.SkyImage.pixel_size
				#print(f"pixel_size = {pixel_size}")
				inv_pixel_size = (1./pixel_size).to(1./u.radian)
				#print("Main loop data:")
				#print(last_measurements)
				#print((obs.Observatory.ant_pos_EW,obs.Observatory.ant_pos_NS))
				#print("end main loop data")
				#update plots

				dm.update_ant_plot((obs.Observatory.ant_pos_EW,obs.Observatory.ant_pos_NS))
				dm.update_ant_proj_plot((obs.Observatory.ant_pos_EW,obs.Observatory.ant_pos_NS),
						obs.Observatory.latitude,
						obs.HA_START + abs(obs.HA_END - obs.HA_START)/2,
						obs.SkyImage.declination)

				dm.update_img_plot(obs.SkyImage.data,pixel_size = pixel_size)
				dm.update_fft_plot(obs.SkyImage.fft_data)

				dm.update_uvc_plot(obs.UVC,inv_pixel_size = inv_pixel_size)
				dm.update_dbe_plot(obs.dirty_beam,pixel_size = pixel_size)

				dm.update_dim_plot(obs.dirty_image,pixel_size = pixel_size)
				dm.update_mft_plot(obs.uv_fft_sampled,inv_pixel_size = inv_pixel_size)
				dm.update_ind_plot(la_config.var_dic,obs)
				dm.update_blit_manager()
				#plt.scatter(alma.ant_pos_EW,alma.ant_pos_NS,c='blue')
				#obs = Observation(almaobj,imgobj)
				#obs.calc_el_curve()
				#plt.show()
	
			except KeyboardInterrupt: #to exit
				p0.terminate()
				p1.terminate()
				#p2.terminate()
				pv.terminate()
				plt.close()
				sys.exit()
				try:
					sys.exit(130)
				except SystemExit:
					os._exit(130)
			except Exception as e: #report an error in main loop and exit.
				trace = traceback.format_exc()
				print(e,"mainloop!!!!!!",trace)
				logging.exception(e,str(trace))

				p0.terminate()
				p1.terminate()
				p2.terminate()
				pv.terminate()
				plt.close()
				sys.exit()
				try:
					sys.exit(130)
				except SystemExit:
					os._exit(130)
		p0.terminate()
		p1.terminate()
		p2.terminate()
		pv.terminate()










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



