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
		ser_conn_0 = {'path':'/dev/ttyUSB0','baud':19200,'timeout':1,'parity':serial.PARITY_NONE,'rtscts':False}
		ser_conn_1 = {'path':'/dev/ttyUSB1','baud':19200,'timeout':1,'parity':serial.PARITY_NONE,'rtscts':False}
		ble_conn = 0x66ce
#		fig,ax = plt.subplots()
#		(ln,) = ax.plot([-100,100],[-100,100],color='blue',marker = "o",linestyle='None',animated = True)
#		current_ranges = np.array([ax.get_xlim(),ax.get_ylim()])
#		print(current_ranges)
#		bm = BlitManager(fig.canvas, [ln,])
		d00 = [-6,-3,0,+3]
		d01 = [0,1,2,3]
		d10 = [-80,-50,-30,-10]
		d11 = [0,5,15,20]
		d20 = [0.2,1,50,500]
		d210 = [1,2,3,12]
		d211 = [0,50,100,250]
		d30 = ['Galaxy','Planet','BH','Misc']
		d31 = ['1.jpg','2.jpg','3.jpg','4.jpg']
		var_dic = \
			{0:[['hr_angle']*4,list(zip([operator.add]*4,d00)),[d01]*4,["[hr]"]*4],
			 1:[['obj_dec']*4,list(zip([operator.add]*4,d10)),[d11]*4,["[deg]"]*4],
			 2:[['int_time']*2+ ['obs_freq']*2,list(zip([operator.mul]*2+[operator.add]*2,d20)),[d210]*2+[d211]*2,["[hr]"]*2+["[GHz]"]*2],
			 3:[['img_file']*4,list(zip([operator.add]*4,d30)),[d31]*4,[" "]*4]}
			
		swi_bits = [52,44,36]


		plt.style.use('dark_background')
		sleep_time = 0.1
		ser0_controller = ReadController(sleep_time = sleep_time)
		ser1_controller = ReadController(sleep_time = sleep_time)
		vid_controller = VideoReadController()
		#ble_controller = ReadController()

		last_measurements = manager.dict()
		video_dict = manager.dict()
		#ser0_controller._loop_read(last_measurements,ser_conn_0,'ser')
		p0 = Process(target = ser0_controller._loop_read,args = (last_measurements,ser_conn_0,'ser'),kwargs={'verbose':False})
		p1 = Process(target = ser1_controller._loop_read,args = (last_measurements,ser_conn_1,'ser'),kwargs={'verbose':False})
		pv = Process(target = vid_controller.read_vid,args  = (video_dict,))
		#p2 = Process(target = ble_controller._loop_read,args=(last_measurements,ble_conn,'ble'))
		p0.start()
		p1.start()
		pv.start()
		time.sleep(3)
		#p2.start()
		#print("Loading BLE, please wait...")
		alma = Observatory(ant_pos_bit_file='./ant_pos.2.txt')
		alma.load_ant_pos_bit_file()
		#print(last_measurements)
		ant_usb_id,ctrl_usb_id = id_usb_conn(last_measurements)
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
		dm.init_ant_proj_plot()
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
				#print(last_measurements)
				#sleep(1)
				#plt.pause(0.1)
				#if len(time_list) > 0:
				#	print((np.asarray(time_list) - np.roll(np.asarray(time_list),1))[1:])
				#print(last_measurements)
				#print(alma.ant_pos_EW,alma.ant_pos_NS)
				#print(obs.Control.all_button_bits,obs.Control.rot_switch_pos)
				#print(imgobj.declination)
				try:
					obs.Observatory.set_ant_pos(last_measurements)
				except Exception as e:
					print(e, "error in set ant pos")
					logging.exception(e,"antpos")
					
				try:
					obs.Observatory.make_baselines()
				except Exception as e:
					print(e, "error in make baselines")
					logging.exception(e,"blines")
				#time.sleep(0.5)	
				#print("B")	
				try:
					obs.update_obs_from_control(last_measurements,video_stream = video_dict['val'])
				except Exception as e:
					print(e, "error in update from controls")
					logging.exception(e,"contro")
				#obs.update_obs()#measurements_dict = last_measurements)#,obs_frequency = next(freq_cycle))
				try:
					obs.calc_el_curve()
				except Exception as e:
					print(e, "error in curve")
					logging.exception(e,"el curve")
				try:
					obs.make_uv_coverage()
				except Exception as e:
					print(e, "error in uv cov")
					logging.exception(e,"uv cov")
				try:
					obs.grid_uv_coverage()
				except Exception as e:
					print(e, "error in grid")
					logging.exception(e,"gridding")
				try:
					obs.make_masked_arr(weights="uniform")
				except Exception as e:
					print(e, "error in masking")
					logging.exception(e,"masking")
				try:
					obs.make_dirty_arr()
				except Exception as e:
					print(e, "error in mk dirty")
					logging.exception(e,"dirty")
				#print(np.max(obs.dirty_beam),np.min(obs.dirty_beam),np.mean(obs.dirty_beam),np.percentile(obs.dirty_beam,[5,15,50,65,95]))
				dm.update_ant_plot((obs.Observatory.ant_pos_EW,obs.Observatory.ant_pos_NS))
				dm.update_ant_proj_plot((obs.Observatory.ant_pos_EW,obs.Observatory.ant_pos_NS),
						obs.Observatory.latitude,
						obs.HA_START + abs(obs.HA_END - obs.HA_START)/2,
						obs.SkyImage.declination)

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
				pv.terminate()
				plt.close()
				sys.exit()
				try:
					sys.exit(130)
				except SystemExit:
					os._exit(130)
			except Exception as e:
				print("error! wtf?",e)
				print(traceback.format_exc())
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
		p0.terminate()
		p1.terminate()
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



