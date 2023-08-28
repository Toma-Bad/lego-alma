import numpy as np
from matplotlib import colors,cm
import matplotlib.pyplot as plt
from astropy.io import ascii
from matplotlib import colors,cm
import operator
import itertools
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
	def __init__(self,maxoffset = 250):
		self._blit_manager_list = []
		self._axes = []
		self._ln_arr = []
		self.maxoffset = maxoffset
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
		

	def init_ant_plot(self, maxoffset = None, row = 0, col = 0):
		if maxoffset is None:
			maxoffset  = self.maxoffset 
		else:
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
	
	def init_ant_proj_plot(self, row = 1, col = 0):
		maxoffset = self.maxoffset
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
		dummy_img = np.zeros((size,size))+0.5
		ii = np.ravel_multi_index((row,col),(self._nrows,self._ncols))
		#im = self._axes[ii].imshow(dummy_img,vmin = 0.,vmax = 1.,cmap="rainbow")
		im = self._axes[ii].imshow(dummy_img,vmin = 0.,vmax = 1.,cmap="hot")
		self._axes[ii].axes.get_xaxis().set_visible(False)
		self._axes[ii].axes.get_yaxis().set_visible(False)
		self._ln_arr[ii] = im
	def init_mft_plot(self,size=480,row= 1,col = 3):
		dummy_img = np.zeros((size,size))
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
				self._ind_text.append(self._axes[ii].text(0.3 + 0.65/len(self.rot_text_menu[0])*_xx,0.5 - 0.75/len(self.rot_text_menu) * _yy, self.rot_text_menu[_yy][_xx],transform=self._axes[ii].transAxes,color="white",ha="left", va="center",bbox=dict(boxstyle="square,pad=0.3",fc="black", ec="steelblue", lw=2)))
		status_text = "HA Start: {} \nHA End: {} \nDec: {} \nInt time: {} \nFrequency: {} \nImage file: {}".format(Observatory.HA_START,
				Observatory.HA_END,
				Observatory.SkyImage.declination,
				abs(Observatory.HA_END-Observatory.HA_START),
				Observatory.obs_frequency,
				Observatory.SkyImage.filename) 		

		self._ind_status_text = self._axes[ii].text(0.0,0.3,status_text,transform = self._axes[ii].transAxes,color="white",ha="left",va="center",bbox=dict(boxstyle="square,pad=0.1",fc="black", ec="steelblue", lw=2))
		self._ind_now_setting_text = self._axes[ii].text(0.15,0.3," ",transform = self._axes[ii].transAxes,color="white",ha="left",va="center",bbox=dict(boxstyle="square,pad=0.1",fc="black", ec="steelblue", lw=2))
		
	#	var_dic = \
	#		{0:[['HA']*4,list(zip([operator.add]*4,d00)),[d01]*4],
	#		 1:[['Dec']*4,list(zip([operator.add]*4,d10)),[d11]*4],
	#		 2:[['Int']*2+ ['obs_frequency']*2,list(zip([operator.mul]*2+[operator.add]*2,d20)),[d210]*2+[d211]*2],
	#		 3:[['Fname']*4,list(zip([operator.add]*4,d30)),[d31]*4]}
		


	def setup_blit_manager(self):
		artist_list = [_ for _ in self._ln_arr if _ is not None] + self._ind_text + [self._ind_status_text,self._ind_now_setting_text]
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
		#print(data,(-maxoffset < data[0])&(data[0] < maxoffset),(-maxoffset < data[1])&(data[1] < maxoffset),data,maxoffset,"arrayloc")
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
	def update_ant_proj_plot(self,data,observatory_latitude,hrangle,dec,row = 1, col = 0):
		maxoffset = self.maxoffset
		ln_ind = np.ravel_multi_index((row,col),self._axes_shape)
		data = np.array(data)
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
		#print("HRANGLE: ",hrangle)
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
		plt_data = np.log10(np.abs(data))
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
	def update_ind_plot(self,var_dic,Observation,param_disp_names = None):
		var_dic = Observation.var_dic
		r0,r1,r2,s1,s2,s3 = tuple([Observation.Control.button_dict[_bid].get_state() for _bid in Observation.Control.button_ids])
		upper_text = ['0','1','2','3']
		if param_disp_names is None:
			vdnt = {
					"hr_angle":"Hr Angle",
					"obj_dec":"Declin",
					"int_time":"Int Time",
					"obs_freq":"Obs Freq",
					"img_file":"Img File"
					}
		else:
			vdnt = dict(zip([var_dic[r0][0][_ir1] for _ir1 in range(len(var_dic[r0][0]))],param_disp_names))

		op_to_text = {operator.add : '+', operator.mul : 'x'}	
		mid_text = [vdnt[var_dic[r0][0][_ir1]]+" "+ var_dic[r0][3][_ir1]+": "+ str(var_dic[r0][1][_ir1][1]) for _ir1 in range(len(var_dic[r0][0]))]
		low_text = [op_to_text[var_dic[r0][1][r1][0]] + " " + str(var_dic[r0][2][r1][_ir2]) for _ir2 in  range(len(var_dic[r0][0]))]
		self.rot_text_menu = [upper_text,mid_text,low_text]
		flat_text_menu = list(itertools.chain.from_iterable(self.rot_text_menu))
		_it = 0
		for _yy in range(len(self.rot_text_menu)):
			for _xx in range(len(self.rot_text_menu[0])):
				self._ind_text[_it].set_text(flat_text_menu[_it])
				if (_yy == 0 and _xx == r0) or (_yy == 1 and _xx == r1) or (_yy == 2 and _xx == r2):
					self._ind_text[_it].set_bbox(dict(facecolor="red"))
					self._ind_text[_it].set_fontsize(15)
				else:
					self._ind_text[_it].set_bbox(dict(facecolor="black"))
					self._ind_text[_it].set_fontsize(12)
				_it += 1
		status_text = "HA Start: {} \nHA End: {} \nDec: {} \nInt Time: {} \nFrequency: {} \nImage file: {}".format(Observation.HA_START,
				Observation.HA_END,
				Observation.SkyImage.declination,
				abs(Observation.HA_END-Observation.HA_START),
				Observation.obs_frequency,
				Observation.SkyImage.filename) 		
		try:
			now_setting_text = "Now setting:\n "+vdnt[var_dic[r0][0][r1]]+" = {:.1f}".format(Observation.rot_value)
		except:
			now_setting_text = "Now setting:\n "+vdnt[var_dic[r0][0][r1]]+" = {}".format(Observation.rot_value)
		if s1 == 1:
			self._ind_status_text.set_text(status_text)
			self._ind_status_text.set_bbox(dict(facecolor="red"))
			self._ind_now_setting_text.set_text(now_setting_text)
			self._ind_now_setting_text.set_fontsize(15)
			self._ind_now_setting_text.set_bbox(dict(facecolor="red"))
		else:
			self._ind_status_text.set_bbox(dict(facecolor="black"))
			self._ind_now_setting_text.set_bbox(dict(facecolor="black"))
			self._ind_now_setting_text.set_text(now_setting_text)
			self._ind_now_setting_text.set_fontsize(15)



