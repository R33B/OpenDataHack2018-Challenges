
import numpy as np
from netCDF4 import Dataset as NetCDFFile

############################################################################
def WindComponents_to_hsv(arr_1, arr_2):

	import cv2

	# Transform flow components to HSV image (alpha channel pixel intensity based on WS, color [colorwheel] based on WD)
	numRow,numCol = np.shape(arr_1)
	hsv = np.zeros((numRow,numCol,3), dtype=np.uint8)
	hsv[...,1] = 255
	mag, ang = cv2.cartToPolar(arr_1, arr_2)
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = 255	#cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	rgba = np.zeros((numRow, numCol, 4))
	rgba[:,:,:3] = rgb
	rgba[:,:,3] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	return rgba

############################################################################
def netCDF_to_arrayTrim(nc, trim, strVar):
	array_lat = nc.variables['latitude'][:].data		# [ 90. 89.75 89.5 89.25 ... -89.5 -89.75 -90.0]
	array_lon = nc.variables['longitude'][:].data	# [  0.  0.25 0.5 0.75 1. ... 359.25 359.5 359.75]

	# Cell-corners positions  {Top-Left to Bottom-Right}
	iniLat = trim[0]
	finLat = trim[1]
	iniLon = trim[2]
	finLon = trim[3]
	trim_lat = np.arange(iniLat, finLat-0.001, -0.25)
	trim_lon = np.arange(iniLon, finLon+0.001, 0.25)
	trim_lon[trim_lon<0] = trim_lon[trim_lon<0] + 360.

	index_lat = np.where(np.in1d(array_lat, trim_lat, assume_unique=True))[0]
	index_lon = np.searchsorted(array_lon, trim_lon)

	# Trim variable
	var = nc.variables[strVar][:].data
	c,r = np.meshgrid(index_lon, index_lat)
	arr = var[:,r.flatten(), c.flatten()]
	arr = arr.reshape(var.shape[0], len(index_lat), len(index_lon))
	return arr

############################################################################
def netCDF_to_JSON(nc, trim, strVar):

	list_Files = glob.glob('./data/netCDF/ea_t_single_levels_*.nc')
	list_Files.sort()

	df = pd.DataFrame()
	for strFile in list_Files:

		# Load netCDF
		nc = NetCDFFile(strFile)
		u_100 = nc.variables['u_100'][:]
		v_100 = nc.variables['v_100'][:]

		# Datetime
		dateRef = datetime.datetime.strptime(strFile.split('/')[-1], 'ea_t_single_levels_%Y-%m-%d.nc')

		df = df.append({'datetime': dateRef, 'numRow': u_100.shape[0], 'numCol': u_100.shape[1], 'u_100': u_100.flatten(), 'v_100': v_100.flatten()})


	# Save DataFrame as JSON
	df.to_json('dataset.json')


############################################################################
if __name__ == "__main__":

	iniLat = 48.0
	finLat = 32.0
	iniLon = -16.0
	finLon = 8.0
	trim = [iniLat, finLat, iniLon, finLon]

	nc = NetCDFFile('download_24h.nc')
	strVar = 't'
	
	arr = netCDF_to_arrayTrim(nc, trim, strVar)
	print 'arr: ', arr.shape
	'''
	# WindComponents_to_hsv
	arr_1 = np.random.rand(29,61)
	arr_2 = np.random.rand(29,61)
	WindComponents_to_hsv(arr_1, arr_2)
	'''
