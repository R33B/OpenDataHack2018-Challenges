
import numpy as np
from netCDF4 import Dataset as NetCDFFile

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
	t = nc.variables[strVar][:].data
	c,r = np.meshgrid(index_lon, index_lat)
	arr = np.squeeze(t)[r.flatten(), c.flatten()]
	arr = arr.reshape(r.shape)
	return arr

############################################################################
if __name__ == "__main__":

	iniLat = 43.0
	finLat = 36.0
	iniLon = -10.0
	finLon = 5.0
	trim = [iniLat, finLat, iniLon, finLon]

	nc = NetCDFFile('download.nc')
	strVar = 't'
	
	netCDF_to_arrayTrim(nc, trim, strVar)