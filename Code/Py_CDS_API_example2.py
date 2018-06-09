#!/usr/bin/env python
#
# Multiple time example: https://software.ecmwf.int/wiki/display/COPSRV/CDS+web+API+%28cdsapi%29+training
#
import cdsapi
from datetime import datetime, timedelta
 
p_levels = [str(z)  for z in ([1] + list(range(50, 1050, 50)))]
c = cdsapi.Client()
 
 
def days_of_month(y, m):
    d0 = datetime(y, m, 1)
    d1 = datetime(y, m + 1, 1)
    out = list()
    while d0 < d1:
        out.append(d0.strftime('%Y-%m-%d'))
        d0 += timedelta(days=1)
    return out
 
for y in range(2008, 2018):
    for m in range(1,13):
        for d in days_of_month(y, m):
            c.retrieve("reanalysis-era5-pressure-levels",
                       {
                           "variable": "temperature",
                           "pressure_level": p_levels,
                           "product_type": "reanalysis",
                           "date": d,
                           "time":[
                               '00:00','01:00','02:00',
                               '03:00','04:00','05:00',
                               '06:00','07:00','08:00',
                               '09:00','10:00','11:00',
                               '12:00','13:00','14:00',
                               '15:00','16:00','17:00',
                               '18:00','19:00','20:00',
                               '21:00','22:00','23:00'
                           ],
                           "format": "netcdf"
                       },
                       "ea_t_{day}.nc".format(day=d)
                       )