#!/usr/bin/env python
#
# Multiple time example: https://software.ecmwf.int/wiki/display/COPSRV/CDS+web+API+%28cdsapi%29+training
#
import cdsapi
import time_tools as tto

########################################################################################################################
# settings
########################################################################################################################

# p_levels = [str(z) for z in ([1] + list(range(50, 1050, 50)))]
p_levels = [900]

min_year = 2016
max_year = 2017
min_month = 1
max_month = 13

########################################################################################################################
# execution
########################################################################################################################

c = cdsapi.Client()
 
for y in range(min_year, max_year):
    for m in range(min_month, max_month):
        for d in tto.days_of_month(y, m):
            if False:
                c.retrieve("reanalysis-era5-pressure-levels",
                           {
                               "variable": "temperature",
                               "pressure_level": p_levels,
                               "product_type": "reanalysis",
                               "date": d,
                               "time": [
                                   '00:00', '01:00', '02:00',
                                   '03:00', '04:00', '05:00',
                                   '06:00', '07:00', '08:00',
                                   '09:00', '10:00', '11:00',
                                   '12:00', '13:00', '14:00',
                                   '15:00', '16:00', '17:00',
                                   '18:00', '19:00', '20:00',
                                   '21:00', '22:00', '23:00'
                               ],
                               "format": "netcdf"
                           },
                           "ea_t_p_levels{day}.nc".format(day=d)
                           )

            if True:
                c.retrieve('reanalysis-era5-single-levels',
                           {
                            'variable': ['100m_u_component_of_wind', '100m_v_component_of_wind',
                                         '10m_u_component_of_wind', '10m_v_component_of_wind'],
                            'product_type': 'reanalysis',
                            'date': d,
                            'time': [
                                '00:00', '01:00', '02:00',
                                '03:00', '04:00', '05:00',
                                '06:00', '07:00', '08:00',
                                '09:00', '10:00', '11:00',
                                '12:00', '13:00', '14:00',
                                '15:00', '16:00', '17:00',
                                '18:00', '19:00', '20:00',
                                '21:00', '22:00', '23:00'
                                ],
                                'format': 'netcdf'
                            },
                            "data\\ea_t_single_levels_{day}.nc".format(day=d)
                            )
