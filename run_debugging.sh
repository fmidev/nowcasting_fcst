#!/bin/bash

python2 debugging.py --obsdata testdata/2019020409/obs_2t.grib2 --modeldata testdata/2019020409/fcst_2t.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi2_2t.grib2 --parameter Temperature
python2 debugging.py --obsdata testdata/2019020409/obs_2r.grib2 --modeldata testdata/2019020409/fcst_2r.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi2_2r.grib2 --parameter Dewpoint
python2 debugging.py --obsdata testdata/2019020409/obs_msl.grib2 --modeldata testdata/2019020409/fcst_msl.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi2_msl.grib2 --parameter msl
python2 debugging.py --obsdata testdata/2019020409/obs_10si.grib2 --modeldata testdata/2019020409/fcst_10si.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi2_10si.grib2 --parameter wind10

python2 debugging.py --obsdata testdata/2019032708/obs_tprate.grib2 --modeldata testdata/2019032708/fcst_tprate.grib2 --bgdata testdata/2019032708/mnwc_tprate.grib2 --interpolated_data testdata/2019032708/output/interpolated_tprate_uusi.grib2 --parameter precipitation_bg_1h

python2 debugging.py --obsdata testdata/2019042611/obs_tprate.grib2 --modeldata testdata/2019042611/fcst_tprate.grib2 --bgdata testdata/2019042611/mnwc_tprate.grib2 --interpolated_data testdata/2019042611/output/interpolated_tprate_uusi.grib2 --parameter precipitation_bg_1h
