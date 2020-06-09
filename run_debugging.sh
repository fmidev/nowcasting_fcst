#!/bin/bash
# This is a debugging script that runs with example data in the Git repo and outputs diagnostics plots from the runs to directory figures/

mkdir -p figures/fields
mkdir -p figures/jumpiness_absdiff
mkdir -p figures/jumpiness_meandiff
mkdir -p figures/jumpiness_ratio
mkdir -p figures/linear_change
mkdir -p figures/linear_change3h
mkdir -p figures/linear_change4h

### OLD DEBUGGING CALLS
# python2 debugging.py --obsdata testdata/2019020409/obs_2t.grib2 --modeldata testdata/2019020409/fcst_2t.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi2_2t.grib2 --parameter Temperature
# python2 debugging.py --obsdata testdata/2019020409/obs_2r.grib2 --modeldata testdata/2019020409/fcst_2r.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi2_2r.grib2 --parameter Dewpoint
# python2 debugging.py --obsdata testdata/2019020409/obs_msl.grib2 --modeldata testdata/2019020409/fcst_msl.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi2_msl.grib2 --parameter msl
# python2 debugging.py --obsdata testdata/2019020409/obs_10si.grib2 --modeldata testdata/2019020409/fcst_10si.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi2_10si.grib2 --parameter wind10

# python2 debugging.py --obsdata testdata/2019032708/obs_tprate.grib2 --modeldata testdata/2019032708/fcst_tprate.grib2 --bgdata testdata/2019032708/mnwc_tprate.grib2 --interpolated_data testdata/2019032708/output/interpolated_tprate_uusi.grib2 --parameter precipitation_bg_1h

# python2 debugging.py --obsdata testdata/2019042611/obs_tprate.grib2 --modeldata testdata/2019042611/fcst_tprate.grib2 --bgdata testdata/2019042611/mnwc_tprate.grib2 --interpolated_data testdata/2019042611/output/interpolated_tprate_uusi.grib2 --parameter precipitation_bg_1h
# python2 debugging.py --obsdata testdata/TCC/mnwc.grib2 --modeldata testdata/TCC/smartmet.grib2 --interpolated_data testdata/TCC/output/smoothed_mnwc_edited.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed

### NEW CALLS
# python2 debugging.py --obs_data testdata/2019020409/obs_2t.grib2 --model_data testdata/2019020409/fcst_2t.grib2 --output_data testdata/2019020409/output/interpolated_2t.grib2 --parameter Temperature --mode analysis_fcst_smoothed --predictability 4
# python2 debugging.py --obs_data testdata/2019020409/obs_2r.grib2 --model_data testdata/2019020409/fcst_2r.grib2 --output_data testdata/2019020409/output/interpolated_2r.grib2 --parameter Dewpoint --mode analysis_fcst_smoothed --predictability 4
# python2 debugging.py --obs_data testdata/2019020409/obs_msl.grib2 --model_data testdata/2019020409/fcst_msl.grib2 --output_data testdata/2019020409/output/interpolated_msl.grib2 --parameter msl --mode analysis_fcst_smoothed --predictability 4
# python2 debugging.py --obs_data testdata/2019020409/obs_10si.grib2 --model_data testdata/2019020409/fcst_10si.grib2 --output_data testdata/2019020409/output/interpolated_10si.grib2 --parameter win10 --mode analysis_fcst_smoothed --predictability 4
# python2 debugging.py --model_data testdata/TCC/smartmet.grib2 --dynamic_nwc_data testdata/TCC/mnwc.grib2 --output_data testdata/TCC/output/smoothed_mnwc_TCC.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 9
# python2 debugging.py --obs_data testdata/2020052509/obs_tprate.grib2 --model_data testdata/2020052509/fcst_tprate.grib2 --background_data testdata/2020052509/mnwc_tprate.grib2 --dynamic_nwc_data testdata/2020052509/mnwc_tprate_full.grib2 --extrapolated_data testdata/2020052509/ppn_tprate.grib2 --detectability_data testdata/radar_detectability_field_255_280.h5 --output_data testdata/2020052509/output/smoothed_mnwc_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 5
