#!/bin/bash
# This is a simple debugging script that takes timstamp as an input and runs the program through using the corresponding testdata
# The scripts retrieves input data from S3 to nowcasting_fcst/testdata

# PARAMETER INPUTS: $1 is the timedate (ie. 202401250500 25.1.2024 klo 05:00utc), $2 is parameter, $4 is mode either snwc or hrnwc ($4 is yes/no (plot_diagnostics))

timedate=$1
parameter=$2 # 2r, 2t, tprate, rprate, cc, 10si, gust
mode=$3 # snwc or hrnwc (pot,total_cloud_cover,tprate,rainrate_15min_bg,rprate)
PLOT_DIAGNOSTICS="no"
snwctime=${timedate:0:10}
echo $snwctime
wrkdir=/home/users/hietal/statcal/python_projects/nowcasting_fcst
#mkdir -p figures/fields
#mkdir -p figures/jumpiness_absdiff
#mkdir -p figures/jumpiness_meandiff
#mkdir -p figures/jumpiness_ratio
#mkdir -p figures/linear_change
#mkdir -p figures/linear_change3h
#mkdir -p figures/linear_change4h

### RETRIEVE LATEST DATA
cd testdata/
if [ "$parameter" == "2t" ]; then
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/fcst_2t.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/mnwc_2t.grib2
  cd output/ 
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/interpolated_2t.grib2
  cd ../..
  python3 call_interpolation.py --model_data testdata/fcst_2t.grib2 --dynamic_nwc_data testdata/mnwc_2t.grib2 --output_data testdata/output/new1_2t.grib2 --parameter 2t --mode model_fcst_smoothed --predictability 11 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_2t.grib2 testdata/output/new1_2t.grib2
elif [ "$parameter" == "2r" ]; then
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/fcst_2r.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/mnwc_2r.grib2
  cd output/
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/interpolated_2r.grib2
  cd ../..
  python3 call_interpolation.py --model_data testdata/fcst_2r.grib2 --dynamic_nwc_data testdata/mnwc_2r.grib2 --output_data testdata/output/new1_2r.grib2 --parameter 2r --mode model_fcst_smoothed --predictability 11 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_2r.grib2 testdata/output/new1_2r.grib2
elif [ "$parameter" == "10si" ]; then
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/fcst_10si.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/mnwc_10si.grib2
  cd output/
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/interpolated_10si.grib2
  cd ../..
  python3 call_interpolation.py --model_data testdata/fcst_10si.grib2 --dynamic_nwc_data testdata/mnwc_10si.grib2 --output_data testdata/output/new1_10si.grib2 --parameter 10si --mode model_fcst_smoothed --predictability 11 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_10si.grib2 testdata/output/new1_10si.grib2
elif [ "$parameter" == "gust" ]; then
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/fcst_gust.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/mnwc_gust.grib2
  cd output/
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/interpolated_gust.grib2
  cd ../..
  python3 call_interpolation.py --model_data testdata/fcst_gust.grib2 --dynamic_nwc_data testdata/mnwc_gust.grib2 --output_data testdata/output/new1_gust.grib2 --parameter gust --mode model_fcst_smoothed --predictability 11 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_gust.grib2 testdata/output/new1_gust.grib2
# TPRATE
elif [ "$parameter" == "tprate" ]; then
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/fcst_tprate.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/mnwc_tprate.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/mnwc_tprate_full.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/ppn_tprate.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/ppn_tprate_obs.grib2
  cd output/
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/interpolated_tprate.grib2
  cd ../..
  python3 call_interpolation.py --obs_data testdata/ppn_tprate_obs.grib2 --model_data testdata/fcst_tprate.grib2 --background_data testdata/mnwc_tprate.grib2 --dynamic_nwc_data testdata/mnwc_tprate_full.grib2 --extrapolated_data testdata/ppn_tprate.grib2 --output_data testdata/output/new1_tprate.grib2 --parameter tprate --mode model_fcst_smoothed --predictability 11 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_tprate.grib2 testdata/output/new1_tprate.grib2
elif [ "$parameter" == "pot" ] && [ "$mode" == "hrnwc" ]; then
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/thundercast_tstm.grib2
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/mnwc_tstm.grib2
  cd output/
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/interpolated_tstm.grib2
  cd ../..
  python3 call_interpolation.py --dynamic_nwc_data testdata/mnwc_tstm.grib2 --extrapolated_data testdata/thundercast_tstm.grib2 --output_data testdata/output/hrnwc1_pot.grib2 --parameter pot --mode model_fcst_smoothed --predictability 7 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_tstm.grib2 testdata/output/hrnwc1_pot.grib2
elif [ "$parameter" == "pot" ] && [ "$mode" == "snwc" ]; then
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/fcst_tstm.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/thundercast_tstm.grib2 # hrnwc_tstm.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/mnwc_tstm.grib2
  cd output/
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/interpolated_tstm.grib2
  cd ../..
  python3 call_interpolation.py --model_data testdata/fcst_tstm.grib2 --dynamic_nwc_data testdata/mnwc_tstm.grib2 --output_data testdata/output/new1_tstm.grib2 --parameter pot --mode model_fcst_smoothed --predictability 10 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_tstm.grib2 testdata/output/new1_tstm.grib2
elif [ "$parameter" == "cc" ] && [ "$mode" == "hrnwc" ]; then
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/cloudcast_cc.grib2
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/mnwc_cc.grib2
  cd output/
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/interpolated_cc.grib2
  cd ../..
  python3 call_interpolation.py --dynamic_nwc_data testdata/mnwc_cc.grib2 --extrapolated_data testdata/cloudcast_cc.grib2 --output_data testdata/output/hrnwc1_cc.grib2 --parameter cc --mode model_fcst_smoothed --predictability 7 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_cc.grib2 testdata/output/hrnwc1_cc.grib2
elif [ "$parameter" == "cc" ] && [ "$mode" == "snwc" ]; then
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/fcst_cc.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/cloudcast_cc.grib2
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/mnwc_cc.grib2
  cd output/
  s3cmd get s3://routines-data/smartmet-nwc/production/2.5/"$snwctime"/interpolated_cc.grib2
  cd ../..
  python3 call_interpolation.py --model_data testdata/fcst_cc.grib2 --dynamic_nwc_data testdata/mnwc_cc.grib2 --extrapolated_data testdata/cloudcast_cc.grib2 --output_data testdata/output/new1_cc.grib2 --parameter cc --mode model_fcst_smoothed --predictability 11 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_cc.grib2 testdata/output/new1_cc.grib2
elif [ "$parameter" == "rprate" ] && [ "$mode" == "hrnwc" ]; then
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/mnwc_rprate_obs.grib2
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/mnwc_rprate.grib2
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/ppn_rprate.grib2
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/ppn_rprate_obs.grib2
    cd output/
  s3cmd get s3://routines-data/hrnwc/preop/"$timedate"/interpolated_rprate.grib2
  cd ../..
  python3 call_interpolation.py --obs_data testdata/ppn_rprate_obs.grib2 --background_data testdata/mnwc_rprate_obs.grib2 --dynamic_nwc_data testdata/mnwc_rprate.grib2 --extrapolated_data testdata/ppn_rprate.grib2 --output_data testdata/output/hrnwc1_rprate.grib2 --parameter rainrate_15min_bg --mode model_fcst_smoothed --predictability 7 --plot_diagnostics $PLOT_DIAGNOSTICS
  grib_compare testdata/output/interpolated_rprate.grib2 testdata/output/hrnwc1_rprate.grib2
else
  echo "parameter not found"
  exit 1
fi

cd testdata/
rm *.grib2
#find . -name '*.grib2' -delete
