# Observation-forecast smoother, using OpenCV-library
# jussi.ylhaisi@fmi.fi
# tuuli.perttula@fmi.fi

# This script interpolates data between two fields (analysis and forecast) separated by $PREDICTABILITY.

# Input is [obsdata] (like latest LAPS analysis: one timestep, forecast hour 0) and [modeldata] the model field that is to be used in interpolating (like official "edited" forecast: x timesteps, up until forecast hour x-1). The first timestamp of the modeldata corresponds to the timestamp of the analysis field. -> e.g. The example fcst dataset in testdata/testdata_nwc_230700/ has a time length of 6 but forecasts ranging only up until forecast hour 5.

# Output is a image-morphed interpolated forecast field (x timesteps, up until forecast hour x-1), having the same forecast length as modeldata. The beginning part of the edited forecast up until $PREDICTABILITY-1 hours is replaced with the analysis-blended forecast. If forecast steps in the modeldata and $SECONDS_BETWEEN_STEPS are defined in even hours, the interpolated data should have the same time length as modeldata.

# $PREDICTABILITY is taken as a constant value but in practice should be given as a forecast/flow-dependent parameter to this program, calculated in a separate module (or inside this program).

# EXAMPLE RUNS
# python2 call_interpolation.py --obsdata testdata/obsdata_nomissing.grib2 --modeldata testdata/modeldata_nomissing.grib2 --interpolated_data outdata/interp.grib2
wrote file 'outdata/interp.grib2'
# python2 call_interpolation.py --obsdata testdata/2019020409/obs_2t.grib2 --modeldata testdata/2019020409/fcst_2t.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi_2t.grib2 --parameter Temperature
# python2 call_interpolation.py --obsdata testdata/2019020409/obs_msl.grib2 --modeldata testdata/2019020409/fcst_msl.grib2 --interpolated_data testdata/2019020409/output/interpolated_uusi_msl.grib2 --parameter msl
# python2 call_interpolation.py --obsdata testdata/2019032708/obs_tprate.grib2 --modeldata testdata/2019032708/fcst_tprate.grib2 --bgdata testdata/2019032708/mnwc_tprate.grib2 --interpolated_data testdata/2019032708/output/interpolated_tprate.grib2 --parameter precipitation_bg_1h
# EDITING AREA WAS CHANGED AT THE END OF MAY 2019, CAUSING PRECIPITATION CODE TO CRASH. NEW CALLS USING TESTDATA FROM THE NEW EDITING AREA ARE BELOW
# python2 call_interpolation.py --obsdata testdata/2019052809/obs_2t.grib2 --modeldata testdata/2019052809/fcst_2t.grib2 --interpolated_data testdata/2019052809/output/interpolated_uusi_2t.grib2 --parameter Temperature
# python2 call_interpolation.py --obsdata testdata/2019052809/obs_msl.grib2 --modeldata testdata/2019052809/fcst_msl.grib2 --interpolated_data testdata/2019052809/output/interpolated_uusi_msl.grib2 --parameter msl
# python2 call_interpolation.py --obsdata testdata/2019052809/obs_tprate.grib2 --modeldata testdata/2019052809/fcst_tprate.grib2 --bgdata testdata/2019052809/mnwc_tprate.grib2 --interpolated_data testdata/2019052809/output/interpolated_tprate.grib2 --parameter precipitation_bg_1h
# python2 call_interpolation.py --obsdata testdata/TCC/mnwc.grib2 --modeldata testdata/TCC/smartmet.grib2 --interpolated_data testdata/TCC/output/smoothed_mnwc_edited.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed

# AFTER MAJOR CODE REVISION (June2020) EXAMPLE RUNS ARE AS
# python2 call_interpolation.py --obs_data testdata/2020052509/obs_tprate.grib2 --model_data testdata/2020052509/fcst_tprate.grib2 --background_data testdata/2020052509/mnwc_tprate.grib2 --dynamic_nwc_data testdata/2020052509/mnwc_tprate_full.grib2 --extrapolated_data testdata/2020052509/ppn_tprate.grib2 --detectability_data testdata/radar_detectability_field_255_280.h5 --output_data testdata/2020052509/output/smoothed_mnwc_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 9
# python2 call_interpolation.py --model_data testdata/TCC/smartmet.grib2 --dynamic_nwc_data testdata/TCC/mnwc.grib2 --output_data testdata/TCC/output/smoothed_mnwc_TCC.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 9
# python2 call_interpolation.py --obs_data testdata/2019020409/obs_2t.grib2 --model_data testdata/2019020409/fcst_2t.grib2 --output_data testdata/2019020409/output/interpolated_2t.grib2 --parameter Temperature --mode analysis_fcst_smoothed --predictability 4
# python2 call_interpolation.py --obs_data testdata/2019020409/obs_2r.grib2 --model_data testdata/2019020409/fcst_2r.grib2 --output_data testdata/2019020409/output/interpolated_2r.grib2 --parameter Dewpoint --mode analysis_fcst_smoothed --predictability 4
# python2 call_interpolation.py --obs_data testdata/2019020409/obs_msl.grib2 --model_data testdata/2019020409/fcst_msl.grib2 --output_data testdata/2019020409/output/interpolated_msl.grib2 --parameter msl --mode analysis_fcst_smoothed --predictability 4
# python2 call_interpolation.py --obs_data testdata/2019020409/obs_10si.grib2 --model_data testdata/2019020409/fcst_10si.grib2 --output_data testdata/2019020409/output/interpolated_10si.grib2 --parameter win10 --mode analysis_fcst_smoothed --predictability 4



# The testdata files were retrieved from Smartmet Server using commands

# wget -O testdata/obsdata_3.nc --no-proxy 'http://smartmet.fmi.fi/download?param=Pressure&producer=laps_skandinavia&format=netcdf&starttime=201810180200&endtime=201810180200&timestep=60&projection=stereographic&centrallongitude=20,centrallatitude=90,truelatitude=60&bbox=5,55,40,71&gridsize=222,236'
# nccopy -k 4 testdata/obsdata_3.nc testdata/obsdata.nc
# rm testdata/obsdata_3.nc
# AND
# wget -O testdata/modeldata_3.nc --no-proxy 'http://smartmet.fmi.fi/download?param=Pressure&producer=pal_skandinavia&format=netcdf&starttime=201810180200&timestep=60&projection=stereographic&centrallongitude=20,centrallatitude=90,truelatitude=60&bbox=5,55,40,71&gridsize=222,236'
# nccopy -k 4 testdata/modeldata_3.nc testdata/modeldata.nc
# rm testdata/modeldata_3.nc

# => laps_skandinavia -grid needs to have a similar spatial resolution+projection than edited data (pal_skandinavia) => the max available edited resolution. The grid definition needs to be retrieved from the modeldata.nc file
# In this example, a $PREDICTABILITY of 4 hours is used.

# The NEW radar detectability field (file is generated by Rack at dev.elmo and retrieved using url
# http://res.elmo.fmi.fi:8080/venison/cache/2019/05/06/radar/rack/comp/201905061045_radar.rack.comp_PROJ=SCAND_BBOX=5.4211,52.6997,45.3471,70.8257_SITES=fi_SIZE=255,280.h5
# OLD retrieval is
# http://res.elmo.fmi.fi:8080/venison/cache/2019/05/06/radar/rack/comp/201905061045_radar.rack.comp_PROJ=SCAND_BBOX=6,51.3,49.046,70.205_SITES=fi_SIZE=270,300.h5
# The used FMI Scandinavia projection is defined here http://res.elmo.fmi.fi:8080/venison/products/radar/share/proj.cnf
# FMI Scandinavia, Polar Stereographic
# SCAND="+proj=stere +a=6371288 +lon_0=20E +lat_0=90N +lat_ts=60"
# This *might* be a bit different to
# proj4    = +proj=stere +lat_0=90 +lat_ts=60 +lon_0=20 +k=1 +x_0=0 +y_0=0 +a=6371220 +b=6371220 +units=m +no_defs, but in all cases less than 0.01 degrees for both latitude and longitude! -> negligible, use as provided by Rack in SCAND projection




# KNOWN ISSUES AND DEVELOPMENT POSSIBILITIES
# Function read_nc has no kind of error checking of the data atm. The min/max values are taken from the raw data fields as provided. A named list for each parameter? Should here be some error checking based on plausible min/max values? missingvalue -checking already exist in the function read_nc.
# nodata fields can (in principle) be different between the timesteps.
# predictability is a constant value atm whereas in reality it has both flow- and variable-dependence.
# Value gaussian_filter_sigma is also an ad-hoc constant value atm, whereas it should depend on the spatial variability difference of the two fields
# For reading in 1h radar accumulations as an input data, the HDF5 reader function needs to be re-imported from the nowcasting-repo.


# Known issues:
# nodata fields are taken from the model field (nodata = nodata2), but this value is ASSUMED to be the same also for all the other fields. nodata1,nodatax1,... would need to be replaced in all the other fields with nodata values.
