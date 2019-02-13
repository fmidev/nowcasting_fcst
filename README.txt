# Observation-forecast smoother, using OpenCV-library
# jussi.ylhaisi@fmi.fi
# tuuli.perttula@fmi.fi

# This script interpolates data between two fields (analysis and forecast) separated by $PREDICTABILITY.

# Input is [obsdata] (like latest LAPS analysis: one timestep, forecast hour 0) and [modeldata] the model field that is to be used in interpolating (like official "edited" forecast: x timesteps, up until forecast hour x-1). The first timestamp of the modeldata corresponds to the timestamp of the analysis field. -> e.g. The example fcst dataset in testdata/testdata_nwc_230700/ has a time length of 6 but forecasts ranging only up until forecast hour 5.

# Output is a image-morphed interpolated forecast field (x timesteps, up until forecast hour x-1), having the same forecast length as modeldata. The beginning part of the edited forecast up until $PREDICTABILITY-1 hours is replaced with the analysis-blended forecast. If forecast steps in the modeldata and $SECONDS_BETWEEN_STEPS are defined in even hours, the interpolated data should have the same time length as modeldata.

# $PREDICTABILITY is taken as a constant value but in practice should be given as a forecast/flow-dependent parameter to this program, calculated in a separate module (or inside this program).

# EXAMPLE RUNS
# python call_interpolation.py --obsdata testdata/obsdata_nomissing.grib2 --modeldata testdata/modeldata_nomissing.grib2 --interpolated_data outdata/interp.grib2
wrote file 'outdata/interp.grib2'
# python call_interpolation.py --obsdata testdata/testdata_nwc_2019020406UTC/obs_2t.grib2 --modeldata testdata/testdata_nwc_2019020406UTC/fcst_2t.grib2 --interpolated_data testdata/testdata_nwc_2019020406UTC/output/interpolated_uusi_2t.grib2



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

# KNOWN ISSUES AND DEVELOPMENT POSSIBILITIES
# Function read_nc has no kind of error checking of the data atm. The min/max values are taken from the raw data fields as provided. A named list for each parameter? Should here be some error checking based on plausible min/max values? missingvalue -checking already exist in the function read_nc.
# nodata fields can (in principle) be different between the timesteps.
# predictability is a constant value atm whereas in reality it has both flow- and variable-dependence.
# Value gaussian_filter_sigma is also an ad-hoc constant value atm, whereas it should depend on the spatial variability difference of the two fields
# For reading in 1h radar accumulations as an input data, the HDF5 reader function needs to be re-imported from the nowcasting-repo.

# PARAMETERS used are Pressure Temperature DewPoint Humidity WindSpeedMS (currently available in LAPS-analysis)
