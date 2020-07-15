# Smartmet nowcast observation-nowcast-forecast smoother
This smoother is used to create a deterministic nowcast that incorporates several data sources available in the nowcasting regime. The output of the smoother is a smoothly behaving nowcast/forecast, the beginning part of which up until $PREDICTABILITY-1 hours is created by using image-morphing algorithms and the related interpolation. The output data from the $PREDICTABILITY hour onwards is completely model-based. If forecast steps in the modeldata and $SECONDS_BETWEEN_STEPS are defined in even hours, the output data will have the same time length as modeldata. The output nowcast can then be pasted back to the early part of the longer NWP forecast (of e.g. 10 days). The end result is a spatially and temporally seamless medium-range forecast, that preserves smooth development of the physical fields without discontinuities and can be rapidly updated with the latest nowcast data.

## Authors
jussi.ylhaisi@fmi.fi
tuuli.perttula@fmi.fi

## Usage
Code is written in Python 2.7 and heavily uses eccodes Python bindings. The program can be run in a separate container. The needed library dependencies (but not all!) are listed in the file requirements.txt.

### Example runs
```console
python2 call_interpolation.py --model_data testdata/latest/fcst_tprate.grib2 --background_data testdata/latest/mnwc_tprate.grib2 --dynamic_nwc_data testdata/latest/mnwc_tprate_full.grib2 --extrapolated_data testdata/latest/ppn_tprate.grib2 --detectability_data testdata/radar_detectability_field_255_280.h5 --output_data testdata/latest/output/smoothed_mnwc_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 9
python2 call_interpolation.py --model_data testdata/latest/fcst_cc.grib2 --dynamic_nwc_data testdata/latest/mnwc_cc.grib2 --output_data testdata/latest/output/smoothed_mnwc_TCC.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 9
python2 call_interpolation.py --obs_data testdata/latest/obs_2t.grib2 --model_data testdata/latest/fcst_2t.grib2 --output_data testdata/latest/output/interpolated_2t.grib2 --parameter Temperature --mode analysis_fcst_smoothed --predictability 4
python2 call_interpolation.py --obs_data testdata/latest/obs_2r.grib2 --model_data testdata/latest/fcst_2r.grib2 --output_data testdata/latest/output/interpolated_2r.grib2 --parameter Dewpoint --mode analysis_fcst_smoothed --predictability 4
python2 call_interpolation.py --obs_data testdata/latest/obs_msl.grib2 --model_data testdata/latest/fcst_msl.grib2 --output_data testdata/latest/output/interpolated_msl.grib2 --parameter msl --mode analysis_fcst_smoothed --predictability 4
python2 call_interpolation.py --obs_data testdata/latest/obs_10si.grib2 --model_data testdata/latest/fcst_10si.grib2 --output_data testdata/latest/output/interpolated_10si.grib2 --parameter wind10m --mode analysis_fcst_smoothed --predictability 4
```

### Input parameters
Parameter|Explanation|Obligatory|Default value
----|----|----|----
`obs_data`|Observation data field or similar, used as 0h forecast|no|`none`
`model_data`|Model data field, towards which the nowcast is smoothed|yes|`none`
`background_data`|Background data field for the 0h forecast where obsdata is spatially merged to|no|`none`
`dynamic_nwc_data`|Dynamic nowcasting model data field, which is smoothed to modeldata. If extrapolated_data is provided, it is spatially smoothed with dynamic_nwc_data. First timestep of 0h should not be included in this data!|no|`none`
`extrapolated_data`|Nowcasting model data field acquired using extrapolation methods (like PPN), which is smoothed to modeldata. If dynamic_nwc_data is provided, extrapolated_data is spatially smoothed with it. First timestep of 0h should not be included in this data!|no|`none`
`detectability_data`|Radar detectability field, which is used in spatial blending of obsdata and bgdata|no|`testdata/radar_detectability_field_255_280.h5`
`output_data`|Output file name for nowcast data field|yes|`none`
`seconds_between_steps`|Timestep of output data in seconds|yes|3600
`predictability`|Predictability in hours, at which forecast is completely based on model_data. Smoothed nowcast fields are generated between 0h and predictability hours|yes|4
`parameter`|Parameter name|yes|`none`
`mode`|Either "analysis_fcst_smoothed" or "model_fcst_smoothed" mode. In "analysis_fcst_smoothed" mode, nowcasts are interpolated between 0h (obs_data/background_data) and predictability hours (model_data). In "model_fcst_smoothed" mode, nowcasts are individually interpolated for each forecast length between dynamic_nwc_data/extrapolated/data and model_data and their corresponding forecasts|yes|`analysis_fcst_smoothed`
`gaussian_filter_sigma`|This parameter sets the blurring intensity of the of the analysis field|no|0.5
`R_min`|Minimum precipitation intensity for optical flow computations. Values below R_min are set to zero|no|0.1
`R_max`|Maximum precipitation intensity for optical flow computations. Values above R_max are clamped|no|30
`DBZH_min`|Minimum DBZH for optical flow computations. Values below DBZH_min are set to zero|no|10
`DBZH_max`|Maximum DBZH for optical flow computations. Values above DBZH_max are clamped|no|45
`farneback_params`|Location of farneback params configuration file|yes|`compute_advinterp.cfg`
`plot_diagnostics`|If this option is set to yes, program plots out several diagnostics to files|yes|`no`

### Further instructions
* Minimum input varies depending on the input parameters. At least some data (either obsdata, dynamic_nwc_data or extrapolated_data) in the beginning of the nowcast needs to be given. If obsdata/background_data is not given, the first timestep will simply be the model_data field.
* This smoother only works for complete fields. No constant values etc. can be given.
* Parameter added_hours for the function read() can be used to easily increase the timestamps of the input data, if necessary. This feature was made as precipitation_1h -timestamps are not unambiguously defined in the input data.

## Known issues, features and development ideas
* "verif" mode (calculation of several verification metrics for all available producers, using past data) does not exist at the moment.
* Input parameters {R_min, R_max, DBZH_min, DBZH_max} are only used at the subroutine read_HDF5()
* Function read_nc has no kind of error checking of the data atm. The min/max values are taken from the raw data fields as provided. A named list for each parameter? Should here be some error checking based on plausible min/max values? missingvalue -checking already exist in the function read_nc.
* Input parameter gaussian_filter_sigma is not used at the moment. It is also an ad-hoc constant value, whereas it should depend on the spatial variability difference of the two fields.
* Program is very picky on the spatial representation of the input data. All the input data needs to have the same grid definition with similar gridpoints!
* nodata values are taken from the model field (nodata = nodata2) and this value is ASSUMED to be the same also for all the other fields (nodata1,nodatax1, ...). The mask_nodata -fields of the other data need to have nodata values, but this is not checked at all.
* nodata fields can (in principle) be different between the timesteps, but the subroutine define_common_mask_for_fields() does not analyse them by any means.
* predictability is a constant value atm whereas in reality it has both flow- and variable-dependence.
* Mode "analysis_fcst_smoothed" is based on the "frozen turbulence" assumption, so any dynamic development of the forecast field between the 0h and predictability is not taken into account (e.g. daily max temps are underestimated in those afternoon runs when the modeled Tmax occurs between 0h and predictability hours)
* Use extreme caution if you are using other value for the parameter $seconds_between_Steps than 3600 (so not even-hour forecasts).

## Science issues
* Weightings between extrapolated_data/dynamic_nwc_data/model_data can be quite easily changed by altering the input parameter "sigmoid_steepness" of the subroutine interpolate_fcst.model_smoothing()
* The radar detectability mask used in the spatial combination of dynamic_nwc_data/detectability_data is extended as a function of the forecast length (as implied by the parameter mask_inflation)
* OpenCV-library is used for the motion field calculation and the interpolation of the fields. The parameters used by the Farneback optical flow algorithm are separately stored in the file compute_advinterp.cfg. These parameters are always constant for a specific variable. For the variable "precipitation_1h_bg", the Farneback parameters are different compared to all the other variables.

## Plotting diagnostics
The program contains a diagnostics script diagnostics_debugging.sh (calls the retrieval script retrieval/testdata/latest/retrieval_script.sh), that retrieves the latest input fields of the smoother and plots out various diagnostics plots from several input/output fields.

## Retrieval queries of the detectability fields
### The NEW radar detectability field (file is generated by Rack at dev.elmo and retrieved using url
http://res.elmo.fmi.fi:8080/venison/cache/2019/05/06/radar/rack/comp/201905061045_radar.rack.comp_PROJ=SCAND_BBOX=5.4211,52.6997,45.3471,70.8257_SITES=fi_SIZE=255,280.h5
### OLD retrieval is
http://res.elmo.fmi.fi:8080/venison/cache/2019/05/06/radar/rack/comp/201905061045_radar.rack.comp_PROJ=SCAND_BBOX=6,51.3,49.046,70.205_SITES=fi_SIZE=270,300.h5
### The used FMI Scandinavia projection is defined at
http://res.elmo.fmi.fi:8080/venison/products/radar/share/proj.cnf
and is "FMI Scandinavia, Polar Stereographic".
SCAND="+proj=stere +a=6371288 +lon_0=20E +lat_0=90N +lat_ts=60"
This projection *might* be a bit different to
proj4 = +proj=stere +lat_0=90 +lat_ts=60 +lon_0=20 +k=1 +x_0=0 +y_0=0 +a=6371220 +b=6371220 +units=m +no_defs
used by the model data, but the difference between all the corresponding datapoints is less than 0.01 degrees for both latitude and longitude! -> As the difference is negligible, use detectability field provided by Rack in SCAND projection as such.