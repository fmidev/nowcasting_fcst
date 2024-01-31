# -*- coding: utf-8 -*-
import interpolate_fcst
from fileutils import read, write, read_input, read_background_data_and_make_mask
from datautils import farneback_params_config, define_common_mask_for_fields
import numpy as np
import argparse
import datetime
import configparser
import sys
import time
import cv2
from scipy.ndimage import gaussian_filter, distance_transform_edt

# from scipy.misc import imresize -> from PIL imort Image (imresize FEATURE IS NOT SUPPORTED ATM)


def main():

    # Parameter name needs to be given as an argument!
    if options.parameter == None:
        raise NameError("Parameter name needs to be given as an argument!")

    # give default values if no model_data
    nodata = None
    timestamp2 = None

    # Read parameters from config file for interpolation or optical flow algorithm.
    farneback_params = farneback_params_config(options.farneback_params)

    # For accumulated 1h precipitation, larger surrounding area for Farneback params are used: winsize 30 -> 150 and poly_n 20 -> 61
    if options.parameter == "precipitation_1h_bg":
        farneback_params = list(farneback_params)
        farneback_params[2] = 150
        farneback_params[4] = 61
        farneback_params = tuple(farneback_params)
    # Like mentioned at https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html, a reasonable value for poly_sigma depends on poly_n.
    # Here we use a fraction 4.6 for these values.
    fb_params = (
        farneback_params[0],
        farneback_params[1],
        farneback_params[2],
        farneback_params[3],
        farneback_params[4],
        (farneback_params[4] / 4.6),
        farneback_params[6],
    )


    """ Reading input parameters """
    # Model data = the final NWP data for which the nowcast data is blended, i.e the 10d forecast. This is not obligatory for all use cases!
    # If parameter is precipitation 1h/instant, total cloud cover, probability of thunder (and there is dynamic_nwc_data) no need of model_data.
    if options.model_data != None: 
        (image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2, nodata2, longitudes2, latitudes2, quantity2) = read_input(options, options.model_data, use_as_template=(options.dynamic_nwc_data is None))
        nodata = nodata2

    # Read in observation data for precipitation (Time stamp is analysis time!)
    if options.obs_data != None:
        (image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1, nodata1, longitudes1, latitudes1, quantity1) = read_input(options, options.obs_data, use_as_template=False)

    # If observation data is supplemented with background data (to which Finnish radar data is spatially smoothed), create a spatial mask and combine these two fields
    if options.background_data != None:
        (image_array3, quantity3_min, quantity3_max, timestamp3, mask_nodata3, nodata3, longitudes3, latitudes3, quantity3) = read_input(options, options.background_data,use_as_template=False)

    # Loading in dynamic_nwc_data (MNWC). This can (or not) include the analysis step!
    if options.dynamic_nwc_data != None:
        (image_arrayx1, quantityx1_min, quantityx1_max, timestampx1, mask_nodatax1, nodatax1, longitudesx1, latitudesx1, quantityx1) = read_input(options, options.dynamic_nwc_data,use_as_template=True)

    # Loading in extrapolated_data (observations based nowcast: PPN, Cloudcast, Thundercast).
    if options.extrapolated_data != None:
        (image_arrayx2, quantityx2_min, quantityx2_max, timestampx2, mask_nodatax2, nodatax2, longitudesx2, latitudesx2, quantityx2) = read_input(options, options.extrapolated_data, use_as_template=False)

    """ Creating spatial composite for the first time step from radar obs and NWP model background data for precipitation """
    if "image_array1" in locals() and "image_array3" in locals():
        # Read radar composite field used as mask for Finnish data. Lat/lon info is not stored in radar composite HDF5 files, but these are the same! (see README.txt)
        # We are assuming that the analysis time includes the radar mask information and not using separate detectability_data
        weights_bg = read_background_data_and_make_mask(
            image_file=None,
            input_mask=mask_nodata1.mask,
            mask_smaller_than_borders=4,
            smoother_coefficient=0.2,
            gaussian_filter_coefficient=3,
        )
        weights_obs = 1 - weights_bg
        if weights_bg.shape != image_array1.shape[1:] != image_array3.shape[1:]:
            raise ValueError(
                "Model data, background data and image do not all have same grid size!"
            )
        # Adding up the two fields (obs_data for area over Finland, bg field for area outside Finland)
        image_array1[0, :, :] = (
            weights_obs * image_array1[0, :, :] + weights_bg * image_array3[0, :, :]
        )
        mask_nodata1 = mask_nodata3
    # If background data is not available, but obs data is, DO NOTHING
    # If only background data is available, use that as image_array and timestamp
    if "image_array3" in locals() and "image_array1" not in locals():
        image_array1 = image_array3
        mask_nodata1 = mask_nodata3
        timestamp1 = timestamp3

    """ Creating spatial composite for the forecast timesteps from radar nowcast (1h, instant rate) and NWP data. First value is the first leadtime (1h or 15min) since 0h is created separately """
    if ("image_arrayx1" in locals() and "image_arrayx2" in locals()
        and options.parameter in ["precipitation_1h_bg", "rainrate_15min_bg"]
    ):
        weights_obs_extrap = np.zeros(image_arrayx2.shape)
        if type(timestampx1) == list and type(timestampx2) == list:
            # Finding out time steps in extrapolated_data that are also found in dynamic_nwc_data
            dynamic_nwc_data_common_indices = [
                timestampx1.index(x) if x in timestampx1 else None for x in timestampx2
            ]
        if len(dynamic_nwc_data_common_indices) > 0:
            # Spatially combine dynamic_nwc forecast to image_arrayx3 (initialise the array here)
            image_arrayx3 = np.copy(image_arrayx1)
            # Combine data for each forecast length
            for common_index in range(0, len(dynamic_nwc_data_common_indices)):
                # # A larger mask could be used here in case of constant mask and longer forecast length (PPN advects precipitation data also outside Finnish borders -> increase mask by 8 pixels/hour)
                # mask_inflation = common_index*6
                mask_inflation = -6
                # Read radar composite field used as mask for Finnish data. Lat/lon info is not stored in radar composite HDF5 files, but these are the same! (see README.txt)
                weights_bg = read_background_data_and_make_mask(
                    image_file=None,
                    input_mask=mask_nodatax2.mask[common_index, :, :],
                    mask_smaller_than_borders=(-2 - mask_inflation),
                    smoother_coefficient=0.2,
                    gaussian_filter_coefficient=3,
                )
                weights_obs = 1 - weights_bg
                weights_obs_extrap[common_index, :, :] = weights_obs
                # plt.imshow(weights_obs)
                # plt.show()
                if (
                    weights_bg.shape
                    != image_arrayx1.shape[1:]
                    != image_arrayx2.shape[1:]
                ):
                    raise ValueError(
                        "Model data, background data and image do not all have same grid size!"
                    )
                # Adding up the two fields (obs_data for area over Finland, bg field for area outside Finland)
                if dynamic_nwc_data_common_indices[common_index] != None:
                    image_arrayx3[
                        dynamic_nwc_data_common_indices[common_index], :, :
                    ] = (
                        weights_obs * image_arrayx2[common_index, :, :]
                        + weights_bg
                        * image_arrayx1[
                            dynamic_nwc_data_common_indices[common_index], :, :
                        ]
                    )
            # After nwc_dynamic_data and extrapolated_data have been spatially combined for all the forecast steps, then calculate interpolated values between the fields image_arrayx1 and image_arrayx3
            # As there always is more than one common timestamp between these data, always combine them using model_smoothing -method!
            if (
                options.mode == "model_fcst_smoothed"
                or options.mode == "analysis_fcst_smoothed"
            ):
                # Defining mins and max in all data
                R_min_nwc = min(image_arrayx1.min(), image_arrayx3.min())
                R_max_nwc = max(image_arrayx1.max(), image_arrayx3.max())
                # Code only supports precipitation extrapolation data (PPN). Using other variables will cause an error. predictability/R_min/sigmoid_steepness are variable-dependent values! Here predictability is len(timestampx2) and not len(timestampx2)+1! -> last timestep of image_arrayx2 recieves a weight 0!
                if options.parameter in ["precipitation_1h_bg", "rainrate_15min_bg"]:
                    if nodata == None:
                        nodata = nodatax1
                    image_arrayx1 = interpolate_fcst.model_smoothing(
                        obsfields=image_arrayx3,
                        modelfields=image_arrayx1,
                        mask_nodata=define_common_mask_for_fields(mask_nodatax1),
                        farneback_params=fb_params,
                        predictability=len(timestampx2),
                        seconds_between_steps=options.seconds_between_steps,
                        R_min=R_min_nwc,
                        R_max=R_max_nwc,
                        missingval=nodata,
                        logtrans=False,
                        sigmoid_steepness=-4.5,
                    )
                else:
                    raise ValueError(
                        "Only precipitation_1h_bg variable is supported by the code! Provide variable-dependent value for sigmoid_steepness! Revise also the bg mask used!"
                    )
            else:
                raise ValueError(
                    "Mode must be either model_fcst_smoothed or analysis_fcst_smoothed!"
                )
        else:
            raise ValueError(
                "Check your data! Only one common forecast step between dynamic_nwc_data and extrapolated_data and there's no sense in combining these two forecast sources spatially!"
            )
    # If only dynamic_nwc_data is available, use that (so do nothing)
    # If only extrapolated_data is available, use that as nowcasting data
    if "image_arrayx2" in locals() and "image_arrayx1" not in locals():
        image_arrayx1 = image_arrayx2
        mask_nodatax1 = mask_nodatax2
        timestampx1 = timestampx2


    """ Blending dynamic_nwc data with extrapolated_data when the parameter is NOT precipitation or rainrate (currently used in HRNWC production for cloudcast and Thundercast """
    if (
        "image_arrayx1" in locals()
        and "image_arrayx2" in locals()
        and options.parameter != "precipitation_1h_bg"
        and options.parameter != "rainrate_15min_bg"
    ):
        weights_obs_extrap = np.zeros(image_arrayx2.shape)
        if type(timestampx1) == list and type(timestampx2) == list:
            # Finding out time steps in extrapolated_data that are also found in dynamic_nwc_data
            dynamic_nwc_data_common_indices = [
                timestampx1.index(x) if x in timestampx1 else None for x in timestampx2
            ]
            print("Common timestep in mnwc and extapolated data",len(dynamic_nwc_data_common_indices))
        if len(dynamic_nwc_data_common_indices) > 0:
            # Calculate interpolated values between the fields image_arrayx1 and image_arrayx2
            # As there always is more than one common timestamp between these data, always combine them using model_smoothing -method!
            if (
                options.mode == "model_fcst_smoothed"
                or options.mode == "analysis_fcst_smoothed"
            ):
                # Defining mins and max in all data
                R_min_nwc = min(image_arrayx1.min(), image_arrayx2.min())
                R_max_nwc = max(image_arrayx1.max(), image_arrayx2.max())
                # Code only supports precipitation extrapolation data (PPN). Using other variables will cause an error. predictability/R_min/sigmoid_steepness are variable-dependent values! Here predictability is len(timestampx2) and not len(timestampx2)+1! -> last timestep of image_arrayx2 recieves a weight 0!
                if options.parameter != "precipitation_1h_bg":
                    if nodata == None:
                        nodata = nodatax1
                    image_arrayx1 = interpolate_fcst.model_smoothing(
                        obsfields=image_arrayx2,
                        modelfields=image_arrayx1,
                        mask_nodata=define_common_mask_for_fields(mask_nodatax1),
                        farneback_params=fb_params,
                        predictability=len(timestampx2) - 1,
                        seconds_between_steps=options.seconds_between_steps,
                        R_min=R_min_nwc,
                        R_max=R_max_nwc,
                        missingval=nodata,
                        logtrans=False,
                        sigmoid_steepness=-5,
                    )

            # if (options.mode == "model_fcst_smoothed"):
            # interpolated_advection=interpolate_fcst.model_smoothing(obsfields=image_array1, modelfields=image_array2, mask_nodata=mask_nodata, farneback_params=fb_params, predictability=options.predictability, seconds_between_steps=options.seconds_between_steps, R_min=R_min, R_max=R_max, missingval=nodata, logtrans=False, sigmoid_steepness=-5)

            else:
                raise ValueError(
                    "Mode must be either model_fcst_smoothed or analysis_fcst_smoothed!"
                )
    # If only dynamic_nwc_data is available, use that (so do nothing)
    # If only extrapolated_data is available, use that as nowcasting data
    if "image_arrayx2" in locals() and "image_arrayx1" not in locals():
        image_arrayx1 = image_arrayx2
        mask_nodatax1 = mask_nodatax2
        timestampx1 = timestampx2

    # If analysis time step from both/and (obs_data/background_data) is available, set that as the first time step in the combined dataset of extrapolated_data/dynamic_nwc_data
    if "image_array1" in locals():
        # If nwc/extrapolated data is available
        if "image_arrayx1" in locals():
            if image_array1.shape[1:] != image_arrayx1.shape[1:]:
                print(
                    "obsdata and dynamic_nwc/extrapolation data have different grid sizes! Cannot combine!"
                )
            else:
                if timestamp2 == None:
                    timestamp2 = timestampx1
                nwc_model_indices = [
                    timestamp2.index(x) if x in timestamp2 else None
                    for x in timestampx1
                ]
                obs_model_indices = timestamp2.index(timestamp1[0])
                # Use obsdata as such for the first time step and nwc data for the forecast steps after the analysis hour
                # Inflate image_arrayx1 if it has no analysis time step
                if nwc_model_indices[0] == 1 and obs_model_indices == 0:
                    image_array1 = np.append(image_array1, image_arrayx1, axis=0)
                    timestamp1.extend(timestampx1)
                # If image_arrayx1 contains also analysis step, replace first time step in image_arrayx1 with obsdata
                if (
                    nwc_model_indices[0] == 0
                    and obs_model_indices == 0
                    and len(nwc_model_indices) > 1
                ):
                    image_array1 = np.append(
                        image_array1, image_arrayx1[1:, :, :], axis=0
                    )
                    timestamp1.extend(timestampx1[1:])
                mask_nodata1 = define_common_mask_for_fields(
                    mask_nodata1, mask_nodatax1
                )
                # If nwc data is not either analysis time step or 1-hour forecast, throw an error
                if nwc_model_indices[0] > 1:
                    raise ValueError(
                        "Check your nwc input data! It needs more short forecast time steps!"
                    )
    else:
        # If nwc/extrapolated data is available, use that
        if "image_arrayx1" in locals():
            image_array1 = image_arrayx1
            timestamp1 = timestampx1
        else:
            raise ValueError(
                "no obsdata or nwc data available! Cannot smooth anything!"
            )

    """ If model data (10d forecast) is not provided, HRNWC production """
    # For precipitation the spatially combined, smoothed precip field is stored. For other parameter the nwp nowcast and extrapolated data is the final product
    if options.parameter in ["precipitation_1h_bg", "rainrate_15min_bg"]:
        interpolated_advection = image_array1
    elif (
        "image_arrayx1" in locals()
        and options.parameter != "precipitation_1h_bg"
        and options.parameter != "rainrate_15min_bg"
    ):
        interpolated_advection = image_arrayx1

    """ If model_data is provided, SNWC production """
    if options.model_data != None:
        # If needed, fill up "obsfields" data array with model data (so that timestamp2 and timestamp1 will eventually be the same)
        # Find out obsdata indices that coincide with modeldata (None values indicate time steps that there is no obsdata available for those modeldata time steps)
        model_obs_indices = [
            timestamp1.index(x) if x in timestamp1 else None for x in timestamp2
        ]
        # If even a single forecast time index in model_data is missing from the obsfields data, fill it out with model data
        if all(x != None for x in model_obs_indices) == False:
            # Use modeldata as base data
            image_array_temp = np.copy(image_array2)
            timestamp_temp = [i for i in timestamp2]
            # Replace image_array_temp with obsdata for those time stamps that there is obsdata available
            # These obs indices will be assigned (Just remove None values from model_obs_indices)
            assigned_obs_indices = [i for i in model_obs_indices if i != None]
            # Obsdata is assigned to these following indices
            model_assignable_indices = [
                model_obs_indices.index(i) for i in model_obs_indices if i != None
            ]
            image_array_temp[model_assignable_indices, :, :] = image_array1[
                assigned_obs_indices, :, :
            ]
            image_array1 = np.copy(image_array_temp)
            timestamp1 = timestamp_temp
        # Now exists image_array1 (obs/nwc data) and image_array2 (model data)

        # Define nodata masks separately and commonly
        mask_nodata1 = np.ma.masked_where(image_array1 == nodata, image_array1)
        mask_nodata2 = np.ma.masked_where(image_array2 == nodata, image_array2)
        # Replace all values according to mask_nodata
        mask_nodata = define_common_mask_for_fields(mask_nodata1, mask_nodata2)
        image_array1[:, mask_nodata] = nodata
        image_array2[:, mask_nodata] = nodata

        # Checking out that model grid sizes correspond to each other
        if image_array1.shape != image_array2.shape or timestamp1 != timestamp2:
            raise ValueError(
                "image_array1.shape and image_array2.shape do not correspond to each other!"
            )

        # Defining mins and max in all data
        R_min = min(image_array1.min(), image_array2.min())
        R_max = max(image_array1.max(), image_array2.max())

        """ Interpolate data in either analysis_fcst_smoothed or model_fcst_smoothed -mode """
        if options.mode == "analysis_fcst_smoothed":
            interpolated_advection = interpolate_fcst.advection(
                obsfields=image_array1,
                modelfields=image_array2,
                mask_nodata=mask_nodata,
                farneback_params=fb_params,
                predictability=options.predictability,
                seconds_between_steps=options.seconds_between_steps,
                R_min=R_min,
                R_max=R_max,
                missingval=nodata,
                logtrans=False,
            )
        if options.mode == "model_fcst_smoothed":
            interpolated_advection = interpolate_fcst.model_smoothing(
                obsfields=image_array1,
                modelfields=image_array2,
                mask_nodata=mask_nodata,
                farneback_params=fb_params,
                predictability=options.predictability,
                seconds_between_steps=options.seconds_between_steps,
                R_min=R_min,
                R_max=R_max,
                missingval=nodata,
                logtrans=False,
                sigmoid_steepness=-5,
            )

    """ QC for output fields, if thresholds are provided, do not change possible missing data (9999) """
    missing_data = 9999 # missing data in grib
    val_min = None # Threshold for min QC value of the parameter
    val_max = None # Threshold for max QC value of the parameter
    if options.parameter == "precipitation_1h_bg":
        val_max = 22 # maximum hourly precip value
        val_min = 0
    if options.parameter == "total_cloud_cover":
        val_max = 1
        val_min = 0
    if options.parameter == "2r":
        val_max = 100
        val_min = 0
    if options.parameter == "pot":
        val_max = 100
        val_min = 0

    if val_min != None:
        mask_min = np.where((interpolated_advection < val_min) & (interpolated_advection != missing_data))
        interpolated_advection[mask_min] = val_min
    if val_max != None:
        mask_max = np.where((interpolated_advection > val_max) & (interpolated_advection != missing_data))
        interpolated_advection[mask_max] = val_max

    """ Save interpolated field to a new file """
    write(
        interpolated_data=interpolated_advection,
        image_file=None,
        write_file=options.output_data,
        variable=options.parameter,
        predictability=options.predictability,
        t_diff=options.time_offset,
        grib_write_options=options.grib_write_options,
        seconds_between_steps=options.seconds_between_steps,
    )

    """ PLOT OUT DIAGNOSTICS FROM THE DATA """
    if options.plot_diagnostics == "yes":
        from PlotData import PlotData

        if "weights_obs_extrap" in locals():
            PlotData(options, image_array1, image_array2, weights_obs_extrap)
        else:
            PlotData(options, image_array1, image_array2)


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(argument_default=None)
    parser.add_argument(
        "--obs_data", help="Observation data field or similar, used as 0h forecast"
    )
    parser.add_argument(
        "--model_data", help="Model data field, towards which the nowcast is smoothed"
    )
    parser.add_argument(
        "--background_data",
        help="Background data field for the 0h forecast where obsdata is spatially merged to",
    )
    parser.add_argument(
        "--dynamic_nwc_data",
        help="Dynamic nowcasting model data field, which is smoothed to modeldata. If extrapolated_data is provided, it is spatially smoothed with dynamic_nwc_data. First timestep of 0h should not be included in this data!",
    )
    parser.add_argument(
        "--extrapolated_data",
        help="Nowcasting model data field acquired using extrapolation methods (like PPN), which is smoothed to modeldata. If dynamic_nwc_data is provided, extrapolated_data is spatially smoothed with it. First timestep of 0h should not be included in this data!",
    )
    parser.add_argument(
        "--time_offset",
        default=3600,
        type=int,
        help="Adjust analysis time in output grib files by this amount of seconds. Leadtimes are adjusted to match the same wall clock time",
    )
    parser.add_argument(
        "--detectability_data",
        help="Radar detectability field, which is used in spatial blending of obsdata and bgdata",
    )
    parser.add_argument("--output_data", help="Output file name for nowcast data field")
    parser.add_argument(
        "--seconds_between_steps",
        type=int,
        default=3600,
        help="Timestep of output data in seconds",
    )
    parser.add_argument(
        "--predictability",
        type=int,
        default="4",
        help="Predictability in hours. Between the analysis and forecast of this length, forecasts need to be interpolated",
    )
    parser.add_argument("--parameter", help="Variable which is handled.")
    parser.add_argument(
        "--mode",
        default="model_fcst_smoothed",
        help='Either "analysis_fcst_smoothed" or "model_fcst_smoothed" mode. In "analysis_fcst_smoothed" mode, nowcasts are interpolated between 0h (obs_data/background_data) and predictability hours (model_data). In "model_fcst_smoothed" mode, nowcasts are individually interpolated for each forecast length between dynamic_nwc_data/extrapolated/data and model_data and their corresponding forecasts',
    )
    parser.add_argument(
        "--gaussian_filter_sigma",
        type=float,
        default=0.5,
        help="This parameter sets the blurring intensity of the of the analysis field.",
    )
    parser.add_argument(
        "--R_min",
        type=float,
        default=0.1,
        help="Minimum precipitation intensity for optical flow computations. Values below R_min are set to zero.",
    )
    parser.add_argument(
        "--R_max",
        type=float,
        default=30.0,
        help="Maximum precipitation intensity for optical flow computations. Values above R_max are clamped.",
    )
    parser.add_argument(
        "--DBZH_min",
        type=float,
        default=10,
        help="Minimum DBZH for optical flow computations. Values below DBZH_min are set to zero.",
    )
    parser.add_argument(
        "--DBZH_max",
        type=float,
        default=45,
        help="Maximum DBZH for optical flow computations. Values above DBZH_max are clamped.",
    )
    parser.add_argument(
        "--farneback_params",
        default="compute_advinterp.cfg",
        help="location of farneback params configuration file",
    )
    parser.add_argument(
        "--plot_diagnostics",
        default="no",
        help="If this option is set to yes, program plots out several diagnostics to files.",
    )
    parser.add_argument(
        "--grib_write_options",
        type=str,
        default=None,
        help="Grib key-value pairs passed directly to grib writing,comma separated list (k=v,...)",
    )

    options = parser.parse_args()

    # compatibility to make calling of the program easier
    if options.parameter == "tprate":
        options.parameter = "precipitation_1h_bg"
    elif options.parameter in ["cc", "lcc", "mcc", "hcc"]:
        options.parameter = "total_cloud_cover"

    main()
