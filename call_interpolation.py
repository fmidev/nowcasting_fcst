# -*- coding: utf-8 -*-
import interpolate_fcst
from fileutils import read, write
import numpy as np
import argparse
import datetime
import configparser
import sys
import time
import cv2
from scipy.ndimage import gaussian_filter, distance_transform_edt

# from scipy.misc import imresize -> from PIL imort Image (imresize FEATURE IS NOT SUPPORTED ATM)


def farneback_params_config(config_file_name):
    config = configparser.ConfigParser()
    config.read(config_file_name)
    farneback_pyr_scale = config.getfloat("optflow", "pyr_scale")
    farneback_levels = config.getint("optflow", "levels")
    farneback_winsize = config.getint("optflow", "winsize")
    farneback_iterations = config.getint("optflow", "iterations")
    farneback_poly_n = config.getint("optflow", "poly_n")
    farneback_poly_sigma = config.getfloat("optflow", "poly_sigma")
    farneback_params = (
        farneback_pyr_scale,
        farneback_levels,
        farneback_winsize,
        farneback_iterations,
        farneback_poly_n,
        farneback_poly_sigma,
        0,
    )
    return farneback_params


def read_background_data_and_make_mask(
    image_file=None,
    input_mask=None,
    mask_smaller_than_borders=4,
    smoother_coefficient=0.2,
    gaussian_filter_coefficient=3,
):
    # mask_smaller_than_borders controls how many pixels the initial mask is compared to obs field
    # smoother_coefficient controls how long from the borderline bg field is affecting
    # gaussian_filter_coefficient sets the smoothiness of the weight field

    # Read in detectability data field and change all not-nodata values to zero
    if image_file is not None:
        (
            image_array4,
            quantity4_min,
            quantity4_max,
            timestamp4,
            mask_nodata4,
            nodata4,
        ) = read(image_file)
        image_array4[np.where(~np.ma.getmask(mask_nodata4))] = 0
        mask_nodata4_p = np.sum(np.ma.getmask(mask_nodata4), axis=0) > 0
    else:
        mask_nodata4_p = input_mask
    # Creating a linear smoother field: More weight for bg near the bg/obs border and less at the center of obs field
    # Gaussian smoother widens the coefficients so initially calculate from smaller mask the values
    used_mask = distance_transform_edt(
        np.logical_not(mask_nodata4_p)
    ) - distance_transform_edt(mask_nodata4_p)
    # Allow used mask to be bigger or smaller than initial mask given
    used_mask2 = np.where(used_mask <= mask_smaller_than_borders, True, False)
    # Combine with boolean input mask if that is given
    if input_mask is not None:
        used_mask2 = np.logical_or(used_mask2, input_mask)
    used_mask = distance_transform_edt(np.logical_not(used_mask2))
    weights_obs = gaussian_filter(used_mask, gaussian_filter_coefficient)
    weights_obs = weights_obs / (smoother_coefficient * np.max(weights_obs))
    # Cropping values to between 0 and 1
    weights_obs = np.where(weights_obs > 1, 1, weights_obs)
    # Leaving only non-zero -values which are inside used_mask2
    weights_obs = np.where(np.logical_not(used_mask2), weights_obs, 0)
    weights_bg = 1 - weights_obs

    return weights_bg


def define_common_mask_for_fields(*args):
    """Calculate a combined mask for each input. Some input values might have several timesteps, but here define a mask if ANY timestep for that particular gridpoint has a missing value"""
    stacked = np.sum(np.ma.getmaskarray(args[0]), axis=0) > 0
    if len(args) == 0:
        return stacked
    for arg in args[1:]:
        try:
            stacked = np.logical_or(
                stacked, (np.sum(np.ma.getmaskarray(arg), axis=0) > 0)
            )
        except:
            raise ValueError("grid sizes do not match!")
    return stacked


def main():
    # # For testing purposes set test datafiles
    # options.obs_data = "testdata/14/ppn_tprate_obs.grib2"
    # options.model_data = "testdata/14/fcst_tprate.grib2"
    # options.background_data = "testdata/14/mnwc_tprate.grib2"
    # options.dynamic_nwc_data = "testdata/14/mnwc_tprate_full.grib2"
    # options.extrapolated_data = "testdata/14/ppn_tprate.grib2"
    # options.detectability_data = "testdata/radar_detectability_field_255_280.h5"
    # options.output_data = "testdata/14/output/interpolated_tprate.grib2"
    # options.parameter = "precipitation_1h_bg"
    # options.mode = "model_fcst_smoothed"
    # options.predictability = 8

    # # For testing purposes set test datafiles
    # options.obs_data = None # "testdata/latest/obs_tp.grib2"
    # options.model_data = "testdata/14/fcst_2r.grib2"
    # options.background_data = None #"testdata/14/mnwc_tprate.grib2"
    # options.dynamic_nwc_data = "testdata/14/mnwc_2r.grib2"
    # options.extrapolated_data = None #testdata/14/ppn_tprate.grib2"
    # options.detectability_data = "testdata/radar_detectability_field_255_280.h5"
    # options.output_data = "testdata/14/output/interpolated_2r.grib2"
    # options.parameter = "2r"
    # options.mode = "model_fcst_smoothed"
    # options.predictability = 4

    #     # For testing purposes set test datafiles
    #     options.obs_data = None # "testdata/latest/obs_tp.grib2"
    #     options.model_data = "testdata/12/fcst_cc.grib2"
    #     options.background_data = "testdata/12/mnwc_cc.grib2"
    #     options.dynamic_nwc_data = "testdata/12/mnwc_cc.grib2"
    #     options.extrapolated_data = None #"testdata/12/ppn_tprate.grib2"
    #     options.detectability_data = "testdata/radar_detectability_field_255_280.h5"
    #     options.output_data = "testdata/12/output/interpolated_cc.grib2"
    #     options.parameter = "total_cloud_cover"
    #     options.mode = "model_fcst_smoothed"
    #     options.predictability = 8

    # Read parameters from config file for interpolation or optical flow algorithm.
    farneback_params = farneback_params_config(options.farneback_params)
    # give default values if no model_data
    nodata = None
    timestamp2 = None

    # For accumulated 1h precipitation, larger surrounding area for Farneback params are used: winsize 30 -> 150 and poly_n 20 -> 61
    if options.parameter == "precipitation_1h_bg":
        farneback_params = list(farneback_params)
        farneback_params[2] = 150
        farneback_params[4] = 61
        farneback_params = tuple(farneback_params)
    # Like mentioned at https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html, a reasonable value for poly_sigma depends on poly_n. Here we use a fraction 4.6 for these values.
    fb_params = (
        farneback_params[0],
        farneback_params[1],
        farneback_params[2],
        farneback_params[3],
        farneback_params[4],
        (farneback_params[4] / 4.6),
        farneback_params[6],
    )

    # Parameter name needs to be given as an argument!
    if options.parameter == None:
        raise NameError("Parameter name needs to be given as an argument!")

    # Model datafile needs to be given as an argument! This is obligatory, as obs/nwcdata is smoothed towards it!
    # Model data contains also the analysis time step!
    # ASSUMPTION USED IN THE REST OF THE CODE: model_data HAS ALWAYS ALL THE NEEDED TIME STEPS!
    # Exception! If parameter is precipitation 1h (and there is dynamic_nwc_data) no need of model_data. In this case a 9h nwc data is produced with PPN and MNWC
    # Exception! If parameter is total cloud cover!!!
    if options.model_data != None:
        # For accumulated model precipitation, add one hour to timestamp as it is read in as the beginning of the 1-hour period and not as the end of it
        if options.parameter == "precipitation_1h_bg":
            added_hours = 1
        else:
            added_hours = 0
        (
            image_array2,
            quantity2_min,
            quantity2_max,
            timestamp2,
            mask_nodata2,
            nodata2,
            longitudes2,
            latitudes2,
        ) = read(options.model_data, added_hours, use_as_template=True)
        quantity2 = options.parameter
        # nodata values are always taken from the model field. Presumably these are the same.
        nodata = nodata2
        # Model data is obligatory for this program!
        if np.sum((image_array2 != nodata2) & (image_array2 != None)) == 0:
            raise ValueError("Model datafile contains only missing data!")
            del (
                image_array2,
                quantity2_min,
                quantity2_max,
                timestamp2,
                mask_nodata2,
                nodata2,
                longitudes2,
                latitudes2,
            )
    # elif (options.parameter!='precipitation_1h_bg') & (options.model_data==None) & (options.parameter!='rainrate_15min_bg'):
    #    raise NameError("Model datafile needs to be given as an argument if not precipitation!")

    # Read in observation data (Time stamp is analysis time!)
    if options.obs_data != None:
        # For accumulated model precipitation, add one hour to timestamp as it is read in as the beginning of the 1-hour period and not as the end of it
        if options.parameter == "precipitation_1h_bg":
            added_hours = 1
        else:
            added_hours = 0
        (
            image_array1,
            quantity1_min,
            quantity1_max,
            timestamp1,
            mask_nodata1,
            nodata1,
            longitudes1,
            latitudes1,
        ) = read(options.obs_data, added_hours)
        quantity1 = options.parameter
        # If missing, remove variables and print warning text
        if np.sum((image_array1 != nodata1) & (image_array1 != None)) == 0:
            print("options.obs_data contains only missing data!")
            del (
                image_array1,
                quantity1_min,
                quantity1_max,
                timestamp1,
                mask_nodata1,
                nodata1,
                longitudes1,
                latitudes1,
            )

    # If observation data is supplemented with background data (to which Finnish obsdata is spatially smoothed), read it in, create a spatial mask and combine these two fields
    if options.background_data != None:
        if options.parameter == "precipitation_1h_bg":
            added_hours = 1
        else:
            added_hours = 0
        (
            image_array3,
            quantity3_min,
            quantity3_max,
            timestamp3,
            mask_nodata3,
            nodata3,
            longitudes3,
            latitudes3,
        ) = read(options.background_data, added_hours)
        quantity3 = options.parameter
        # If missing, remove variables and print warning text
        if np.sum((image_array3 != nodata3) & (image_array3 != None)) == 0:
            print("options.background_data contains only missing data!")
            del (
                image_array3,
                quantity3_min,
                quantity3_max,
                timestamp3,
                mask_nodata3,
                nodata3,
                longitudes3,
                latitudes3,
            )

    # Creating spatial composite for the first time step from obs and background data
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

    # Loading in dynamic_nwc_data. This can (or not) include the analysis step!
    if options.dynamic_nwc_data != None:
        if options.parameter == "precipitation_1h_bg":
            added_hours = 1
        else:
            added_hours = 0
        (
            image_arrayx1,
            quantityx1_min,
            quantityx1_max,
            timestampx1,
            mask_nodatax1,
            nodatax1,
            longitudesx1,
            latitudesx1,
        ) = read(options.dynamic_nwc_data, added_hours, use_as_template=True)
        quantityx1 = options.parameter
        # print(timestampx1)
        # If missing, remove variables and print warning text
        if np.sum((image_arrayx1 != nodatax1) & (image_arrayx1 != None)) == 0:
            print("options.dynamic_nwc_data contains only missing data!")
            del (
                image_arrayx1,
                quantityx1_min,
                quantityx1_max,
                timestampx1,
                mask_nodatax1,
                nodatax1,
                longitudesx1,
                latitudesx1,
            )
        # # Remove analysis timestamp from dynamic_nwc_data if it is there! (if previous analysis hour data is used!)
        # # if (options.parameter == 'precipitation_1h_bg'):
        # if (timestamp2[0] in timestampx1):
        #     if timestampx1.index(timestamp2[0]) == 0:
        #         image_arrayx1 = image_arrayx1[1:]
        #         timestampx1 = timestampx1[1:]
        #         mask_nodatax1 = mask_nodatax1[1:]

    # Loading in extrapolated_data (observations based nowcast).
    if options.extrapolated_data != None:
        if options.parameter == "precipitation_1h_bg":
            added_hours = 1
        else:
            added_hours = 0
        (
            image_arrayx2,
            quantityx2_min,
            quantityx2_max,
            timestampx2,
            mask_nodatax2,
            nodatax2,
            longitudesx2,
            latitudesx2,
        ) = read(options.extrapolated_data, added_hours)
        quantityx2 = options.parameter
        # If missing, remove variables and print warning text
        if np.sum((image_arrayx2 != nodatax2) & (image_arrayx2 != None)) == 0:
            print("options.extrapolated_data contains only missing data!")
            del (
                image_arrayx2,
                quantityx2_min,
                quantityx2_max,
                timestampx2,
                mask_nodatax2,
                nodatax2,
                longitudesx2,
                latitudesx2,
            )
        # # For 1h precipitation nowcasts, copy timestamps from timestamp2 (in case PPN timestamps are not properly parsed). Also, this run only supports fixed PPN runtimes (xx:00)
        # if (options.parameter == 'precipitation_1h_bg'):
        #     timestampx2 = timestamp2[1:(len(timestampx2)+1)]

    # If both extrapolated_data and dynamic_nwc_data are read in and the parameter is precipitation, combine them spatially by using mask
    # This contains spatial blending for PPN radar forecast, where first value in data is the 1h forecast, not analysis!
    # Instant precipitation rate and 1h precip accumulation
    if (
        "image_arrayx1" in locals()
        and "image_arrayx2" in locals()
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

    # Blending dynamic_nwc data with extrapolated_data when the parameter is NOT precipitation or rainrate (currently only used for cloudcast data)
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

    # IF NO MODEL DATA IS AVAILABLE
    # For precipitation the spatially combined, smoothed precip field is stored.
    # For other parameter the nwp nowcast and extrapolated data is the final product
    if options.parameter in ["precipitation_1h_bg", "rainrate_15min_bg"]:
        interpolated_advection = image_array1
    elif (
        "image_arrayx1" in locals()
        and options.parameter != "precipitation_1h_bg"
        and options.parameter != "rainrate_15min_bg"
    ):
        interpolated_advection = image_arrayx1

    # Do the rest if the model_data IS included
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

        #     # If the radar detectability field size does not match with that of background field, forcing this field to match that
        #     if (image_array4.shape[3:0:-1] != image_array3.shape[3:0:-1]):
        #         image_array4 = cv2.resize(image_array4[0,:,:], dsize=image_array3.shape[3:0:-1], interpolation=cv2.INTER_NEAREST)
        #         mask_nodata4 = np.ma.masked_where(image_array4 == nodata4,image_array4)
        #         # mask_nodata4 = cv2.resize(mask_nodata4[0,:,:], dsize=image_array3.shape[1:3], interpolation=cv2.INTER_NEAREST)
        #         image_array4 = np.expand_dims(image_array4, axis=0)
        #         mask_nodata4 = np.expand_dims(mask_nodata4, axis=0)
        #     # Checking if all latitude/longitude/timestamps in the different data sources correspond to each other
        #     if (np.array_equal(longitudes1,longitudes2) and np.array_equal(longitudes1,longitudes3) and np.array_equal(latitudes1,latitudes2) and np.array_equal(latitudes1,latitudes3)):
        #         longitudes = longitudes1
        #         latitudes = latitudes1
        #
        #     # Resize observation field to same resolution with model field and slightly blur to make the two fields look more similar for OpenCV.
        #     # NOW THE PARAMETER IS A CONSTANT AD-HOC VALUE!!!
        #     reshaped_size = list(image_array2.shape)
        #     if (options.mode == "analysis_fcst_smoothed"):
        #         reshaped_size[0] = 1
        #     image_array1_reshaped=np.zeros(reshaped_size)
        #     for n in range(0,image_array1.shape[0]):
        #         #Resize
        #         image_array1_reshaped[n]=imresize(image_array1[n], image_array2[0].shape, interp='bilinear', mode='F')
        #         #Blur
        #         image_array1_reshaped[n]=gaussian_filter(image_array1_reshaped[n], options.gaussian_filter_sigma)
        #     image_array1=image_array1_reshaped

        # Interpolate data in either analysis_fcst_smoothed or model_fcst_smoothed -mode
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

    # Implementing QC-thresholds! They are not atm given as parameters to the program!
    prec_max = 22
    if options.parameter == "precipitation_1h_bg":
        interpolated_advection[np.where(interpolated_advection > prec_max)] = prec_max
    CC_max = 1
    if options.parameter == "total_cloud_cover":
        interpolated_advection[np.where(interpolated_advection > CC_max)] = CC_max
    RH_max = 100
    if options.parameter == "2r":
        interpolated_advection[np.where(interpolated_advection > RH_max)] = RH_max
    POT_max = 100
    POT_min = 0
    if options.parameter == "pot":
        interpolated_advection[np.where(interpolated_advection > POT_max)] = POT_max
        interpolated_advection[np.where(interpolated_advection < POT_min)] = POT_min

    # Save interpolated field to a new file
    write(
        interpolated_data=interpolated_advection,
        image_file=None,
        write_file=options.output_data,
        variable=options.parameter,
        predictability=options.predictability,
        t_diff=options.time_offset,
        grib_write_options=options.grib_write_options,
    )

    ### PLOT OUT DIAGNOSTICS FROM THE DATA ###
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
    # parser.add_argument('--detectability_data',
    #                     default="testdata/radar_detectability_field_255_280.h5",
    #                     help='Radar detectability field, which is used in spatial blending of obsdata and bgdata')
    parser.add_argument(
        "--time_offset",
        help="Input/output grib metadata dataTime offset (positive values)",
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
