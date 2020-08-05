# -*- coding: utf-8 -*-
import interpolate_fcst
import h5py
import hiisi
import numpy as np
import argparse
import datetime
import ConfigParser
import netCDF4
import sys
import os
import time
import cv2
from eccodes import *
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt



### FUNCTIONS USED ###

def read(image_file,added_hours=0):
    if image_file.endswith(".nc"):
        return read_nc(image_file,added_hours)
    elif image_file.endswith(".grib2"):
        return read_grib(image_file,added_hours)
    elif image_file.endswith(".h5"):
        return read_HDF5(image_file,added_hours)
    else:
        print "unsupported file type for file: %s" % (image_file)
        


def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print "\t\ttype:", repr(nc_fid.variables[key].dtype)
            for ncattr in nc_fid.variables[key].ncattrs():
                print '\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr))
        except KeyError:
            print "\t\tWARNING: %s does not contain variable attributes" % key

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print "NetCDF Global Attributes:"
        for nc_attr in nc_attrs:
            print '\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print "NetCDF dimension information:"
        for dim in nc_dims:
            print "\tName:", dim 
            print "\t\tsize:", len(nc_fid.dimensions[dim])
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print "NetCDF variable information:"
        for var in nc_vars:
            if var not in nc_dims:
                print '\tName:', var
                print "\t\tdimensions:", nc_fid.variables[var].dimensions
                print "\t\tsize:", nc_fid.variables[var].size
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars



def read_nc(image_nc_file,added_hours):
    tempds = netCDF4.Dataset(image_nc_file)
    internal_variable = tempds.variables.keys()[-1]
    temps = np.array(tempds.variables[internal_variable][:]) # This picks the actual data
    nodata = tempds.variables[internal_variable].missing_value
    time_var = tempds.variables["time"]
    dtime = netCDF4.num2date(time_var[:],time_var.units) # This produces an array of datetime.datetime values
    
    # Outside of area all the values are missing. Leave them as they are. They're not affected by the motion vector calculations
    mask_nodata = np.ma.masked_where(temps == nodata,temps)
    # Pick min/max values from the data
    temps_min= temps[np.where(~np.ma.getmask(mask_nodata))].min()
    temps_max= temps[np.where(~np.ma.getmask(mask_nodata))].max()
    if type(dtime) == list:
        dtime = [(i+datetime.timedelta(hours=added_hours)) for i in dtime]
    else:
        dtime = dtime+datetime.timedelta(hours=added_hours)

    
    return temps, temps_min, temps_max, dtime, mask_nodata, nodata



def read_grib(image_grib_file,added_hours):

    # check comments from read_nc()
    dtime = []
    tempsl = []
    latitudes = []
    longitudes = []
    
    with GribFile(image_grib_file) as grib:
        for msg in grib:
            #print msg.size()
            #print msg.keys()
            ni = msg["Ni"]
            nj = msg["Nj"]
            #print(msg["dataDate"])
            #print(msg["dataTime"])
            #print(datetime.datetime.strptime("{:d}/{:02d}".format(msg["dataDate"], msg["dataTime"]/100), "%Y%m%d/%H"))
            forecast_time = datetime.datetime.strptime("{:d}/{:02d}".format(msg["dataDate"], msg["dataTime"]/100), "%Y%m%d/%H") + datetime.timedelta(hours=msg["forecastTime"])
            # forecast_time = datetime.datetime.strptime("%s%02d" % (msg["dataDate"], msg["dataTime"]), "%Y%m%d%H%M") + datetime.timedelta(hours=msg["forecastTime"])
            # print(forecast_time)
            dtime.append(forecast_time)
            tempsl.append(np.asarray(msg["values"]).reshape(nj, ni))
            latitudes.append(np.asarray(msg["latitudes"]).reshape(nj, ni))
            longitudes.append(np.asarray(msg["longitudes"]).reshape(nj, ni))

            
    temps = np.asarray(tempsl)
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    latitudes = latitudes[0,:,:]
    longitudes = longitudes[0,:,:]
    nodata = 9999

    mask_nodata = np.ma.masked_where(temps == nodata,temps)
    temps_min= temps[np.where(~np.ma.getmask(mask_nodata))].min()
    temps_max= temps[np.where(~np.ma.getmask(mask_nodata))].max()
    if type(dtime) == list:
        dtime = [(i+datetime.timedelta(hours=added_hours)) for i in dtime]
    else:
        dtime = dtime+datetime.timedelta(hours=added_hours)


    return temps, temps_min, temps_max, dtime, mask_nodata, nodata, longitudes, latitudes



def read_HDF5(image_h5_file,added_hours):
    #Read RATE or DBZH from hdf5 file
    print 'Extracting data from image h5 file'
    comp = hiisi.OdimCOMP(image_h5_file, 'r')
    # #Read RATE array if found in dataset      
    # test=comp.select_dataset('RATE')
    # if test != None:
    #     image_array=comp.dataset
    #     quantity='RATE'
    # else:
    #     #Look for DBZH array
    #     test=comp.select_dataset('DBZH')
    #     if test != None:
    #         image_array=comp.dataset
    #         quantity='DBZH'
    #     else:
    #         print 'Error: RATE or DBZH array not found in the input image file!'
    #         sys.exit(1)
    #Look for DBZH array
    test=comp.select_dataset('DBZH')
    if test != None:
        image_array=comp.dataset
        quantity='DBZH'
    else:
        print 'Error: RATE or DBZH array not found in the input image file!'
        sys.exit(1)
    if quantity == 'RATE':
        quantity_min = options.R_min
        quantity_max = options.R_max
    if quantity == 'DBZH':
        quantity_min = options.DBZH_min
        quantity_max = options.DBZH_max
    
    #Read nodata and undetect values from metadata for masking
    gen = comp.attr_gen('nodata')
    pair = gen.next()
    nodata = pair.value
    gen = comp.attr_gen('undetect')
    pair = gen.next()
    undetect = pair.value
    #Read gain and offset values from metadata
    gen = comp.attr_gen('gain')
    pair = gen.next()
    gain = pair.value
    gen = comp.attr_gen('offset')
    pair = gen.next()
    offset = pair.value
    #Read timestamp from metadata
    gen = comp.attr_gen('date')
    pair = gen.next()
    date = pair.value
    gen = comp.attr_gen('time')
    pair = gen.next()
    time = pair.value

    timestamp=date+time
    dtime = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")

    # Flip latitude dimension
    image_array = np.flipud(image_array)
    tempsl = []
    tempsl.append(image_array)
    temps = np.asarray(tempsl)
    
    #Masks of nodata and undetect
    mask_nodata = np.ma.masked_where(temps == nodata,temps)
    mask_undetect = np.ma.masked_where(temps == undetect,temps)
    #Change to physical values
    temps=temps*gain+offset
    #Mask undetect values as 0 and nodata values as nodata
    temps[np.where(np.ma.getmask(mask_undetect))] = 0
    temps[np.where(np.ma.getmask(mask_nodata))] = nodata
    
    temps_min= temps[np.where(~np.ma.getmask(mask_nodata))].min()
    temps_max= temps[np.where(~np.ma.getmask(mask_nodata))].max()
    if type(dtime) == list:
        dtime = [(i+datetime.timedelta(hours=added_hours)) for i in dtime]
    else:
        dtime = dtime+datetime.timedelta(hours=added_hours)


    return temps, temps_min, temps_max, dtime, mask_nodata, nodata



def write(interpolated_data,image_file,write_file,variable,predictability):

    if write_file.endswith(".nc"):
        write_nc(interpolated_data,image_file,write_file,variable,predictability)
    elif write_file.endswith(".grib2"):
        write_grib(interpolated_data,image_file,write_file,variable,predictability)
    else:
        print "unsupported file type for file: %s" % (image_file)
        return
    print "wrote file '%s'" % write_file



def write_grib(interpolated_data,image_grib_file,write_grib_file,variable,predictability):
    # (Almost) all the metadata is copied from modeldata.grib2
    try:
        os.remove(write_grib_file)
    except OSError,e:
        pass

    # Change data type of numpy array
    # interpolated_data = np.round(interpolated_data,2)
    # interpolated_data = interpolated_data.astype('float64')             

    #with GribFile(image_grib_file) as grib:
    #    for msg in grib:
    #        msg["generatingProcessIdentifier"] = 202
    #        msg["centre"] = 86
    #        msg["bitmapPresent"] = True
    #
    #        for i in range(interpolated_data.shape[0]):
    #            msg["forecastTime"] = i
    #            msg["values"] = interpolated_data[i].flatten()
    #
    #            with open(write_grib_file, "a") as out:
    #                msg.write(out)
    #        break # we use only the first grib message as a template

    # This edits each grib message individually
    with GribFile(image_grib_file) as grib:
        i=-1
        for msg in grib:
            msg["generatingProcessIdentifier"] = 202
            msg["centre"] = 86
            msg["bitmapPresent"] = True
            i = i+1 # msg["forecastTime"]
            if (i == interpolated_data.shape[0]):
                break
            msg["values"] = interpolated_data[i,:,:].flatten()
            # print("{} {}: {}".format("histogram of interpolated_data timestep ",i,np.histogram(interpolated_data[i,:,:].flatten(),bins=20,range=(240,300))))
            # print("{} {}: {}".format("histogram of msg[values] timestep ",i,np.histogram(msg["values"],bins=20,range=(240,300))))
            with open(write_grib_file, "a") as out:
                msg.write(out)



def write_nc(interpolated_data,image_nc_file,write_nc_file,variable,predictability):
    
    # All the metadata is copied from modeldata.nc
    nc_f = image_nc_file
    nc_fid = netCDF4.Dataset(nc_f, 'r')
    nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)
    # Open the output file
    w_nc_fid = netCDF4.Dataset(write_nc_file, 'w', format='NETCDF4')
    # Using our previous dimension information, we can create the new dimensions
    data = {}
    # Creating dimensions
    for dim in nc_dims:
        w_nc_fid.createDimension(dim, nc_fid.variables[dim].size)
    # Creating variables
    for var in nc_vars:
        data[var] = w_nc_fid.createVariable(varname=var, datatype=nc_fid.variables[var].dtype,dimensions=(nc_fid.variables[var].dimensions))
        # Assigning attributes for the variables
        for ncattr in nc_fid.variables[var].ncattrs():
            data[var].setncattr(ncattr, nc_fid.variables[var].getncattr(ncattr))
        # Assign the data itself
        if (var in ['time','y','x','lat','lon']):
            w_nc_fid.variables[var][:] = nc_fid.variables[var][:]
        if (var == nc_vars[len(nc_vars)-1]):
            w_nc_fid.variables[var][:] = interpolated_data
    # Creating the global attributes
    w_nc_fid.description = "Blended nowcast forecast, variable %s, predictability %s" %(variable, predictability)
    w_nc_fid.history = 'Created ' + time.ctime(time.time())  
    for attribute in nc_attrs:
        w_nc_fid.setncattr(attribute, nc_fid.getncattr(attribute))
        # w_nc_fid[attribute] = nc_fid.getncattr(attribute)
    # for attribute in nc_attrs:
    #     eval_string = 'w_nc_fid.' + attribute + ' = nc_fid.getncattr("' + attribute + '")'
    #     eval(eval_string)
    # Close the file
    w_nc_fid.close()



def farneback_params_config(config_file_name):
    config = ConfigParser.ConfigParser()
    config.read(config_file_name)
    farneback_pyr_scale  = config.getfloat("optflow", "pyr_scale")
    farneback_levels     = config.getint("optflow", "levels")
    farneback_winsize    = config.getint("optflow", "winsize")
    farneback_iterations = config.getint("optflow", "iterations")
    farneback_poly_n     = config.getint("optflow", "poly_n")
    farneback_poly_sigma = config.getfloat("optflow", "poly_sigma")
    farneback_params = (farneback_pyr_scale,  farneback_levels, farneback_winsize, 
                        farneback_iterations, farneback_poly_n, farneback_poly_sigma, 0)
    return farneback_params



def read_background_data_and_make_mask(image_file, input_mask=None, mask_smaller_than_borders=4, smoother_coefficient=0.2, gaussian_filter_coefficient=3):
    # mask_smaller_than_borders controls how many pixels the initial mask is compared to obs field
    # smoother_coefficient controls how long from the borderline bg field is affecting
    # gaussian_filter_coefficient sets the smoothiness of the weight field
    
    # Read in detectability data field and change all not-nodata values to zero
    image_array4, quantity4_min, quantity4_max, timestamp4, mask_nodata4, nodata4 = read(image_file)
    image_array4[np.where(~np.ma.getmask(mask_nodata4))] = 0
    mask_nodata4_p = np.sum(np.ma.getmask(mask_nodata4),axis=0)>0
    
    # Creating a linear smoother field: More weight for bg near the bg/obs border and less at the center of obs field
    # Gaussian smoother widens the coefficients so initially calculate from smaller mask the values
    used_mask = distance_transform_edt(np.logical_not(mask_nodata4_p)) - distance_transform_edt(mask_nodata4_p)
    # Allow used mask to be bigger or smaller than initial mask given
    used_mask2 = np.where(used_mask <= mask_smaller_than_borders,True,False)
    # Combine with boolean input mask if that is given
    if input_mask is not None:
        used_mask2 = np.logical_or(used_mask2,input_mask)
    used_mask = distance_transform_edt(np.logical_not(used_mask2))
    weights_obs = gaussian_filter(used_mask,gaussian_filter_coefficient)
    weights_obs = weights_obs / (smoother_coefficient*np.max(weights_obs))
    # Cropping values to between 0 and 1
    weights_obs = np.where(weights_obs>1,1,weights_obs)
    # Leaving only non-zero -values which are inside used_mask2
    weights_obs = np.where(np.logical_not(used_mask2),weights_obs,0)
    weights_bg = 1 - weights_obs
    
    return weights_bg



def define_common_mask_for_fields(*args):
    """Calculate a combined mask for each input. Some input values might have several timesteps, but here define a mask if ANY timestep for that particular gridpoint has a missing value"""
    stacked = (np.sum(np.ma.getmaskarray(args[0]),axis=0)>0)
    if (len(args)==0):
        return(stacked)
    for arg in args[1:]:
        try:
            stacked = np.logical_or(stacked,(np.sum(np.ma.getmaskarray(arg),axis=0)>0))
        except:
            raise ValueError("grid sizes do not match!")
    return(stacked)

    

def main():

#    # For testing purposes set test datafiles
#    options.obs_data = None # "testdata/07/obs_tp.grib2"
#    options.model_data = "testdata/07/fcst_tprate.grib2"
#    options.background_data = "testdata/07/mnwc_tprate.grib2"
#    options.dynamic_nwc_data = "testdata/07/mnwc_tprate_full.grib2"
#    options.extrapolated_data = "testdata/07/ppn_tprate.grib2"
#    options.detectability_data = "testdata/radar_detectability_field_255_280.h5"
#    options.output_data = "testdata/07/output/smoothed_mnwc_edited.grib2"
#    options.parameter = "precipitation_1h_bg"
#    options.mode = "model_fcst_smoothed"
#    options.predictability = 9
    
#    # For testing purposes set test datafiles
#    options.obs_data = None
#    options.model_data = "testdata/TCC/smartmet.grib2"
#    options.background_data = None
#    options.dynamic_nwc_data = "testdata/TCC/mnwc.grib2"
#    options.extrapolated_data = None
#    options.detectability_data = "testdata/radar_detectability_field_255_280.h5"
#    options.output_data = "testdata/TCC/output/smoothed_mnwc_edited.grib2"
#    options.parameter = "total_cloud_cover"
#    options.mode = "model_fcst_smoothed"
#    options.predictability = 9

    
    
    # Read parameters from config file for interpolation or optical flow algorithm.
    farneback_params=farneback_params_config(options.farneback_params)
    # For accumulated 1h precipitation, larger surrounding area for Farneback params are used: winsize 30 -> 150 and poly_n 20 -> 61
    if options.parameter == 'precipitation_1h_bg':
        farneback_params = list(farneback_params)
        farneback_params[2] = 150
        farneback_params[4] = 61
        farneback_params = tuple(farneback_params)
    # Like mentioned at https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html, a reasonable value for poly_sigma depends on poly_n. Here we use a fraction 4.6 for these values.
    fb_params = (farneback_params[0],farneback_params[1],farneback_params[2],farneback_params[3],farneback_params[4],(farneback_params[4] / 4.6),farneback_params[6])

    # Parameter name needs to be given as an argument!
    if options.parameter==None:
        raise NameError("Parameter name needs to be given as an argument!")
    
    # Model datafile needs to be given as an argument! This is obligatory, as obs/nwcdata is smoothed towards it!
    # Model data contains also the analysis time step!
    # ASSUMPTION USED IN THE REST OF THE CODE: model_data HAS ALWAYS ALL THE NEEDED TIME STEPS!
    if options.model_data!=None:
        # For accumulated model precipitation, add one hour to timestamp as it is read in as the beginning of the 1-hour period and not as the end of it
        if (options.parameter == 'precipitation_1h_bg'):
            added_hours = 1
        else:
            added_hours = 0
        image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2, nodata2, longitudes2, latitudes2 = read(options.model_data,added_hours)
        quantity2 = options.parameter
        # nodata values are always taken from the model field. Presumably these are the same.
        nodata = nodata2
    else:
        raise NameError("Model datafile needs to be given as an argument!")

    # Read in observation data (Time stamp is analysis time!)
    if options.obs_data!=None:
        image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1, nodata1, longitudes1, latitudes1 = read(options.obs_data,added_hours=0)
        quantity1 = options.parameter

    # If observation data is supplemented with background data (to which Finnish obsdata is spatially smoothed), read it in, create a spatial mask and combine these two fields
    if options.background_data!=None:
        if (options.parameter == 'precipitation_1h_bg'):
            added_hours = 1
        else:
            added_hours = 0
        image_array3, quantity3_min, quantity3_max, timestamp3, mask_nodata3, nodata3, longitudes3, latitudes3 = read(options.background_data,added_hours)
        quantity3 = options.parameter

    # Creating spatial composite for the first time step from obs and background data
    if 'image_array1' in locals() and 'image_array3' in locals():
        # Read radar composite field used as mask for Finnish data. Lat/lon info is not stored in radar composite HDF5 files, but these are the same! (see README.txt)
        weights_bg = read_background_data_and_make_mask(image_file=options.detectability_data, input_mask=define_common_mask_for_fields(mask_nodata1), mask_smaller_than_borders=4, smoother_coefficient=0.2, gaussian_filter_coefficient=3)
        weights_obs = 1 - weights_bg
        if (weights_bg.shape != image_array1.shape[1:] != image_array3.shape[1:]):
            raise ValueError("Model data, background data and image do not all have same grid size!")
        # Adding up the two fields (obs_data for area over Finland, bg field for area outside Finland)
        image_array1[0,:,:] = weights_obs*image_array1[0,:,:]+weights_bg*image_array3[0,:,:]
        mask_nodata1 = mask_nodata3
    # If background data is not available, but obs data is, DO NOTHING
    # If only background data is available, use that as image_array and timestamp
    if 'image_array3' in locals() and 'image_array1' not in locals():
        image_array1 = image_array3
        mask_nodata1 = mask_nodata3
        timestamp1 = timestamp3
        
    # Loading in dynamic_nwc_data. First value in data is the first forecast step, not analysis!
    if options.dynamic_nwc_data!=None:
        if (options.parameter == 'precipitation_1h_bg'):
            added_hours = 1
        else:
            added_hours = 0
        image_arrayx1, quantityx1_min, quantityx1_max, timestampx1, mask_nodatax1, nodatax1, longitudesx1, latitudesx1 = read(options.dynamic_nwc_data,added_hours)
        quantityx1 = options.parameter
        # Copy timestamps from timestamp2 (minus the first time step)
        if (options.parameter == 'precipitation_1h_bg'):
            timestampx1 = timestamp2[1:]
        
    # Loading in extrapolated_data. Currently supports only PPN data, where first value in data is the the 1h forecast, not analysis!
    if options.extrapolated_data!=None:
        image_arrayx2, quantityx2_min, quantityx2_max, timestampx2, mask_nodatax2, nodatax2, longitudesx2, latitudesx2 = read(options.extrapolated_data)
        quantityx2 = options.parameter
        # For 1h precipitation nowcasts, copy timestamps from timestamp2 (in case PPN timestamps are not properly parsed). Also, this run only supports fixed PPN runtimes (xx:00)
        if (options.parameter == 'precipitation_1h_bg'):
            timestampx2 = timestamp2[1:(len(timestampx2)+1)]
        
    # If both extrapolated_data and dynamic_nwc_data are read in, combine them spatially by using mask
    if 'image_arrayx1' in locals() and 'image_arrayx2' in locals():
        if type(timestampx1)==list and type(timestampx2)==list:
            # Finding out time steps in extrapolated_data that are also found in dynamic_nwc_data
            dynamic_nwc_data_common_indices = [timestampx1.index(x) if x in timestampx1 else None for x in timestampx2]
        if len(dynamic_nwc_data_common_indices)>0:
            image_arrayx3 = np.copy(image_arrayx1)
            # Combine data for each forecast length
            for common_index in range(0,len(dynamic_nwc_data_common_indices)):
                # A larger mask is used for longer forecast length (PPN advects precipitation data also outside Finnish borders -> increase mask by 8 pixels/hour)
                mask_inflation = common_index*6
                # Read radar composite field used as mask for Finnish data. Lat/lon info is not stored in radar composite HDF5 files, but these are the same! (see README.txt)
                weights_bg = read_background_data_and_make_mask(image_file=options.detectability_data, input_mask=define_common_mask_for_fields(mask_nodatax2), mask_smaller_than_borders=(-2-mask_inflation), smoother_coefficient=0.2, gaussian_filter_coefficient=3)
                weights_obs = 1 - weights_bg
                if (weights_bg.shape != image_arrayx1.shape[1:] != image_arrayx2.shape[1:]):
                    raise ValueError("Model data, background data and image do not all have same grid size!")
                # Adding up the two fields (obs_data for area over Finland, bg field for area outside Finland)
                if dynamic_nwc_data_common_indices[common_index]!=None:
                    image_arrayx3[dynamic_nwc_data_common_indices[common_index],:,:] = weights_obs*image_arrayx2[common_index,:,:] + weights_bg*image_arrayx1[dynamic_nwc_data_common_indices[common_index],:,:]
            # After nwc_dynamic_data and extrapolated_data have been spatially combined for all the forecast steps, then calculate interpolated values between the fields image_arrayx1 and image_arrayx3
            # As there always is more than one common timestamp between these data, always combine them using model_smoothing -method!
            if (options.mode == "model_fcst_smoothed" or options.mode == "analysis_fcst_smoothed"):
                # Defining mins and max in all data
                R_min_nwc = min(image_arrayx1.min(),image_arrayx3.min())
                R_max_nwc = max(image_arrayx1.max(),image_arrayx3.max())
                # Code only supports precipitation extrapolation data (PPN). Using other variables will cause an error. predictability/R_min/sigmoid_steepness are variable-dependent values! Here predictability is len(timestampx2) and not len(timestampx2)+1! -> last timestep of image_arrayx2 recieves a weight 0! 
                if (options.parameter == "precipitation_1h_bg"):
                   image_arrayx1 = interpolate_fcst.model_smoothing(obsfields=image_arrayx3, modelfields=image_arrayx1, mask_nodata=define_common_mask_for_fields(mask_nodatax1), farneback_params=fb_params, predictability=len(timestampx2), seconds_between_steps=options.seconds_between_steps, R_min=R_min_nwc, R_max=R_max_nwc, missingval=nodata, logtrans=False, sigmoid_steepness=-3)
                else:
                    raise ValueError("Only precipitation_1h_bg variable is supported by the code! Provide variable-dependent value for sigmoid_steepness! Revise also the bg mask used!")
            else:
                raise ValueError("Mode must be either model_fcst_smoothed or analysis_fcst_smoothed!")
        else:
            raise ValueError("Check your data! Only one common forecast step between dynamic_nwc_data and extrapolated_data and there's no sense in combining these two forecast sources spatially!")
    # If only dynamic_nwc_data is available, use that (so do nothing)
    # If only extrapolated_data is available, use that as nowcasting data
    if 'image_arrayx2' in locals() and 'image_arrayx1' not in locals():
        image_arrayx1 = image_arrayx2
        mask_nodatax1 = mask_nodatax2
        timestampx1 = timestampx2

        
    # If (obsdata) (like LAPS) is available, set that as the first time step in the combined dataset of extrapolated_data/dynamic_nwc_data
    if 'image_array1' in locals():
        # If nwc/extrapolated data is available
        if 'image_arrayx1' in locals():
            if image_array1.shape[1:]!=image_arrayx1.shape[1:]:
                print("obsdata and dynamic_nwc/extrapolation data have different grid sizes! Cannot combine!")
            else:
                nwc_model_indices = [timestamp2.index(x) if x in timestamp2 else None for x in timestampx1]
                obs_model_indices = timestamp2.index(timestamp1[0])
                # Use obsdata as such for the first time step and nwc data for the forecast steps after the analysis hour
                # Inflate image_arrayx1 if it has no analysis time step
                if nwc_model_indices[0]==1 and obs_model_indices==0:
                    image_array1 = np.append(image_array1,image_arrayx1,axis=0)
                    timestamp1.extend(timestampx1)
                # If image_arrayx1 contains also analysis step, replace first time step in image_arrayx1 with obsdata
                if nwc_model_indices[0]==0 and obs_model_indices==0 and len(nwc_model_indices>1):
                    image_array1 = np.append(image_array1,image_arrayx1[1:,:,:],axis=0)
                    timestamp1.extend(timestampx1[1:])
                mask_nodata1 = define_common_mask_for_fields(mask_nodata1,mask_nodatax1)
                # If nwc data is not either analysis time step or 1-hour forecast, throw an error
                if nwc_model_indices[0]>1:
                    raise ValueError("Check your nwc input data! It needs more short forecast time steps!")
    else:
        # If nwc/extrapolated data is available, use that
        if 'image_arrayx1' in locals():
            image_array1 = image_arrayx1
            timestamp1 = timestampx1
        else:
            raise ValueError("no obsdata or nwc data available! Cannot smooth anything!")

    # If needed, fill up "obsfields" data array with model data (so that timestamp2 and timestamp1 will eventually be the same)
    # Find out obsdata indices that coincide with modeldata (None values indicate time steps that there is no obsdata available for those modeldata time steps)
    model_obs_indices = [timestamp1.index(x) if x in timestamp1 else None for x in timestamp2]
    # If even a single forecast time index in model_data is missing from the obsfields data, fill it out with model data
    if (all(x!=None for x in model_obs_indices) == False):
        # Use modeldata as base data
        image_array_temp = np.copy(image_array2)
        timestamp_temp = [i for i in timestamp2]
        # Replace image_array_temp with obsdata for those time stamps that there is obsdata available
        # These obs indices will be assigned (Just remove None values from model_obs_indices)
        assigned_obs_indices = [i for i in model_obs_indices if i!=None]
        # Obsdata is assigned to these following indices
        model_assignable_indices = [model_obs_indices.index(i) for i in model_obs_indices if i!=None]
        image_array_temp[model_assignable_indices,:,:] = image_array1[assigned_obs_indices,:,:]
        image_array1 = np.copy(image_array_temp)
        timestamp1 = timestamp_temp
    # Now exists image_array1 (obs/nwc data) and image_array2 (model data)

    # Define nodata masks separately and commonly
    mask_nodata1 = np.ma.masked_where(image_array1 == nodata,image_array1)
    mask_nodata2 = np.ma.masked_where(image_array2 == nodata,image_array2)
    # Replace all values according to mask_nodata
    mask_nodata = define_common_mask_for_fields(mask_nodata1,mask_nodata2)
    image_array1[:,mask_nodata] = nodata
    image_array2[:,mask_nodata] = nodata
    
    # Checking out that model grid sizes correspond to each other
    if (image_array1.shape != image_array2.shape or timestamp1 != timestamp2):
        raise ValueError("image_array1.shape and image_array2.shape do not correspond to each other!")

    # Defining mins and max in all data
    R_min = min(image_array1.min(),image_array2.min())
    R_max = max(image_array1.max(),image_array2.max())

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
    if (options.mode == "analysis_fcst_smoothed"):
        interpolated_advection=interpolate_fcst.advection(obsfields=image_array1, modelfields=image_array2, mask_nodata=mask_nodata, farneback_params=fb_params, predictability=options.predictability, seconds_between_steps=options.seconds_between_steps, R_min=R_min, R_max=R_max, missingval=nodata, logtrans=False)
    if (options.mode == "model_fcst_smoothed"):
        interpolated_advection=interpolate_fcst.model_smoothing(obsfields=image_array1, modelfields=image_array2, mask_nodata=mask_nodata, farneback_params=fb_params, predictability=options.predictability, seconds_between_steps=options.seconds_between_steps, R_min=R_min, R_max=R_max, missingval=nodata, logtrans=False, sigmoid_steepness=-3.5)

        
    # Save interpolated field to a new file
    write(interpolated_data=interpolated_advection,image_file=options.model_data,write_file=options.output_data,variable=options.parameter,predictability=options.predictability)








    
    ### PLOT OUT DIAGNOSTICS FROM THE DATA ###
    if (options.plot_diagnostics == 'yes'):
        import diagnostics_functions
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        from mpl_toolkits.basemap import Basemap
        from mpl_toolkits.basemap import shiftgrid
        
        # Create directories which do not yet exist
        if not os.path.exists("figures/"):
            os.makedirs("figures/")
        if not os.path.exists("figures/fields/"):
            os.makedirs("figures/fields/")
        if not os.path.exists("figures/linear_change/"):
            os.makedirs("figures/linear_change/")
        if not os.path.exists("figures/linear_change3h/"):
            os.makedirs("figures/linear_change3h/")
        if not os.path.exists("figures/linear_change4h/"):
            os.makedirs("figures/linear_change4h/")
        if not os.path.exists("figures/jumpiness_absdiff/"):
            os.makedirs("figures/jumpiness_absdiff/")
        if not os.path.exists("figures/jumpiness_meandiff/"):
            os.makedirs("figures/jumpiness_meandiff/")
        if not os.path.exists("figures/jumpiness_ratio/"):
            os.makedirs("figures/jumpiness_ratio/")

        interpolated_advection_uusi, quantity1_min, quantity1_max, timestamp1, mask_nodata1, nodata1, longitudes, latitudes = read(options.output_data)
        outdir = "figures/fields/"

        # Plot unmodified obs data if it exists
        if options.obs_data!=None:
            image_array_obs_data, quantity_obs_data_min, quantity_obs_data_max, timestamp_obs_data, mask_nodata_obs_data, nodata_obs_data, longitudes_obs_data, latitudes_obs_data = read(options.obs_data)
            quantity_plot = options.parameter
            # Plotting LAPS field as it is in the uncombined file
            title = "LAPS "
            outfile = outdir + "image_array_obs_data.png"
            diagnostics_functions.plot_imshow_on_map(image_array_obs_data[0,:,:],0,1,outfile,"jet",title,longitudes,latitudes)

        # Plot unmodified background data if it exists
        if options.background_data!=None:
            image_array_plot, quantity_plot_min, quantity_plot_max, timestamp_plot, mask_nodata_plot, nodata_plot, longitudes_plot, latitudes_plot = read(options.background_data)
            quantity_plot = options.parameter
            title = "MNWC 0hours"
            outfile = outdir + "image_array_MNWC_0hours.png"
            diagnostics_functions.plot_imshow_on_map(image_array_plot[0,:,:],0,1,outfile,"jet",title,longitudes,latitudes)
            if 'mask_nodata_obs_data' in locals():
                # Read radar composite field used as mask for Finnish data. Lat/lon info is not stored in radar composite HDF5 files, but these are the same! (see README.txt)
                weights_bg = read_background_data_and_make_mask(image_file=options.detectability_data, input_mask=define_common_mask_for_fields(mask_nodata_obs_data), mask_smaller_than_borders=4, smoother_coefficient=0.2, gaussian_filter_coefficient=3)
                weights_obs = 1 - weights_bg
                title = "weights "
                outfile = outdir + "image_array_weights.png"
                diagnostics_functions.plot_imshow_on_map(weights_bg,0,1,outfile,"jet",title,longitudes,latitudes)

        # Plot dynamic_nwc_data if it exists
        if options.dynamic_nwc_data!=None:
            if (options.parameter == 'precipitation_1h_bg'):
                added_hours = 1
            else:
                added_hours = 0
            image_array_plot, quantity_plot_min, quantity_plot_max, timestamp_plot, mask_nodata_plot, nodata_plot, longitudes_plot, latitudes_plot = read(options.dynamic_nwc_data,added_hours)
            for n in (range(0, image_array_plot.shape[0])):
                outdir = "figures/fields/"
                # PLOTTING AND SAVING TO FILE
                title = options.parameter + " MNWC fc=+" + str(n+added_hours) + "h"
                outfile = outdir + "field" + options.parameter + " MNWC_fc=+" + str(n) + "h.png"
                diagnostics_functions.plot_imshow(image_array_plot[n,:,:],R_min,R_max,outfile,"jet",title)

        # Plot extrapolated_data if it exists
        if options.extrapolated_data!=None:
            if (options.parameter == 'precipitation_1h_bg'):
                added_hours = 1
            else:
                added_hours = 0
            image_array_plot, quantity_plot_min, quantity_plot_max, timestamp_plot, mask_nodata_plot, nodata_plot, longitudes_plot, latitudes_plot = read(options.extrapolated_data)
            for n in (range(0, image_array_plot.shape[0])):
                outdir = "figures/fields/"
                # PLOTTING AND SAVING TO FILE
                title = options.parameter + " extrapolated_data fc=+" + str(n+added_hours) + "h"
                outfile = outdir + "field" + options.parameter + " extrapolated_data_fc=+" + str(n) + "h.png"
                diagnostics_functions.plot_imshow(image_array_plot[n,:,:],R_min,R_max,outfile,"jet",title)


        # NOW PLOT DIAGNOSTICS FROM THE FIELDS

        # TIME SERIES FROM THE FIELD MEAN
        fc_lengths=np.arange(0,interpolated_advection.shape[0])
        outdir = "figures/"
        outfile = outdir + "Field_mean_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + ".png"
        plt.plot(fc_lengths, np.mean(interpolated_advection,axis=(1,2)), linewidth=2.0, label="temperature")
        title = "Field mean, " + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S")
        plt.title(title)
        plt.tight_layout(pad=0.)
        # plt.xticks([])
        # plt.yticks([])
        plt.savefig(outfile,bbox_inches='tight', pad_inches=0)
        plt.close()

        # JUMPINESS CHECKS
        # PREVAILING ASSUMPTION: THE CHANGE IN INDIVIDUAL GRIDPOINTS IS VERY LINEAR

        # RESULT ARRAYS
        linear_change_boolean = np.ones(interpolated_advection.shape)
        gp_abs_difference = np.ones(interpolated_advection.shape)
        gp_mean_difference = np.ones(interpolated_advection.shape)
        ratio_meandiff_absdiff = np.ones(interpolated_advection.shape)

        # 0) PLOT image_array1 FIELDS
        for n in (range(0, image_array1.shape[0])):
            outdir = "figures/fields/"
            # PLOTTING AND SAVING TO FILE
            title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + "nwc_data=+" + str(n) + "h"
            outfile = outdir + "field" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_nwc_data=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(image_array1[n,:,:],R_min,R_max,outfile,"jet",title)

        # 0a) PLOT image_array2 FIELDS
        if options.model_data!=None:
            image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2, nodata2, longitudes2, latitudes2 = read(options.model_data,added_hours=0)
            quantity2 = options.parameter
            # nodata values are always taken from the model field. Presumably these are the same.
            nodata = nodata2
            for n in (range(0, image_array2.shape[0])):
                outdir = "figures/fields/"
                # PLOTTING AND SAVING TO FILE
                title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + "fc_model=+" + str(n) + "h"
                outfile = outdir + "field" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc_model=+" + str(n) + "h.png"
                diagnostics_functions.plot_imshow(image_array2[n,:,:],R_min,R_max,outfile,"jet",title)

        # 0b) PLOT DMO FIELDS
        for n in (range(0, interpolated_advection.shape[0])):
            outdir = "figures/fields/"
            # PLOTTING AND SAVING TO FILE
            title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + "fc=+" + str(n) + "h"
            outfile = outdir + "field" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(interpolated_advection[n,:,:],R_min,R_max,outfile,"jet",title)

        # 0c) PLOT DMO FIELDS
        for n in (range(0, interpolated_advection_uusi.shape[0])):
            outdir = "figures/fields/"
            # PLOTTING AND SAVING TO FILE
            title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + "fcip=+" + str(n) + "h"
            outfile = outdir + "field" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fcip=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(interpolated_advection_uusi[n,:,:],R_min,R_max,outfile,"jet",title)

        # 0d) PLOT DMO FIELDS
        for n in (range(0, interpolated_advection_uusi.shape[0])):
            outdir = "figures/fields/"
            # PLOTTING AND SAVING TO FILE
            title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + "fc_minus_model=+" + str(n) + "h"
            outfile = outdir + "field" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fcip_minus_model=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(interpolated_advection_uusi[n,:,:]-image_array2[n,:,:],-1,1,outfile,"jet",title)




        # 1a) TRUE IS ASSIGNED TO THOSE TIME+LOCATION POINTS THAT HAVE PREVIOUS TIMESTEP VALUE LESS (MORE) AND NEXT TIMESTEP VALUE MORE (LESS) THAN THE CORRESPONDING VALUE. SO NO CHANGE IN THE SIGN OF THE DERIVATIVE.
        for n in (range(1, 5)):
            outdir = "figures/linear_change/"
            gp_increased = (interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:] > 0) & (interpolated_advection[(n+1),:,:] - interpolated_advection[n,:,:] > 0)
            gp_reduced = (interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:] < 0) & (interpolated_advection[(n+1),:,:] - interpolated_advection[n,:,:] < 0)
            linear_change_boolean[n,:,:] = gp_reduced + gp_increased
            gp_outside_minmax_range = np.max(np.maximum(np.maximum(0,(interpolated_advection[n,:,:]-np.maximum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+1),:,:]))),abs(np.minimum(0,(interpolated_advection[n,:,:]-np.minimum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+1),:,:]))))))
            # PLOTTING AND SAVING TO FILE
            title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n) + "h, " + str(round(np.mean(linear_change_boolean[n,:,:])*100,2)) + "% gridpoints is \n inside range " + str(n-1) + "h..." + str(n+1) + "h, outside range field max value " + str(round(gp_outside_minmax_range,2))
            outfile = outdir + "linear_change_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(linear_change_boolean[n,:,:],0,1,outfile,"jet",title)

        # 1b) TRUE IS ASSIGNED TO THOSE TIME+LOCATION POINTS THAT HAVE PREVIOUS TIMESTEP VALUE LESS (MORE) AND NEXT+1 TIMESTEP VALUE MORE (LESS) THAN THE CORRESPONDING VALUE. SO NO CHANGE IN THE SIGN OF THE DERIVATIVE.
        for n in (range(1, 4)):
            outdir = "figures/linear_change3h/"
            gp_increased = ((interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:]) > 0) & ((interpolated_advection[(n+2),:,:] - interpolated_advection[n,:,:]) > 0)
            gp_reduced = ((interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:]) < 0) & ((interpolated_advection[(n+2),:,:] - interpolated_advection[n,:,:]) < 0)
            linear_change_boolean[n,:,:] = gp_reduced + gp_increased
            gp_outside_minmax_range = np.max(np.maximum(np.maximum(0,(interpolated_advection[n,:,:]-np.maximum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+2),:,:]))),abs(np.minimum(0,(interpolated_advection[n,:,:]-np.minimum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+2),:,:]))))))
            # PLOTTING AND SAVING TO FILE
            title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n) + "h, " + str(round(np.mean(linear_change_boolean[n,:,:])*100,2)) + "% gridpoints is \n inside range " + str(n-1) + "h..." + str(n+2) + "h, outside range field max value " + str(round(gp_outside_minmax_range,2))
            outfile = outdir + "linear_change_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(linear_change_boolean[n,:,:],0,1,outfile,"jet",title)

        # 1c) TRUE IS ASSIGNED TO THOSE TIME+LOCATION POINTS THAT HAVE PREVIOUS TIMESTEP VALUE LESS (MORE) AND NEXT+2 TIMESTEP VALUE MORE (LESS) THAN THE CORRESPONDING VALUE. SO NO CHANGE IN THE SIGN OF THE DERIVATIVE.
        for n in (range(1, 3)):
            outdir = "figures/linear_change4h/"
            gp_increased = (interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:] > 0) & (interpolated_advection[(n+3),:,:] - interpolated_advection[n,:,:] > 0)
            gp_reduced = (interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:] < 0) & (interpolated_advection[(n+3),:,:] - interpolated_advection[n,:,:] < 0)
            linear_change_boolean[n,:,:] = gp_reduced + gp_increased
            gp_outside_minmax_range = np.max(np.maximum(np.maximum(0,(interpolated_advection[n,:,:]-np.maximum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+3),:,:]))),abs(np.minimum(0,(interpolated_advection[n,:,:]-np.minimum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+3),:,:]))))))
            # PLOTTING AND SAVING TO FILE
            title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n) + "h, " + str(round(np.mean(linear_change_boolean[n,:,:])*100,2)) + "% gridpoints is \n inside range " + str(n-1) + "h..." + str(n+3) + "h, outside range field max value " + str(round(gp_outside_minmax_range,2))
            outfile = outdir + "linear_change_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(linear_change_boolean[n,:,:],0,1,outfile,"jet",title)


        # ANALYSING "JUMPINESS"
        for n in (range(1, 5)):
            # 2) DIFFERENCE OF TIMESTEPS (n-1) AND (n+1)
            gp_abs_difference[n,:,:] = abs(interpolated_advection[(n+1),:,:] - interpolated_advection[(n-1),:,:])
            outdir = "figures/jumpiness_absdiff/"
            title = options.parameter + " absdiff of " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n-1) + "h" + str(n+1) + "h \n field max value " + str(round(np.max(gp_abs_difference[n,:,:]),2))
            outfile = outdir + "jumpiness_absdiff_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(gp_abs_difference[n,:,:],-5,5,outfile,"seismic",title)
            # 3) DIFFERENCE FROM THE MEAN OF (n-1) AND (n+1)
            gp_mean_difference[n,:,:] = abs(interpolated_advection[n,:,:] - ((interpolated_advection[(n+1),:,:] + interpolated_advection[(n-1),:,:])/2))
            outdir = "figures/jumpiness_meandiff/"
            title = options.parameter + " diff from " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n-1) + "h" + str(n+1) + "h_mean \n field max value " + str(round(np.max(gp_mean_difference[n,:,:]),2))
            outfile = outdir + "jumpiness_meandiff_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(gp_mean_difference[n,:,:],-0.2,0.2,outfile,"seismic",title)
            # 4) MEAN DIFF DIVIDED BY ABSDIFF
            ratio_meandiff_absdiff = gp_mean_difference / gp_abs_difference
            outdir = "figures/jumpiness_ratio/"
            title = options.parameter + " meandiff / absdiff ratio % " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n-1) + "h" + str(n+1) + "h \n field max value " + str(round(np.max(ratio_meandiff_absdiff[n,:,:]),2))
            outfile = outdir + "jumpiness_ratio_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(ratio_meandiff_absdiff[n,:,:],0,2,outfile,"seismic",title)

    


if __name__ == '__main__':
    #Parse commandline arguments
    parser = argparse.ArgumentParser(argument_default=None)
    parser.add_argument('--obs_data',
                        help='Observation data field or similar, used as 0h forecast')
    parser.add_argument('--model_data',
                        help='Model data field, towards which the nowcast is smoothed')
    parser.add_argument('--background_data',
                        help='Background data field for the 0h forecast where obsdata is spatially merged to')
    parser.add_argument('--dynamic_nwc_data',
                        help='Dynamic nowcasting model data field, which is smoothed to modeldata. If extrapolated_data is provided, it is spatially smoothed with dynamic_nwc_data. First timestep of 0h should not be included in this data!')
    parser.add_argument('--extrapolated_data',
                        help='Nowcasting model data field acquired using extrapolation methods (like PPN), which is smoothed to modeldata. If dynamic_nwc_data is provided, extrapolated_data is spatially smoothed with it. First timestep of 0h should not be included in this data!')
    parser.add_argument('--detectability_data',
                        default="testdata/radar_detectability_field_255_280.h5",
                        help='Radar detectability field, which is used in spatial blending of obsdata and bgdata')
    parser.add_argument('--output_data',
                        help='Output file name for nowcast data field')
    parser.add_argument('--seconds_between_steps',
                        type=int,
                        default=3600,
                        help='Timestep of output data in seconds')
    parser.add_argument('--predictability',
                        type=int,
                        default='4',
                        help='Predictability in hours. Between the analysis and forecast of this length, forecasts need to be interpolated')
    parser.add_argument('--parameter',
                        help='Variable which is handled.')
    parser.add_argument('--mode',
                        default='analysis_fcst_smoothed',
                        help='Either "analysis_fcst_smoothed" or "model_fcst_smoothed" mode. In "analysis_fcst_smoothed" mode, nowcasts are interpolated between 0h (obs_data/background_data) and predictability hours (model_data). In "model_fcst_smoothed" mode, nowcasts are individually interpolated for each forecast length between dynamic_nwc_data/extrapolated/data and model_data and their corresponding forecasts')
    parser.add_argument('--gaussian_filter_sigma',
                        type=float,
                        default=0.5,
                        help='This parameter sets the blurring intensity of the of the analysis field.')
    parser.add_argument('--R_min',
                        type=float,
                        default=0.1,
                        help='Minimum precipitation intensity for optical flow computations. Values below R_min are set to zero.')
    parser.add_argument('--R_max',
                        type=float,
                        default=30.0,
                        help='Maximum precipitation intensity for optical flow computations. Values above R_max are clamped.')
    parser.add_argument('--DBZH_min',
                        type=float,
                        default=10,
                        help='Minimum DBZH for optical flow computations. Values below DBZH_min are set to zero.')
    parser.add_argument('--DBZH_max', 
                        type=float,
                        default=45,
                        help='Maximum DBZH for optical flow computations. Values above DBZH_max are clamped.')
    parser.add_argument('--farneback_params',
                        default='compute_advinterp.cfg',
                        help='location of farneback params configuration file')
    parser.add_argument('--plot_diagnostics',
                        default='no',
                        help='If this option is set to yes, program plots out several diagnostics to files.')

    options = parser.parse_args()
    main()
