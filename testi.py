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


# FUNCTIONS USED

def read(image_file):
    if image_file.endswith(".nc"):
        return read_nc(image_file)
    elif image_file.endswith(".grib2"):
        return read_grib(image_file)
    elif image_file.endswith(".h5"):
        return read_HDF5(image_file)
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



def read_nc(image_nc_file):
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

    return temps, temps_min, temps_max, dtime, mask_nodata, nodata



def read_grib(image_grib_file):

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

            forecast_time = datetime.datetime.strptime("%s%02d" % (msg["dataDate"], msg["dataTime"]), "%Y%m%d%H%M") + datetime.timedelta(hours=msg["forecastTime"])
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

    return temps, temps_min, temps_max, dtime, mask_nodata, nodata, longitudes, latitudes



def read_HDF5(image_h5_file):
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



def main():

    print(options)

    # Read parameters from config file for interpolation or optical flow algorithm.
    farneback_params=farneback_params_config(options.farneback_params)
    # For accumulated 1h precipitation, larger surrounding area for Farneback params are used: winsize 30 -> 150 and poly_n 20 -> 61
    if options.parameter == 'precipitation_bg_1h':
        farneback_params = list(farneback_params)
        farneback_params[2] = 150
        farneback_params[4] = 61
        farneback_params = tuple(farneback_params)

    # Parameter name needs to be given as an argument!
    if options.parameter==None:
        raise NameError("Parameter name needs to be given as an argument!")
    
    # Model datafile needs to be given as an argument! This is obligatory, as obs/nwcdata is smoothed towards it!
    if options.modeldata!=None:
        image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2, nodata2, longitudes2, latitudes2 = read(options.modeldata)
        quantity2 = options.parameter
    else:
        raise NameError("Model datafile needs to be given as an argument!")

    # Read in observation data (forecast step 0)
    if options.obsdata!=None:
       image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1, nodata1, longitudes1, latitudes1 = read(options.obsdata)
       quantity1 = options.parameter

       


if __name__ == '__main__':

    #Parse commandline arguments
    parser = argparse.ArgumentParser(argument_default=None)
    parser.add_argument('--obs_data',
                        help='Obs data, representing the first time step used in image morphing.')
    parser.add_argument('--model_data',
                        help='Model data, from the analysis timestamp up until the end of the available 10-day forecast.')
    parser.add_argument('--background_data',
                        help='Background field data where obsdata is merged to for the forecast step t=0, so having the same timestamp as obs data.')
    parser.add_argument('--dynamic_nwc_data',
                        help='Dynamic nowcasting model data. This is smoothed to modeldata. If extrapolation data is provided, it is spatially smoothed to bgmodeldata')
    parser.add_argument('--extrapolated_nwc_data',
                        help='Nowcasting model data acquired using extrapolation methods (like PPN)')
    parser.add_argument('--detectability_data',
                        default="testdata/radar_detectability_field_255_280.h5",
                        help='Radar detectability field, which is used in spatial blending of obsdata and bgdata.')
    parser.add_argument('--interpolated_data',
                        help='Output file name for nowcast data.')
    parser.add_argument('--seconds_between_steps',
                        type=int,
                        default=3600,
                        help='Seconds between interpolated steps.')
    parser.add_argument('--predictability',
                        type=int,
                        default='4',
                        help='Predictability in hours. Between the analysis and forecast of this length, forecasts need to be interpolated')
    parser.add_argument('--parameter',
                        help='Variable which is handled.')
    parser.add_argument('--mode',
                        default='analysis_fcst_smoothed',
                        help='Either "verif", "analysis_fcst_smoothed" or "model_fcst_smoothed" mode. In verification mode, verification statistics are calculated from the blended forecasts. In forecast mode no.')
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
                        help='Minimum DBZH for optical flow computations. Values below DBZH_min are clamped.')
    parser.add_argument('--DBZH_max', 
                        type=float,
                        default=45,
                        help='Maximum DBZH for optical flow computations. Values above DBZH_max are clamped.')
    parser.add_argument('--farneback_params',
                        default='compute_advinterp.cfg',
                        help='location of farneback params configuration file')

    options = parser.parse_args()
    main()
