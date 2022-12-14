from eccodes import *
import h5py
import netCDF4
import sys
import time
import datetime
import numpy as np
import os
import fsspec

GRIB_MESSAGE_TEMPLATE = None
GRIB_MESSAGE_STEP = None

def read(image_file,added_hours=0,read_coordinates=False,use_as_template=False):
    if image_file.endswith(".nc"):
        print(f"Reading {image_file}")
        return read_nc(image_file,added_hours)
    elif image_file.endswith(".grib2"):
        return read_grib(image_file,added_hours,read_coordinates,use_as_template)
    elif image_file.endswith(".h5"):
        print(f"Reading {image_file}")
        return read_HDF5(image_file,added_hours)
    else:
        sys.exit("unsupported file type for file: %s" % (image_file))

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
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def read_nc(image_nc_file,added_hours):
    tempds = netCDF4.Dataset(image_nc_file)
    internal_variable = list(tempds.variables.keys())[-1]
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


def read_file_from_s3(grib_file):
    uri = "simplecache::{}".format(grib_file)

    return fsspec.open_local(uri, s3={'anon': True, 'client_kwargs': { 'endpoint_url':'https://lake.fmi.fi'}})


def read_grib(grib_file,added_hours,read_coordinates,use_as_template=False):
    global GRIB_MESSAGE_TEMPLATE
    global GRIB_MESSAGE_STEP

    start = time.time()
    def read_leadtime(gh):
       tr = codes_get_long(gh, "indicatorOfUnitOfTimeRange")
       ft = codes_get_long(gh, "forecastTime")
       if tr == 1:
           return datetime.timedelta(hours=ft)
       if tr == 0:
           return datetime.timedelta(minutes=ft)

       raise Exception("Unknown indicatorOfUnitOfTimeRange: {:%d}".format(tr))


    # check comments from read_nc()
    dtime = []
    tempsl = []
    latitudes = []
    longitudes = []

    wrk_grib_file = grib_file

    if grib_file.startswith("s3://"):
        wrk_grib_file = read_file_from_s3(grib_file)


    with open(wrk_grib_file) as fp:
        while True:
            gh = codes_grib_new_from_file(fp)
            if gh is None:
                break

            ni = codes_get_long(gh, "Ni")
            nj = codes_get_long(gh, "Nj")
            data_date = codes_get_long(gh, "dataDate")
            data_time = codes_get_long(gh, "dataTime")
            lt = read_leadtime(gh)
            forecast_time = datetime.datetime.strptime("{:d}/{:04d}".format(data_date, data_time), "%Y%m%d/%H%M") + lt
            dtime.append(forecast_time)
            values = np.asarray(codes_get_values(gh))
            tempsl.append(values.reshape(nj, ni))
            if read_coordinates:
                shape = codes_get_long(gh, "shapeOfTheEarth")
                if shape not in (0,1,6):
                    print("Error: Data is defined in a spheroid which eccodes can't derive coordinates from. Another projection library such as proj4 should be used")
                    sys.exit(1)
                latitudes.append(np.asarray(codes_get_array(gh, "latitudes").reshape(nj, ni)))
                longitudes.append(np.asarray(codes_get_array(gh, "longitudes").reshape(nj, ni)))

            if use_as_template:
                if GRIB_MESSAGE_TEMPLATE is None:
                    GRIB_MESSAGE_TEMPLATE = codes_clone(gh)

                if GRIB_MESSAGE_STEP is None and lt > datetime.timedelta(minutes=0):
                    GRIB_MESSAGE_STEP = lt

            if codes_get_long(gh, "numberOfMissing") == ni*nj:
                print("File {} leadtime {} contains only missing data!".format(grib_file, lt))
                sys.exit(1)

            codes_release(gh)

    temps = np.asarray(tempsl)
    if len(latitudes) > 0:
        latitudes = np.asarray(latitudes)
        longitudes = np.asarray(longitudes)
        latitudes = latitudes[0,:,:]
        longitudes = longitudes[0,:,:]

    nodata = 9999
#
#    temps_min = None
#    temps_max = None
#
    mask_nodata = np.ma.masked_where(temps == nodata,temps)
#    if options.plot_diagnostics == "yes":
#        if len(temps[np.where(~np.ma.getmask(mask_nodata))])>0:
#            temps_min = temps[np.where(~np.ma.getmask(mask_nodata))].min()
#            temps_max = temps[np.where(~np.ma.getmask(mask_nodata))].max()
#        else:
#            print("input " + grib_file + " contains only missing data!")
#            temps_min = nodata
#            temps_max = nodata

    if type(dtime) == list:
        dtime = [(i+datetime.timedelta(hours=added_hours)) for i in dtime]
    else:
        dtime = dtime+datetime.timedelta(hours=added_hours)

    print("Read {} in {:.2f} seconds".format(grib_file, time.time() - start))
#    return temps, temps_min, temps_max, dtime, mask_nodata, nodata, longitudes, latitudes
    return temps, None, None, dtime, mask_nodata, nodata, longitudes, latitudes


def read_HDF5(image_h5_file,added_hours):
    #Read RATE or DBZH from hdf5 file
    print('Extracting data from image h5 file')
    hf = h5py.File(image_h5_file,'r')
    ### hiisi way
    # comp = hiisi.OdimCOMP(image_h5_file, 'r')
    #Look for DBZH or RATE array
    if (str(hf['/dataset1/data1/what'].attrs.__getitem__('quantity')) in ("b'DBZH'","b'RATE'")) == False:
        print('Error: RATE or DBZH array not found in the input image file!')
        sys.exit(1)

    # Read actual data in
    image_array = hf['/dataset1/data1/data'][:]

    #Read nodata/undetect/gain/offset/date/time values from metadata
    nodata = hf['/dataset1/data1/what'].attrs.__getitem__('nodata')
    undetect = hf['/dataset1/data1/what'].attrs.__getitem__('undetect')
    gain = hf['/dataset1/data1/what'].attrs.__getitem__('gain')
    offset = hf['/dataset1/data1/what'].attrs.__getitem__('offset')
    date = str(hf['/what'].attrs.__getitem__('date'))[2:-1]
    time = str(hf['/what'].attrs.__getitem__('time'))[2:-1]

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


def write(interpolated_data,image_file,write_file,variable,predictability,t_diff,grib_write_options):

    if write_file.endswith(".nc"):
        write_nc(interpolated_data,image_file,write_file,variable,predictability,t_diff)
    elif write_file.endswith(".grib2"):
        write_grib(interpolated_data,write_file,t_diff,grib_write_options)
    else:
        print("write: unsupported file type for file: %s" % (image_file))
        return
    print("wrote file '%s'" % write_file)


def write_grib_message(fpout,interpolated_data,t_diff,write_options):
    global GRIB_MESSAGE_TEMPLATE
    global GRIB_MESSAGE_STEP
    assert(GRIB_MESSAGE_TEMPLATE is not None)

    # For 1km PPN+MNWC forecast adjust the output grib dataTime (analysis time) since the 1h leadtime is used instead of 0h. Metadata taken from MNWC 
    if t_diff == None:
        t_diff = 0
    t_diff = int(t_diff)

    dataDate = int(codes_get_long(GRIB_MESSAGE_TEMPLATE, "dataDate"))
    dataTime = int(codes_get_long(GRIB_MESSAGE_TEMPLATE, "dataTime")) 
    analysistime = datetime.datetime.strptime("{}{:04d}".format(dataDate, dataTime), "%Y%m%d%H%M")
    analysistime = analysistime + datetime.timedelta(hours=t_diff)
    codes_set_long(GRIB_MESSAGE_TEMPLATE, "dataDate", int(analysistime.strftime("%Y%m%d")))
    codes_set_long(GRIB_MESSAGE_TEMPLATE, "dataTime", int(analysistime.strftime("%H%M")))
    codes_set_long(GRIB_MESSAGE_TEMPLATE, "bitsPerValue", 24)
    codes_set_long(GRIB_MESSAGE_TEMPLATE, "generatingProcessIdentifier", 202)
    codes_set_long(GRIB_MESSAGE_TEMPLATE, "centre", 86)
    codes_set_long(GRIB_MESSAGE_TEMPLATE, "bitmapPresent", 1)

    base_lt = datetime.timedelta(hours=1)

    is_minutes = True if GRIB_MESSAGE_STEP == datetime.timedelta(minutes=15) else False

    if is_minutes:
        codes_set_long(GRIB_MESSAGE_TEMPLATE, "indicatorOfUnitForTimeRange", 0) # minute
        base_lt = datetime.timedelta(minutes=15)

    pdtn = codes_get_long(GRIB_MESSAGE_TEMPLATE, "productDefinitionTemplateNumber")

    if write_options is not None:
        for opt in write_options.split(','):
            k,v = opt.split('=')
            codes_set_long(GRIB_MESSAGE_TEMPLATE, k, int(v))

    for i in range(interpolated_data.shape[0]):

        lt = base_lt * i

        if pdtn == 8:
            lt -= base_lt
            assert(codes_get_long(GRIB_MESSAGE_TEMPLATE, "indicatorOfUnitForTimeRange") == 1) # hour
            lt_end = analysistime + datetime.timedelta(hours=codes_get_long(GRIB_MESSAGE_TEMPLATE, "lengthOfTimeRange"))

            # these are not mandatory but some software uses them
            codes_set_long(GRIB_MESSAGE_TEMPLATE, "yearOfEndOfOverallTimeInterval", int(lt_end.strftime("%Y")))
            codes_set_long(GRIB_MESSAGE_TEMPLATE, "monthOfEndOfOverallTimeInterval", int(lt_end.strftime("%m")))
            codes_set_long(GRIB_MESSAGE_TEMPLATE, "dayOfEndOfOverallTimeInterval", int(lt_end.strftime("%d")))
            codes_set_long(GRIB_MESSAGE_TEMPLATE, "hourOfEndOfOverallTimeInterval", int(lt_end.strftime("%H")))
            codes_set_long(GRIB_MESSAGE_TEMPLATE, "minuteOfEndOfOverallTimeInterval", int(lt_end.strftime("%M")))
            codes_set_long(GRIB_MESSAGE_TEMPLATE, "secondOfEndOfOverallTimeInterval", int(lt_end.strftime("%S")))

        if is_minutes:
            codes_set_long(GRIB_MESSAGE_TEMPLATE, "forecastTime", lt.total_seconds() / 60)
        else:
            codes_set_long(GRIB_MESSAGE_TEMPLATE, "forecastTime", lt.total_seconds() / 3600)

        codes_set_values(GRIB_MESSAGE_TEMPLATE, interpolated_data[i,:,:].flatten())
        codes_write(GRIB_MESSAGE_TEMPLATE, fpout)

    codes_release(GRIB_MESSAGE_TEMPLATE)
    fpout.close()


def write_grib(interpolated_data,write_grib_file,t_diff,write_options):
    # (Almost) all the metadata is copied from modeldata.grib2
    try:
        os.remove(write_grib_file)
    except OSError as e:
        pass

    fpout = None
    if write_grib_file.startswith("s3://"):
        openfile = fsspec.open('simplecache::{}'.format(write_grib_file),
                            'wb',
                            s3={'anon':False, 'key':os.environ['S3_ACCESS_KEY_ID'], 'secret':os.environ['S3_SECRET_ACCESS_KEY'],'client_kwargs':{'endpoint_url':'https://lake.fmi.fi'}})
        with openfile as fpout:
            write_grib_message(fpout,interpolated_data,t_diff,write_options)
    else:
        with open(str(write_grib_file), "wb") as fpout:
             write_grib_message(fpout,interpolated_data,t_diff,write_options)


def write_nc(interpolated_data,image_nc_file,write_nc_file,variable,predictability,t_diff):

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

