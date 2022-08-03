import numpy as np
import xarray as xr
import pygrib as pg
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy
import metview as mv



########### DEBUGGING FUNCTIONS #############

def plot_imshow(temps,vmin,vmax,outfile,cmap,title):
    plt.imshow(temps,cmap=cmap,vmin=vmin,vmax=vmax,origin="lower")
    #plt.axis('off')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout(pad=0.)
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig(outfile,bbox_inches='tight', pad_inches=0)
    plt.close()



    
def plot_imshow_on_map(temps,vmin,vmax,outfile,cmap,title,longitudes,latitudes):

    # For those longitudes that are over 180 degrees, reduce 360 degrees from them
    longitudes[longitudes > 180] = longitudes[longitudes > 180] - 360

    grid_lon, grid_lat = [longitudes, latitudes]
    
    proj = cartopy.crs.LambertConformal(central_latitude = int(np.mean(latitudes)), 
                             central_longitude = int(np.mean(longitudes)), 
                             standard_parallels = (25, 25))
    #proj = cartopy.crs.PlateCarree()
    #ncolors = 256
    
    temps[np.where(temps<=0.1)] = None
    ncolors = 128
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors))
    # change alpha values
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
    # create a colormap object
    # register this new colormap with matplotlib
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
    plt.register_cmap(cmap=map_object)
    ax = plt.axes(projection = proj)
    # x, y = ax(*[grid_lon, grid_lat])
    x, y = [grid_lon, grid_lat]
    cm = ax.pcolormesh(x, y, temps, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap, transform=cartopy.crs.PlateCarree())
    ax.coastlines('50m')
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.RIVERS)
    ax.gridlines()
    ax.set_title(title)
    plt.colorbar(cm, fraction = 0.046, pad = 0.04, orientation="horizontal")
    # plt.show()

    # # Transform longitudes defined in 0...360 to -180...180
    # longitudes[longitudes>180] = (180-longitudes[(longitudes>180)]%180)
    # grid_lon, grid_lat = [longitudes, latitudes]
    # m = Basemap(projection='cyl', llcrnrlon=longitudes.min(),urcrnrlon=longitudes.max(),llcrnrlat=latitudes.min(),urcrnrlat=latitudes.max(),resolution='c')
    # x, y = m(*[grid_lon, grid_lat])
    # m.pcolormesh(x,y,temps,shading='flat',cmap=cmap,vmin=vmin,vmax=vmax)
    # m.colorbar(location='right')
    # m.drawcoastlines()
    # m.drawmapboundary()
    # m.drawparallels(np.arange(-90.,120.,10.),labels=[1,0,0,0])
    # m.drawmeridians(np.arange(-180.,180.,10.),labels=[0,0,0,1])
    
    # plt.title(title)
    # plt.tight_layout(pad=0.)
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig(outfile,bbox_inches='tight', pad_inches=0, dpi=1200)
    plt.close()


def plot_contourf_map_scandinavia_1(grib_file, vmin, vmax, outfile, date, title):
    ds = mv.read(grib_file)
    #ds = mv.read(data=ds, param='thpb')
    #ds = mv.Fieldset(path='/home/korpinen/Downloads/_mars-webmars-private-svc-blue-008-6fe5cac1a363ec1525f54343b6cc9fd8-51dOBi.grib')
    #lats = mv.nx(ds)  # returns a numPy array
    #lons = mv.ny(ds)  # returns a numPy array
    vals = mv.values(ds)
    print(mv.grib_get(ds, ['shortName', 'level', 'paramId']))

    #tstm = ds.select
    tstm = ds["tstm"]
    #print(mv.minvalue(tstm), mv.maxvalue(tstm))

    #tstm = mv.read(data=ds, param="3060")
    u = mv.read(data=ds, param="10u")
    v = mv.read(data=ds, param="10v")
    #spd = mv.sqrt(u * u + v * v)
    my_view = mv.geoview(map_area_definition="CORNERS", map_projection="polar_stereographic")
    # set up the coastlines
    my_coast = mv.mcoast(map_coastline_land_shade="ON", map_coastline_land_shade_colour="CREAM")
    cont = mv.mcont(legend="on",
                    contour="off",
                    contour_level_selection_type="level_list",
                    contour_level_list=[0, 0.2, 0.5, 0.8, 1],
                    contour_label="off",
                    contour_shade="on",
                    contour_shade_colour_method="gradients",
                    contour_shade_method="area_fill",
                    contour_gradients_colour_list=["RGB(0.1532,0.1187,0.5323)",
                                                    "RGB(0.5067,0.7512,0.8188)",
                                                    "RGB(0.9312,0.9313,0.9275)",
                                                    "RGB(0.9523,0.7811,0.3104)",
                                                    "RGB(0.594,0.104,0.104)",],
                    contour_gradients_step_list=20)
    # define output
    mv.setoutput(mv.pdf_output(output_name=outfile))
    view = mv.geoview(map_area_definition="corners",
                      area=[25, -60, 75, 60])
    mv.plot(view, tstm[0], mv.mcont(contour_automatic_setting='ecmwf', legend='on'))

    # plt.savefig(outfile, pad_inches=0, dpi=1200)

def plot_contourf_map_scandinavia(grib_file, vmin, vmax, outfile, date, title):
    #ds = xr.open_dataset(grib_file)
    ds = pg.open(grib_file)
    ds.seek(0)
    for v in ds:
        print(v)
        selected_grb = ds.select(name='Thunderstorm probability')[0]
        data, lats, lons = selected_grb.data()
        #lons[lons > 180] = lons[lons > 180] - 360

        grid_lon, grid_lat = [lons, lats]

        proj = cartopy.crs.LambertConformal()

        ax = plt.axes(projection=proj)
        ax.set_extent([5, 35, 52, 72])
        #ax.gridlines()
        x, y = [grid_lon, grid_lat]
        cm = ax.pcolormesh(y, x, data, transform=cartopy.crs.PlateCarree())  # , shading='auto', vmin=vmin, vmax=vmax, cmap='OrRd')
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS)
        plt.title(f"Propability of Thunder, {date}")
        plt.colorbar(cm, fraction=0.046, pad=0.04, orientation="horizontal")
        plt.show()

        """
        data = ds[v].data
        print(ds.keys())
        t = ds[v].plot()
        proj = cartopy.crs.PlateCarree()
        ax = plt.axes(projection=proj)
        ax.set_extent([5, 35, 52, 72])
        #ax.add_feature(cartopy.feature.COASTLINE)
        #ax.add_feature(cartopy.feature.BORDERS)
        #ax.add_feature(cartopy.feature.OCEAN)
        #ax.add_feature(cartopy.feature.LAND)
        # ax.add_feature(cartopy.feature.LAKES)
        # ax.add_feature(cartopy.feature.RIVERS)
        ax.gridlines()
        x, y = [grid_lon, grid_lat]
        for i in range(len(data)):
            d = data[i]
            #d[d <= 0.00001] = np.nan
            cm = ax.pcolormesh(d, transform=cartopy.crs.NorthPolarStereo())# , shading='auto', vmin=vmin, vmax=vmax, cmap='OrRd')
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.add_feature(cartopy.feature.BORDERS)
            plt.title(f"Propability of Thunder, {date}")
            plt.colorbar(cm, fraction=0.046, pad=0.04, orientation="horizontal")
            plt.show()
            #plt.savefig(outfile, pad_inches=0, dpi=1200)
    """
    
def plot_only_colorbar(vmin,vmax,units,outfile,cmap):
    fig = plt.figure(figsize=(8, 1))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = matplotlib.colorbar.ColorbarBase(ax1,cmap=cmap, norm=norm,orientation='horizontal')
    cb1.set_label(units)
    plt.savefig(outfile,bbox_inches='tight')
    plt.close()


def plot_verif_scores(fc_lengths,verif_scores,labels,outfile,title,y_ax_title):
    for n in range(0, verif_scores.shape[0]):
        plt.plot(fc_lengths, verif_scores[n,:], linewidth=2.0, label=str(labels[n]))
    plt.legend(bbox_to_anchor=(0.21, 1))
    plt.title(title)
    plt.xlabel('Forecast length (h)')
    plt.ylabel(y_ax_title)
    plt.savefig(outfile,bbox_inches='tight', pad_inches=0)
    plt.close()

