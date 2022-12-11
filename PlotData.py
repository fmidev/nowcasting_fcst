import matplotlib.pyplot as plt
import diagnostics_functions
from fileutils import read
import os

class PlotData:
    def __init__(self, options, image_array1, image_array2, weights_obs_extrap=None):
        self.options = options
        self.weights_obs_extrap = weights_obs_extrap
        self.image_array1 = image_array1
        self.image_array2 = image_array2
        self.interpolated_advection, self.quantity1_min, self.quantity1_max, \
        self.timestamp1, self.mask_nodata1, self.nodata1, self.longitudes, \
        self.latitudes = read(self.options.output_data, read_coordinates=False)
        self.plot_default()
    #Lisätään jokin vipu, jos haluaa saada muutkin kuvat ulos

    @staticmethod
    def generate_dirs(*args: str):
        cwd = os.getcwd()
        if not os.path.exists(f"{'/'.join(args)}"):
            os.makedirs(f"{'/'.join(args)}")
        return f"{cwd}/{'/'.join(args)}"

    @staticmethod
    def generate_file_name(path: str, name: str):
        return f"{path}/{''.join(name)}"

    def generate_date(self, i):
        return self.timestamp1[i].strftime("%Y-%m-%d : %H:%M"), self.timestamp1[i].strftime("%Y%m%d%H")

    def plot_default(self):
        # Create directories which do not yet exist
        """
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
        """
        outdir = self.generate_dirs('figures', 'fields')
        # Defining min and max in all data
        R_min = self.interpolated_advection.min()
        if (self.options.parameter == 'precipitation_1h_bg'):
            R_max = 3
        elif (self.options.parameter == '2t'):
            R_max = 280
        # plotting interpolated_advection fields
        else:
            R_max = self.interpolated_advection.max()
        # PLOTTING AND SAVING TO FILE
        title = self.options.parameter.replace("_", " ")
        datetitle, dateformat = self.generate_date(0)
        fig_name = self.generate_file_name(outdir, "".join(["fields_", title, "_", dateformat, "_h.png"]))
        diagnostics_functions.plot_contourf_map_scandinavia(self.options.output_data, R_min, R_max, fig_name, datetitle, title)
        #diagnostics_functions.plot_imshow_map_scandinavia(self.options.output_data, R_min, R_max, fig_name, datetitle, title)

    def plot_all(self):

        R_min = self.interpolated_advection.min()
        if (self.options.parameter == 'precipitation_1h_bg'):
            R_max = 3
            added_hours = 1
        elif (self.options.parameter == '2t'):
            R_max = 280
        # plotting interpolated_advection fields
        else:
            added_hours = 0
            R_max = self.interpolated_advection.max()

        # Plot unmodified obs data if it exists
        outdir = self.generate_dirs('figures', 'fields')
        if self.options.obs_data != None:
            image_array_obs_data, quantity_obs_data_min, quantity_obs_data_max, timestamp_obs_data, mask_nodata_obs_data, nodata_obs_data, longitudes_obs_data, latitudes_obs_data = read(
                self.options.obs_data)
            quantity_plot = self.options.parameter
            # Plotting LAPS field as it is in the uncombined file
            title = "OBS (LAPS/PPN) "
            outfile = outdir + "image_array_obs_data.png"
            #diagnostics_functions.plot_imshow_map_scandinavia(image_array_obs_data[0, :, :], 0, 1, outfile, "jet", title, self.longitudes, self.latitudes)

        # Plot unmodified background data if it exists
        if self.options.background_data != None:
            image_array_plot, quantity_plot_min, quantity_plot_max, timestamp_plot, mask_nodata_plot, nodata_plot, \
            longitudes_obs_data, latitudes_plot = read(self.options.background_data)
            quantity_plot = self.options.parameter
            title = "MNWC 0hours"
            outfile = outdir + "image_array_MNWC_0hours.png"
            #diagnostics_functions.plot_imshow_map_scandinavia(image_array_plot[0, :, :], 0, 1, outfile, "jet", title, self.longitudes, self.latitudes)
            if 'mask_nodata_obs_data' in locals():
                # Read radar composite field used as mask for Finnish data. Lat/lon info is not stored in radar composite HDF5 files, but these are the same! (see README.txt)
                weights_bg = read_background_data_and_make_mask(image_file=self.options.detectability_data,
                                                                input_mask=define_common_mask_for_fields(mask_nodata_obs_data),
                                                                mask_smaller_than_borders=4, smoother_coefficient=0.2,
                                                                gaussian_filter_coefficient=3)
                weights_obs = 1 - weights_bg
                title = "weights 0h"
                outfile = outdir + "image_array_weights_0h.png"
                #diagnostics_functions.plot_imshow_map_scandinavia(weights_bg, 0, 1, outfile, "jet", title, self.longitudes, self.latitudes)
                if 'weights_obs_extrap' in locals():
                    weights_obs_all = np.concatenate((weights_obs[np.newaxis, :], self.weights_obs_extrap), axis=0)
                    # fig, ax = plt.subplots(1,weights_obs_all.shape[0])
                    for im_no in np.arange(weights_obs_all.shape[0]):
                        title = f"weights {im_no}h"
                        outfile = outdir + f"image_array_weights_{im_no}h.png"
                        #diagnostics_functions.plot_imshow_map_scandinavia(weights_obs_all[im_no, :, :], 0, 1, outfile, "jet", title, self.longitudes, self.latitudes)

                        # ax[0,im_no].diagnostics_functions.plot_imshow_map_scandinavia(weights_obs_all[im_no,:,:],0,1,outfile,"jet",title,longitudes,latitudes)

        # Plot dynamic_nwc_data if it exists
        if self.options.dynamic_nwc_data != None:
            if (self.options.parameter == 'precipitation_1h_bg'):
                added_hours = 1
            else:
                added_hours = 0
            image_array_plot, quantity_plot_min, quantity_plot_max, timestamp_plot, mask_nodata_plot, nodata_plot, longitudes_plot, latitudes_plot = read(
                self.options.dynamic_nwc_data, added_hours)
            for n in (list(range(0, image_array_plot.shape[0]))):
                outdir = "figures/fields/"
                # PLOTTING AND SAVING TO FILE
                title = self.options.parameter + " MNWC fc=+" + str(n + added_hours) + "h"
                outfile = outdir + "field" + self.options.parameter + " MNWC_fc=+" + str(n) + "h.png"
                diagnostics_functions.plot_imshow(image_array_plot[n, :, :], R_min, R_max, outfile, "jet", title)

        # Plot extrapolated_data if it exists
        if self.options.extrapolated_data != None:
            if (self.options.parameter == 'precipitation_1h_bg'):
                added_hours = 1
            else:
                added_hours = 0
            image_array_plot, quantity_plot_min, quantity_plot_max, timestamp_plot, mask_nodata_plot, nodata_plot, longitudes_plot, latitudes_plot = read(self.options.extrapolated_data)
            for n in (list(range(0, image_array_plot.shape[0]))):
                outdir = "figures/fields/"
                # PLOTTING AND SAVING TO FILE
                title = self.options.parameter + " extrapolated_data fc=+" + str(n + added_hours) + "h"
                outfile = outdir + "field" + self.options.parameter + " extrapolated_data_fc=+" + str(n) + "h.png"
                diagnostics_functions.plot_imshow(image_array_plot[n, :, :], R_min, R_max, outfile, "jet", title)

        # NOW PLOT DIAGNOSTICS FROM THE FIELDS

        # TIME SERIES FROM THE FIELD MEAN
        fc_lengths = np.arange(0, self.interpolated_advection.shape[0])
        outdir = "figures/"
        outfile = outdir + "Field_mean_" + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + ".png"
        plt.plot(fc_lengths, np.mean(self.interpolated_advection, axis=(1, 2)), linewidth=2.0, label="temperature")
        title = "Field mean, " + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S")
        plt.title(title)
        plt.tight_layout(pad=0.)
        # plt.xticks([])
        # plt.yticks([])
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
        plt.close()

        # JUMPINESS CHECKS
        # PREVAILING ASSUMPTION: THE CHANGE IN INDIVIDUAL GRIDPOINTS IS VERY LINEAR

        # RESULT ARRAYS
        linear_change_boolean = np.ones(self.interpolated_advection.shape)
        gp_abs_difference = np.ones(self.interpolated_advection.shape)
        gp_mean_difference = np.ones(self.interpolated_advection.shape)
        ratio_meandiff_absdiff = np.ones(self.interpolated_advection.shape)

        # 0) PLOT image_array1 FIELDS
        for n in (list(range(0, self.image_array1.shape[0]))):
            outdir = "figures/fields/"
            # PLOTTING AND SAVING TO FILE
            title = self.options.parameter + " " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "nwc_data=+" + str(n) + "h"
            outfile = outdir + "field" + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "_nwc_data=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(self.image_array1[n, :, :], R_min, R_max, outfile, "jet", title)

        # 0a) PLOT image_array2 FIELDS
        if self.options.model_data != None:
            image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2, nodata2, longitudes2, latitudes2 = read(self.options.model_data, added_hours=0)
            quantity2 = self.options.parameter
            # nodata values are always taken from the model field. Presumably these are the same.
            nodata = nodata2
            for n in (list(range(0, image_array2.shape[0]))):
                outdir = "figures/fields/"
                # PLOTTING AND SAVING TO FILE
                title = self.options.parameter + " " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "fc_model=+" + str(n) + "h"
                outfile = outdir + "field" + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc_model=+" + str(n) + "h.png"
                diagnostics_functions.plot_imshow(image_array2[n, :, :], R_min, R_max, outfile, "jet", title)

        # 0b) PLOT DMO FIELDS
        for n in (list(range(0, self.interpolated_advection.shape[0]))):
            outdir = "figures/fields/"
            # PLOTTING AND SAVING TO FILE
            title = self.options.parameter + " " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "fc=+" + str(n) + "h"
            outfile = outdir + "field" + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(self.interpolated_advection[n, :, :], R_min, R_max, outfile, "jet", title)

        # 0d) PLOT DMO FIELD MINUS MODEL
        for n in (list(range(0, self.interpolated_advection.shape[0]))):
            outdir = "figures/fields/"
            # PLOTTING AND SAVING TO FILE
            title = self.options.parameter + " " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "fc_minus_model=+" + str(n) + "h"
            outfile = outdir + "field" + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc_minus_model=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(self.interpolated_advection[n, :, :] - self.image_array2[n, :, :], -1, 1, outfile, "jet", title)

        # 1a) TRUE IS ASSIGNED TO THOSE TIME+LOCATION POINTS THAT HAVE PREVIOUS TIMESTEP VALUE LESS (MORE) AND NEXT TIMESTEP VALUE MORE (LESS) THAN THE CORRESPONDING VALUE. SO NO CHANGE IN THE SIGN OF THE DERIVATIVE.
        for n in (list(range(1, 5))):
            outdir = "figures/linear_change/"
            gp_increased = (self.interpolated_advection[n, :, :] - self.interpolated_advection[(n - 1), :, :] > 0) & (self.interpolated_advection[(n + 1), :, :] - self.interpolated_advection[n, :, :] > 0)
            gp_reduced = (self.interpolated_advection[n, :, :] - self.interpolated_advection[(n - 1), :, :] < 0) & (self.interpolated_advection[(n + 1), :, :] - self.interpolated_advection[n, :, :] < 0)
            linear_change_boolean[n, :, :] = gp_reduced + gp_increased
            gp_outside_minmax_range = np.max(np.maximum(np.maximum(0, (self.interpolated_advection[n, :, :] - np.maximum(self.interpolated_advection[(n - 1), :, :],
                                    self.interpolated_advection[(n + 1), :, :]))),
                                    abs(np.minimum(0,(self.interpolated_advection[n,:,:] - np.minimum(self.interpolated_advection[(n - 1),:,:],self.interpolated_advection[( n + 1),:, :]))))))
            # PLOTTING AND SAVING TO FILE
            title = self.options.parameter + " " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n) + "h, " + str(round(np.mean(linear_change_boolean[n, :, :]) * 100, 2)) + "% gridpoints is \n inside range " + str(
                n - 1) + "h..." + str(n + 1) + "h, outside range field max value " + str(round(gp_outside_minmax_range, 2))
            outfile = outdir + "linear_change_" + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(linear_change_boolean[n, :, :], 0, 1, outfile, "jet", title)

        # 1b) TRUE IS ASSIGNED TO THOSE TIME+LOCATION POINTS THAT HAVE PREVIOUS TIMESTEP VALUE LESS (MORE) AND NEXT+1 TIMESTEP VALUE MORE (LESS) THAN THE CORRESPONDING VALUE. SO NO CHANGE IN THE SIGN OF THE DERIVATIVE.
        for n in (list(range(1, 4))):
            outdir = "figures/linear_change3h/"
            gp_increased = ((self.interpolated_advection[n, :, :] - self.interpolated_advection[(n - 1), :, :]) > 0) & ((self.interpolated_advection[(n + 2), :, :] - self.interpolated_advection[n, :, :]) > 0)
            gp_reduced = ((self.interpolated_advection[n, :, :] - self.interpolated_advection[(n - 1), :, :]) < 0) & ((self.interpolated_advection[(n + 2), :, :] - self.interpolated_advection[n, :, :]) < 0)
            linear_change_boolean[n, :, :] = gp_reduced + gp_increased
            gp_outside_minmax_range = np.max(np.maximum(np.maximum(0, (self.interpolated_advection[n, :, :] - np.maximum(self.interpolated_advection[(n - 1), :, :],
                                        self.interpolated_advection[(n + 2), :, :]))),
                                        abs(np.minimum(0, (self.interpolated_advection[n, :,:] - np.minimum(self.interpolated_advection[(n - 1),:,:],
                                                                                                       self.interpolated_advection[( n + 2),:,:]))))))
            # PLOTTING AND SAVING TO FILE
            title = self.options.parameter + " " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n) + "h, " + str(
                round(np.mean(linear_change_boolean[n, :, :]) * 100, 2)) + "% gridpoints is \n inside range " + str(
                n - 1) + "h..." + str(n + 2) + "h, outside range field max value " + str(round(gp_outside_minmax_range, 2))
            outfile = outdir + "linear_change_" + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(linear_change_boolean[n, :, :], 0, 1, outfile, "jet", title)

        # 1c) TRUE IS ASSIGNED TO THOSE TIME+LOCATION POINTS THAT HAVE PREVIOUS TIMESTEP VALUE LESS (MORE) AND NEXT+2 TIMESTEP VALUE MORE (LESS) THAN THE CORRESPONDING VALUE. SO NO CHANGE IN THE SIGN OF THE DERIVATIVE.
        for n in (list(range(1, 3))):
            outdir = "figures/linear_change4h/"
            gp_increased = (self.interpolated_advection[n, :, :] - self.interpolated_advection[(n - 1), :, :] > 0) & (
                        self.interpolated_advection[(n + 3), :, :] - self.interpolated_advection[n, :, :] > 0)
            gp_reduced = (self.interpolated_advection[n, :, :] - self.interpolated_advection[(n - 1), :, :] < 0) & (
                        self.interpolated_advection[(n + 3), :, :] - self.interpolated_advection[n, :, :] < 0)
            linear_change_boolean[n, :, :] = gp_reduced + gp_increased
            gp_outside_minmax_range = np.max(np.maximum(np.maximum(0, (self.interpolated_advection[n, :, :] - np.maximum(self.interpolated_advection[(n - 1), :, :],
                                    self.interpolated_advection[(n + 3), :, :]))),
                                    abs(np.minimum(0,( self.interpolated_advection[n, :,:] - np.minimum(self.interpolated_advection[( n - 1), :,:],self.interpolated_advection[( n + 3),:,:]))))))
            # PLOTTING AND SAVING TO FILE
            title = self.options.parameter + " " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n) + "h, " + str(
                round(np.mean(linear_change_boolean[n, :, :]) * 100, 2)) + "% gridpoints is \n inside range " + str(
                n - 1) + "h..." + str(n + 3) + "h, outside range field max value " + str(round(gp_outside_minmax_range, 2))
            outfile = outdir + "linear_change_" + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(linear_change_boolean[n, :, :], 0, 1, outfile, "jet", title)

        # ANALYSING "JUMPINESS"
        for n in (list(range(1, 5))):
            # 2) DIFFERENCE OF TIMESTEPS (n-1) AND (n+1)
            gp_abs_difference[n, :, :] = abs(self.interpolated_advection[(n + 1), :, :] - self.interpolated_advection[(n - 1), :, :])
            outdir = "figures/jumpiness_absdiff/"
            title = self.options.parameter + " absdiff of " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(
                n - 1) + "h" + str(n + 1) + "h \n field max value " + str(round(np.max(gp_abs_difference[n, :, :]), 2))
            outfile = outdir + "jumpiness_absdiff_" + self.options.parameter + self.timestamp1[0].strftime(
                "%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(gp_abs_difference[n, :, :], -5, 5, outfile, "seismic", title)
            # 3) DIFFERENCE FROM THE MEAN OF (n-1) AND (n+1)
            gp_mean_difference[n, :, :] = abs(self.interpolated_advection[n, :, :] - (
                        (self.interpolated_advection[(n + 1), :, :] + self.interpolated_advection[(n - 1), :, :]) / 2))
            outdir = "figures/jumpiness_meandiff/"
            title = self.options.parameter + " diff from " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(
                n - 1) + "h" + str(n + 1) + "h_mean \n field max value " + str(round(np.max(gp_mean_difference[n, :, :]), 2))
            outfile = outdir + "jumpiness_meandiff_" + self.options.parameter + self.timestamp1[0].strftime(
                "%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(gp_mean_difference[n, :, :], -0.2, 0.2, outfile, "seismic", title)
            # 4) MEAN DIFF DIVIDED BY ABSDIFF
            ratio_meandiff_absdiff = gp_mean_difference / gp_abs_difference
            outdir = "figures/jumpiness_ratio/"
            title = self.options.parameter + " meandiff / absdiff ratio % " + self.timestamp1[0].strftime("%Y%m%d%H%M%S") \
                    + " fc_" + str(n - 1) + "h" + str(n + 1) + "h \n field max value " \
                    + str(round(np.max(ratio_meandiff_absdiff[n, :, :]), 2))
            outfile = outdir + "jumpiness_ratio_" + self.options.parameter + self.timestamp1[0].strftime("%Y%m%d%H%M%S") \
                      + "_fc=+" + str(n) + "h.png"
            diagnostics_functions.plot_imshow(ratio_meandiff_absdiff[n, :, :], 0, 2, outfile, "seismic", title)

