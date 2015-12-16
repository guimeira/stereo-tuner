#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <gtk/gtk.h>
#include <cstdio>
#include <cstring>
#include <ctime>

using namespace std;
using namespace cv;

/* Matcher type */
typedef enum {
	BM, SGBM
} MatcherType;

/* Main data structure definition */
struct ChData {
	/* Widgets */
	GtkWidget *main_window; /* Main application window */
	GtkImage *image_left;
	GtkImage *image_right;
	GtkImage *image_depth;
	GtkWidget *rb_bm, *rb_sgbm;
	GtkWidget *sc_block_size, *sc_min_disparity, *sc_num_disparities,
		*sc_disp_max_diff, *sc_speckle_range, *sc_speckle_window_size,
		*sc_p1, *sc_p2, *sc_pre_filter_cap, *sc_pre_filter_size,
		*sc_uniqueness_ratio, *sc_texture_threshold,
		*rb_pre_filter_normalized, *rb_pre_filter_xsobel, *chk_full_dp;
	GtkAdjustment *adj_block_size, *adj_min_disparity, *adj_num_disparities,
	*adj_disp_max_diff, *adj_speckle_range, *adj_speckle_window_size,
	*adj_p1, *adj_p2, *adj_pre_filter_cap, *adj_pre_filter_size,
	*adj_uniqueness_ratio, *adj_texture_threshold;
	GtkWidget *status_bar;
	gint status_bar_context;

	/* OpenCV */
	Ptr<StereoMatcher> stereo_matcher;
	Mat cv_image_left, cv_image_right, cv_image_disparity,
			cv_image_disparity_normalized, cv_color_image;
	MatcherType matcher_type;
	int block_size;
	int disp_12_max_diff;
	int min_disparity;
	int num_disparities;
	int speckle_range;
	int speckle_window_size;
	int pre_filter_cap;
	int pre_filter_size;
	int pre_filter_type;
	int texture_threshold;
	int uniqueness_ratio;
	int p1;
	int p2;
	int mode;

	/* Defalt values */
	static const int DEFAULT_BLOCK_SIZE = 5;
	static const int DEFAULT_DISP_12_MAX_DIFF = -1;
	static const int DEFAULT_MIN_DISPARITY = 0;
	static const int DEFAULT_NUM_DISPARITIES = 64;
	static const int DEFAULT_SPECKLE_RANGE = 0;
	static const int DEFAULT_SPECKLE_WINDOW_SIZE = 0;
	static const int DEFAULT_PRE_FILTER_CAP = 1;
	static const int DEFAULT_PRE_FILTER_SIZE = 5;
	static const int DEFAULT_PRE_FILTER_TYPE =
			StereoBM::PREFILTER_NORMALIZED_RESPONSE;
	static const int DEFAULT_TEXTURE_THRESHOLD = 0;
	static const int DEFAULT_UNIQUENESS_RATIO = 0;
	static const int DEFAULT_P1 = 0;
	static const int DEFAULT_P2 = 0;
	static const int DEFAULT_MODE = StereoSGBM::MODE_SGBM;

	ChData() : matcher_type(BM), block_size(DEFAULT_BLOCK_SIZE), min_disparity(DEFAULT_MIN_DISPARITY),
			num_disparities(DEFAULT_NUM_DISPARITIES), speckle_range(DEFAULT_SPECKLE_RANGE),
			speckle_window_size(DEFAULT_SPECKLE_WINDOW_SIZE), pre_filter_cap(DEFAULT_PRE_FILTER_CAP),
			pre_filter_size(DEFAULT_PRE_FILTER_SIZE), pre_filter_type(DEFAULT_PRE_FILTER_TYPE),
			texture_threshold(DEFAULT_TEXTURE_THRESHOLD),
			uniqueness_ratio(DEFAULT_UNIQUENESS_RATIO), p1(DEFAULT_P1), p2(DEFAULT_P2),
			mode(DEFAULT_MODE)
		{}
};

void update_matcher(ChData *data) {
	Ptr<StereoBM> stereo_bm;
	Ptr<StereoSGBM> stereo_sgbm;

	switch (data->matcher_type) {
	case BM:
		stereo_bm = data->stereo_matcher.dynamicCast<StereoBM>();

		//If we have the wrong type of matcher, let's create a new one:
		if (!stereo_bm) {
			data->stereo_matcher = stereo_bm = StereoBM::create(16, 1);

			gtk_widget_set_sensitive(data->sc_block_size, true);
			gtk_widget_set_sensitive(data->sc_min_disparity, true);
			gtk_widget_set_sensitive(data->sc_num_disparities, true);
			gtk_widget_set_sensitive(data->sc_disp_max_diff, true);
			gtk_widget_set_sensitive(data->sc_speckle_range, true);
			gtk_widget_set_sensitive(data->sc_speckle_window_size, true);
			gtk_widget_set_sensitive(data->sc_p1, false);
			gtk_widget_set_sensitive(data->sc_p2, false);
			gtk_widget_set_sensitive(data->sc_pre_filter_cap, true);
			gtk_widget_set_sensitive(data->sc_pre_filter_size, true);
			gtk_widget_set_sensitive(data->sc_uniqueness_ratio, true);
			gtk_widget_set_sensitive(data->sc_texture_threshold, true);
			gtk_widget_set_sensitive(data->rb_pre_filter_normalized, true);
			gtk_widget_set_sensitive(data->rb_pre_filter_xsobel, true);
			gtk_widget_set_sensitive(data->chk_full_dp, false);
		}

		stereo_bm->setBlockSize(data->block_size);
		stereo_bm->setDisp12MaxDiff(data->disp_12_max_diff);
		stereo_bm->setMinDisparity(data->min_disparity);
		stereo_bm->setNumDisparities(data->num_disparities);
		stereo_bm->setSpeckleRange(data->speckle_range);
		stereo_bm->setSpeckleWindowSize(data->speckle_window_size);
		stereo_bm->setPreFilterCap(data->pre_filter_cap);
		stereo_bm->setPreFilterSize(data->pre_filter_size);
		stereo_bm->setPreFilterType(data->pre_filter_type);
		stereo_bm->setTextureThreshold(data->texture_threshold);
		stereo_bm->setUniquenessRatio(data->uniqueness_ratio);
		break;

	case SGBM:
		stereo_sgbm = data->stereo_matcher.dynamicCast<StereoSGBM>();

		//If we have the wrong type of matcher, let's create a new one:
		if (!stereo_sgbm) {
			data->stereo_matcher = stereo_sgbm = StereoSGBM::create(
					ChData::DEFAULT_MIN_DISPARITY,
					ChData::DEFAULT_NUM_DISPARITIES, ChData::DEFAULT_BLOCK_SIZE,
					ChData::DEFAULT_P1, ChData::DEFAULT_P2,
					ChData::DEFAULT_DISP_12_MAX_DIFF,
					ChData::DEFAULT_PRE_FILTER_CAP,
					ChData::DEFAULT_UNIQUENESS_RATIO,
					ChData::DEFAULT_SPECKLE_WINDOW_SIZE,
					ChData::DEFAULT_SPECKLE_RANGE, ChData::DEFAULT_MODE);

			gtk_widget_set_sensitive(data->sc_block_size, true);
			gtk_widget_set_sensitive(data->sc_min_disparity, true);
			gtk_widget_set_sensitive(data->sc_num_disparities, true);
			gtk_widget_set_sensitive(data->sc_disp_max_diff, true);
			gtk_widget_set_sensitive(data->sc_speckle_range, true);
			gtk_widget_set_sensitive(data->sc_speckle_window_size, true);
			gtk_widget_set_sensitive(data->sc_p1, true);
			gtk_widget_set_sensitive(data->sc_p2, true);
			gtk_widget_set_sensitive(data->sc_pre_filter_cap, true);
			gtk_widget_set_sensitive(data->sc_pre_filter_size, false);
			gtk_widget_set_sensitive(data->sc_uniqueness_ratio, true);
			gtk_widget_set_sensitive(data->sc_texture_threshold, false);
			gtk_widget_set_sensitive(data->rb_pre_filter_normalized, false);
			gtk_widget_set_sensitive(data->rb_pre_filter_xsobel, false);
			gtk_widget_set_sensitive(data->chk_full_dp, true);
		}

		stereo_sgbm->setBlockSize(data->block_size);
		stereo_sgbm->setDisp12MaxDiff(data->disp_12_max_diff);
		stereo_sgbm->setMinDisparity(data->min_disparity);
		stereo_sgbm->setMode(data->mode);
		stereo_sgbm->setNumDisparities(data->num_disparities);
		stereo_sgbm->setP1(data->p1);
		stereo_sgbm->setP2(data->p2);
		stereo_sgbm->setPreFilterCap(data->pre_filter_cap);
		stereo_sgbm->setSpeckleRange(data->speckle_range);
		stereo_sgbm->setSpeckleWindowSize(data->speckle_window_size);
		stereo_sgbm->setUniquenessRatio(data->uniqueness_ratio);

		break;
	}

	clock_t t;
	t = clock();
	data->stereo_matcher->compute(data->cv_image_left, data->cv_image_right,
			data->cv_image_disparity);
	t = clock() - t;

	gchar *status_message = g_strdup_printf("Disparity computation took %lf milliseconds",((double)t*1000)/CLOCKS_PER_SEC);
	gtk_statusbar_pop(GTK_STATUSBAR(data->status_bar), data->status_bar_context);
	gtk_statusbar_push(GTK_STATUSBAR(data->status_bar), data->status_bar_context, status_message);
	g_free(status_message);

	normalize(data->cv_image_disparity, data->cv_image_disparity_normalized, 0,
			255, CV_MINMAX, CV_8UC1);
	cvtColor(data->cv_image_disparity_normalized, data->cv_color_image,
			CV_GRAY2RGB);
	GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
			(guchar*) data->cv_color_image.data, GDK_COLORSPACE_RGB, false,
			8, data->cv_color_image.cols,
			data->cv_color_image.rows, data->cv_color_image.step,
			NULL, NULL);
	gtk_image_set_from_pixbuf(data->image_depth, pixbuf);
}

void update_interface(ChData *data) {
	if(data->matcher_type == BM) {
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->rb_bm),true);
	} else {
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->rb_sgbm),true);
	}

	gtk_adjustment_set_value(data->adj_block_size,data->block_size);
	gtk_adjustment_set_value(data->adj_min_disparity,data->min_disparity);
	gtk_adjustment_set_value(data->adj_num_disparities,data->num_disparities);
	gtk_adjustment_set_value(data->adj_disp_max_diff,data->disp_12_max_diff);
	gtk_adjustment_set_value(data->adj_speckle_range,data->speckle_range);
	gtk_adjustment_set_value(data->adj_speckle_window_size,data->speckle_window_size);
	gtk_adjustment_set_value(data->adj_p1,data->p1);
	gtk_adjustment_set_value(data->adj_p2,data->p2);
	gtk_adjustment_set_value(data->adj_pre_filter_cap,data->pre_filter_cap);
	gtk_adjustment_set_value(data->adj_pre_filter_size,data->pre_filter_size);
	gtk_adjustment_set_value(data->adj_uniqueness_ratio,data->uniqueness_ratio);
	gtk_adjustment_set_value(data->adj_texture_threshold,data->texture_threshold);
	gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->chk_full_dp),data->mode == StereoSGBM::MODE_HH);

	if(data->pre_filter_type == StereoBM::PREFILTER_NORMALIZED_RESPONSE) {
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->rb_pre_filter_normalized),true);
	} else {
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->rb_pre_filter_xsobel),true);
	}
}

extern "C" {
G_MODULE_EXPORT void on_adj_block_size_value_changed(GtkAdjustment *adjustment,
		ChData *data) {
	gint value;

	if (data == NULL) {
		fprintf(stderr, "WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value(adjustment);

	//the value must be odd, if it is not then set it to the next odd value
	if (value % 2 == 0) {
		value += 1;
		gtk_adjustment_set_value(adjustment, (gdouble) value);
		return;
	}

	//the value must be smaller than the image size
	if (value >= data->cv_image_left.cols
			|| value >= data->cv_image_left.rows) {
		fprintf(stderr, "WARNING: Block size is larger than image size\n");
		return;
	}

	//set the parameter,
	data->block_size = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_min_disparity_value_changed(
		GtkAdjustment *adjustment, ChData *data) {
	gint value;

	if (data == NULL) {
		fprintf(stderr, "WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value(adjustment);

	data->min_disparity = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_num_disparities_value_changed( GtkAdjustment *adjustment, ChData *data ) {
	gint value;

	if (data == NULL) {
		fprintf(stderr,"WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value( adjustment );

	//te value must be divisible by 16, if it is not set it to the nearest multiple of 16
	if (value % 16 != 0)
	{
		value += (16 - value%16);
		gtk_adjustment_set_value( adjustment, (gdouble)value);
		return;
	}

	data->num_disparities = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_disp_max_diff_value_changed(
		GtkAdjustment *adjustment, ChData *data) {
	gint value;

	if (data == NULL) {
		fprintf(stderr, "WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value(adjustment);

	data->disp_12_max_diff = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_speckle_range_value_changed( GtkAdjustment *adjustment, ChData *data ) {
	gint value;

	if (data == NULL) {
		fprintf(stderr,"WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value( adjustment );

	data->speckle_range = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_speckle_window_size_value_changed( GtkAdjustment *adjustment, ChData *data ) {
	gint value;

	if (data == NULL) {
		fprintf(stderr,"WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value( adjustment );

	data->speckle_window_size = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_p1_value_changed( GtkAdjustment *adjustment, ChData *data ) {
	gint value;

	if (data == NULL) {
		fprintf(stderr,"WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value( adjustment );

	data->p1 = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_p2_value_changed( GtkAdjustment *adjustment, ChData *data ) {
	gint value;

	if (data == NULL) {
		fprintf(stderr,"WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value( adjustment );

	data->p2 = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_pre_filter_cap_value_changed( GtkAdjustment *adjustment, ChData *data ) {
	gint value;

	if (data == NULL) {
		fprintf(stderr,"WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value ( adjustment );

	//set the parameter
	data->pre_filter_cap = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_pre_filter_size_value_changed( GtkAdjustment *adjustment, ChData *data ) {
	gint value;

	if (data == NULL) {
		fprintf(stderr,"WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value( adjustment );

	//the value must be odd, if it is not then set it to the next odd value
	if (value % 2 == 0)
	{
		value += 1;
		gtk_adjustment_set_value( adjustment, (gdouble)value);
		return;
	}

	//set the parameter,
	data->pre_filter_size = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_uniqueness_ratio_value_changed( GtkAdjustment *adjustment, ChData *data ) {
	gint value;

	if (data == NULL) {
		fprintf(stderr,"WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value( adjustment );

	data->uniqueness_ratio = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_adj_texture_threshold_value_changed( GtkAdjustment *adjustment, ChData *data ) {
	gint value;

	if (data == NULL) {
		fprintf(stderr,"WARNING: data is null\n");
		return;
	}

	value = (gint) gtk_adjustment_get_value( adjustment );

	data->texture_threshold = value;
	update_matcher(data);
}

G_MODULE_EXPORT void on_algo_ssgbm_clicked(GtkButton *b, ChData *data) {
	if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
		data->matcher_type = SGBM;
		update_matcher(data);
	}
}

G_MODULE_EXPORT void on_algo_sbm_clicked(GtkButton *b, ChData *data) {
	if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
		data->matcher_type = BM;
		update_matcher(data);
	}
}

G_MODULE_EXPORT void on_rb_pre_filter_normalized_clicked(GtkButton *b, ChData *data) {
	if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
		data->pre_filter_type = StereoBM::PREFILTER_NORMALIZED_RESPONSE;
		update_matcher(data);
	}
}

G_MODULE_EXPORT void on_rb_pre_filter_xsobel_clicked(GtkButton *b, ChData *data) {
	if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
		data->pre_filter_type = StereoBM::PREFILTER_XSOBEL;
		update_matcher(data);
	}
}

G_MODULE_EXPORT void on_chk_full_dp_clicked(GtkButton *b, ChData *data) {
	if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
		data->mode = StereoSGBM::MODE_HH;
	} else {
		data->mode = StereoSGBM::MODE_SGBM;
	}
	update_matcher(data);
}

G_MODULE_EXPORT void on_btn_save_clicked(GtkButton *b, ChData *data) {
	GtkWidget *dialog;
	GtkFileChooser *chooser;
	GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_SAVE;
	gint res;

	dialog = gtk_file_chooser_dialog_new("Save File", GTK_WINDOW(data->main_window), action, "Cancel", GTK_RESPONSE_CANCEL, "Save", GTK_RESPONSE_ACCEPT, NULL);
	chooser = GTK_FILE_CHOOSER(dialog);
	gtk_file_chooser_set_do_overwrite_confirmation(chooser, TRUE);
	gtk_file_chooser_set_current_name(chooser, "params.yml");

	GtkFileFilter *filter_yml = gtk_file_filter_new();
	gtk_file_filter_set_name(filter_yml,"YAML file (*.yml)");
	gtk_file_filter_add_pattern(filter_yml,"*.yml");

	GtkFileFilter *filter_xml = gtk_file_filter_new();
	gtk_file_filter_set_name(filter_xml, "XML file(*.xml)");
	gtk_file_filter_add_pattern(filter_xml,"*.xml");

	gtk_file_chooser_add_filter(chooser,filter_yml);
	gtk_file_chooser_add_filter(chooser,filter_xml);

	res = gtk_dialog_run(GTK_DIALOG(dialog));
	char *filename;
	filename = gtk_file_chooser_get_filename(chooser);
	gtk_widget_destroy(GTK_WIDGET(dialog));

	if(res == GTK_RESPONSE_ACCEPT) {
		int len = strlen(filename);

		if(!strcmp(filename+len-4,".yml") || !strcmp(filename+len-4,".xml")) {
			FileStorage fs(filename, FileStorage::WRITE);

			switch(data->matcher_type) {
			case BM:
				fs <<
				"name" << "StereoMatcher.BM" <<
				"blockSize" << data->block_size <<
				"minDisparity" << data->min_disparity <<
				"numDisparities" << data->num_disparities <<
				"disp12MaxDiff" << data->disp_12_max_diff <<
				"speckleRange" << data->speckle_range <<
				"speckleWindowSize" << data->speckle_window_size <<
				"preFilterCap" << data->pre_filter_cap <<
				"preFilterSize" << data->pre_filter_size <<
				"uniquenessRatio" << data->uniqueness_ratio <<
				"textureThreshold" << data->texture_threshold <<
				"preFilterType" << data->pre_filter_type;
				break;

			case SGBM:
				fs <<
				"name" << "StereoMatcher.SGBM" <<
				"blockSize" << data->block_size <<
				"minDisparity" << data->min_disparity <<
				"numDisparities" << data->num_disparities <<
				"disp12MaxDiff" << data->disp_12_max_diff <<
				"speckleRange" << data->speckle_range <<
				"speckleWindowSize" << data->speckle_window_size <<
				"P1" << data->p1 <<
				"P2" << data->p2 <<
				"preFilterCap" << data->pre_filter_cap <<
				"uniquenessRatio" << data->uniqueness_ratio <<
				"mode" << data->mode;
				break;
			}
			fs.release();

			GtkWidget *message = gtk_message_dialog_new(GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_INFO, GTK_BUTTONS_CLOSE, "Parameters saved successfully");
			gtk_dialog_run(GTK_DIALOG(message));
			gtk_widget_destroy(GTK_WIDGET(message));
		} else {
			GtkWidget *message = gtk_message_dialog_new(GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE, "Currently the only supported formats are XML and YAML.");
			gtk_dialog_run(GTK_DIALOG(message));
			gtk_widget_destroy(GTK_WIDGET(message));
		}

		g_free(filename);
	}
}

G_MODULE_EXPORT void on_btn_load_clicked(GtkButton *b, ChData *data) {
	GtkWidget *dialog;
	GtkFileChooser *chooser;
	GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
	gint res;

	dialog = gtk_file_chooser_dialog_new("Open File", GTK_WINDOW(data->main_window), action, "Cancel", GTK_RESPONSE_CANCEL, "Open", GTK_RESPONSE_ACCEPT, NULL);
	chooser = GTK_FILE_CHOOSER(dialog);

	GtkFileFilter *filter = gtk_file_filter_new();
	gtk_file_filter_set_name(filter,"YAML or XML file (*.yml, *.xml)");
	gtk_file_filter_add_pattern(filter,"*.yml");
	gtk_file_filter_add_pattern(filter,"*.xml");

	gtk_file_chooser_add_filter(chooser,filter);

	res = gtk_dialog_run(GTK_DIALOG(dialog));
	char *filename;
	filename = gtk_file_chooser_get_filename(chooser);
	gtk_widget_destroy(GTK_WIDGET(dialog));

	if(res == GTK_RESPONSE_ACCEPT) {
		int len = strlen(filename);

		if(!strcmp(filename+len-4,".yml") || !strcmp(filename+len-4,".xml")) {
			FileStorage fs(filename, FileStorage::READ);

			if(!fs.isOpened()) {
				GtkWidget *message = gtk_message_dialog_new(GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE, "Could not open the selected file.");
				gtk_dialog_run(GTK_DIALOG(message));
				gtk_widget_destroy(GTK_WIDGET(message));
			} else {
				string name;
				fs["name"] >> name;

				if(name == "StereoMatcher.BM") {
					data->matcher_type = BM;
					fs["blockSize"] >> data->block_size;
					fs["minDisparity"] >> data->min_disparity;
					fs["numDisparities"] >> data->num_disparities;
					fs["disp12MaxDiff"] >> data->disp_12_max_diff;
					fs["speckleRange"] >> data->speckle_range;
					fs["speckleWindowSize"] >> data->speckle_window_size;
					fs["preFilterCap"] >> data->pre_filter_cap;
					fs["preFilterSize"] >> data->pre_filter_size;
					fs["uniquenessRatio"] >> data->uniqueness_ratio;
					fs["textureThreshold"] >> data->texture_threshold;
					fs["preFilterType"] >> data->pre_filter_type;
					update_interface(data);
					update_matcher(data);

					GtkWidget *message = gtk_message_dialog_new(GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE, "Parameters loaded successfully.");
					gtk_dialog_run(GTK_DIALOG(message));
					gtk_widget_destroy(GTK_WIDGET(message));
				} else if(name == "StereoMatcher.SGBM") {
					data->matcher_type = SGBM;
					fs["blockSize"] >> data->block_size;
					fs["minDisparity"] >> data->min_disparity;
					fs["numDisparities"] >> data->num_disparities;
					fs["disp12MaxDiff"] >> data->disp_12_max_diff;
					fs["speckleRange"] >> data->speckle_range;
					fs["speckleWindowSize"] >> data->speckle_window_size;
					fs["P1"] >> data->p1;
					fs["P2"] >> data->p2;
					fs["preFilterCap"] >> data->pre_filter_cap;
					fs["uniquenessRatio"] >> data->uniqueness_ratio;
					fs["mode"] >> data->mode;
					update_interface(data);
					update_matcher(data);

					GtkWidget *message = gtk_message_dialog_new(GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_INFO, GTK_BUTTONS_CLOSE, "Parameters loaded successfully.");
					gtk_dialog_run(GTK_DIALOG(message));
					gtk_widget_destroy(GTK_WIDGET(message));
				} else {
					GtkWidget *message = gtk_message_dialog_new(GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE, "This file is not valid.");
					gtk_dialog_run(GTK_DIALOG(message));
					gtk_widget_destroy(GTK_WIDGET(message));
				}

				fs.release();
			}
		} else {
			GtkWidget *message = gtk_message_dialog_new(GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE, "Currently the only supported formats are XML and YAML.");
			gtk_dialog_run(GTK_DIALOG(message));
			gtk_widget_destroy(GTK_WIDGET(message));
		}

		g_free(filename);
	}
}
}

int main(int argc, char *argv[]) {
	char default_left_filename[] = "tsukuba/scene1.row3.col3.ppm";
	char default_right_filename[] = "tsukuba/scene1.row3.col5.ppm";
	char *left_filename = default_left_filename;
	char *right_filename = default_right_filename;
	int i;
	int norm_width = 320;
	int norm_height = 240;

	GtkBuilder *builder;
	GError *error = NULL;
	ChData *data;

	/* Parse arguments to find left and right filenames */
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-left") == 0) {
			i++;
			left_filename = argv[i];
		} else if (strcmp(argv[i], "-right") == 0) {
			i++;
			right_filename = argv[i];
		}
	}

	fprintf(stdout, "-left %s\n-right %s\n", left_filename, right_filename);

	/* Init GTK+ */
	gtk_init(&argc, &argv);

	/* Create data */
	data = new ChData();

	/* Create new GtkBuilder object */
	builder = gtk_builder_new();

	/* Load UI from file. If error occurs, report it and quit application.
	 * Replace "tut.glade" with your saved project. */
	if (!gtk_builder_add_from_file(builder, "StereoTuner.glade", &error)) {
		g_warning("%s", error->message);
		g_free(error);
		return (1);
	}

	/* Get main window pointer from UI */
	data->main_window = GTK_WIDGET(gtk_builder_get_object(builder, "window1"));
	data->image_left = GTK_IMAGE(gtk_builder_get_object(builder, "image_left"));
	data->image_right = GTK_IMAGE(
			gtk_builder_get_object(builder, "image_right"));
	data->image_depth = GTK_IMAGE(
			gtk_builder_get_object(builder, "image_disparity"));

	data->sc_block_size = GTK_WIDGET(gtk_builder_get_object(builder, "sc_block_size"));
	data->sc_min_disparity = GTK_WIDGET(gtk_builder_get_object(builder, "sc_min_disparity"));
	data->sc_num_disparities = GTK_WIDGET(gtk_builder_get_object(builder, "sc_num_disparities"));
	data->sc_disp_max_diff = GTK_WIDGET(gtk_builder_get_object(builder, "sc_disp_max_diff"));
	data->sc_speckle_range = GTK_WIDGET(gtk_builder_get_object(builder, "sc_speckle_range"));
	data->sc_speckle_window_size = GTK_WIDGET(gtk_builder_get_object(builder, "sc_speckle_window_size"));
	data->sc_p1 = GTK_WIDGET(gtk_builder_get_object(builder, "sc_p1"));
	data->sc_p2 = GTK_WIDGET(gtk_builder_get_object(builder, "sc_p2"));
	data->sc_pre_filter_cap = GTK_WIDGET(gtk_builder_get_object(builder, "sc_pre_filter_cap"));
	data->sc_pre_filter_size = GTK_WIDGET(gtk_builder_get_object(builder, "sc_pre_filter_size"));
	data->sc_uniqueness_ratio = GTK_WIDGET(gtk_builder_get_object(builder, "sc_uniqueness_ratio"));
	data->sc_texture_threshold = GTK_WIDGET(gtk_builder_get_object(builder, "sc_texture_threshold"));
	data->rb_pre_filter_normalized = GTK_WIDGET(gtk_builder_get_object(builder, "rb_pre_filter_normalized"));
	data->rb_pre_filter_xsobel = GTK_WIDGET(gtk_builder_get_object(builder, "rb_pre_filter_xsobel"));
	data->chk_full_dp = GTK_WIDGET(gtk_builder_get_object(builder, "chk_full_dp"));
	data->status_bar = GTK_WIDGET(gtk_builder_get_object(builder, "status_bar"));
	data->rb_bm = GTK_WIDGET(gtk_builder_get_object(builder, "algo_sbm"));
	data->rb_sgbm = GTK_WIDGET(gtk_builder_get_object(builder, "algo_ssgbm"));
	data->adj_block_size = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_block_size"));
	data->adj_min_disparity = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_min_disparity"));
	data->adj_num_disparities = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_num_disparities"));
	data->adj_disp_max_diff = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_disp_max_diff"));
	data->adj_speckle_range = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_speckle_range"));
	data->adj_speckle_window_size = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_speckle_window_size"));
	data->adj_p1 = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_p1"));
	data->adj_p2 = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_p2"));
	data->adj_pre_filter_cap = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_pre_filter_cap"));
	data->adj_pre_filter_size = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_pre_filter_size"));
	data->adj_uniqueness_ratio = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_uniqueness_ratio"));
	data->adj_texture_threshold = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_texture_threshold"));
	data->status_bar_context = gtk_statusbar_get_context_id(GTK_STATUSBAR(data->status_bar), "Statusbar context");

	//Put images on place
	gtk_image_set_from_file(data->image_left, left_filename);
	gtk_image_set_from_file(data->image_right, right_filename);

	data->cv_image_left = imread(left_filename,0);
	data->cv_image_right = imread(right_filename,0);
	update_matcher(data);

	/* Connect signals */
	gtk_builder_connect_signals(builder, data);

	/* Destroy builder, since we don't need it anymore */
	g_object_unref(G_OBJECT(builder));

	/* Show window. All other widgets are automatically shown by GtkBuilder */
	gtk_widget_show(data->main_window);

	/* Start main loop */
	gtk_main();

	return (0);
}
