all: main.cpp
	g++ -g `pkg-config --cflags gtk+-3.0 gmodule-2.0 opencv4` main.cpp -o stereo-tuner `pkg-config --libs gtk+-3.0 gmodule-export-2.0 opencv4`
