# Compiler settings
CC=g++
CFLAGS=-Isrc -Wall -std=c++11
DEPS = src/image.hpp src/sift.hpp
OBJ = src/image.o src/sift.o

# Rule to compile cpp files to object files, ignoring main-containing files for now
%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Dependencies for the object files
src/sift.o: src/sift.cpp src/sift.hpp src/image.hpp
src/image.o: src/image.cpp src/image.hpp src/stb_image.h src/stb_image_write.h

# Main applications
FIND_KEYPOINTS_OBJ = run/find_keypoints.o $(OBJ)
MATCH_FEATURES_OBJ = run/match_features.o $(OBJ)

# Executable rules
find_keypoints: $(FIND_KEYPOINTS_OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

match_features: $(MATCH_FEATURES_OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

# Rule for compiling main-containing cpp files separately
run/%.o: run/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Build all targets
all: find_keypoints match_features

# Rule for cleaning the project
clean:
	rm -f src/*.o run/*.o find_keypoints match_features


