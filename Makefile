# Compiler settings
CC=g++
CFLAGS=-Isrc -std=c++11 -fopenmp
DEPS = src/image.hpp src/sift.hpp src/parallel_omp_image.hpp src/parallel_omp_sift.hpp
OBJ = src/image.o src/sift.o 
PARALLEL_OMP_OBJ = src/parallel_omp_image.o src/parallel_omp_sift.o

# Rule to compile cpp files to object files, ignoring main-containing files for now
%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Dependencies for the object files
src/sift.o: src/sift.cpp src/sift.hpp src/image.hpp
src/image.o: src/image.cpp src/image.hpp src/stb_image.h src/stb_image_write.h
src/parallel_omp_sift.o: src/parallel_omp_sift.cpp src/parallel_omp_sift.hpp src/parallel_omp_image.hpp
src/parallel_omp_image.o: src/parallel_omp_image.cpp src/parallel_omp_image.hpp src/stb_image.h src/stb_image_write.h

# Main applications
FIND_KEYPOINTS_OBJ = run/find_keypoints.o $(OBJ)
MATCH_FEATURES_OBJ = run/match_features.o $(OBJ)
PARALLEL_OMP_FIND_KEYPOINTS_OBJ = run/parallel_omp_find_keypoints.o $(PARALLEL_OMP_OBJ)
PARALLEL_OMP_MATCH_FEATURES_OBJ = run/parallel_omp_match_features.o $(PARALLEL_OMP_OBJ)

# Executable rules
find_keypoints: $(FIND_KEYPOINTS_OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

match_features: $(MATCH_FEATURES_OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

find_keypoints_omp: $(PARALLEL_OMP_FIND_KEYPOINTS_OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

match_features_omp: $(PARALLEL_OMP_MATCH_FEATURES_OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

# Rule for compiling main-containing cpp files separately
run/%.o: run/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Build all targets
all: find_keypoints match_features find_keypoints_omp match_features_omp

# Rule for cleaning the project
clean:
	rm -f src/*.o run/*.o find_keypoints match_features find_keypoints_omp match_features_omp


