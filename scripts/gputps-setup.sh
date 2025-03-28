#!/bin/bash                                                                                                                           

# Delete makefiles first and then all cmakefiles regarding the previous
# initializiation. This guarantees a fresh build to keep track of current
# GIT-versions.
make clean
find $PWD -iwholename '*cmake*' -not -name CMakeLists.txt -delete

# cmake setup 
# dealii with int32 indices
cmake -DCMAKE_BUILD_TYPE="Release" \
      -DDEAL_II_DIR=/scratch/cucui/lib/dealii_cudampi_mg_int32_trilinos/ \
      ..
      # -DDEAL_II_DIR=/export/home/cucui/SimServ06/lib/dealii_cudampi_mg_int32_trilinos_backup/ \
