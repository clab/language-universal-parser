PROJECT(cnn:parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(lstm-parse lstm-parse.cc)
target_link_libraries(lstm-parse cnn ${Boost_LIBRARIES})

  if (WITH_CUDA_BACKEND)
    add_dependencies(lstm-parse cnncuda)
    target_link_libraries(lstm-parse cnncuda)
    CUDA_ADD_CUBLAS_TO_TARGET(lstm-parse)
  endif (WITH_CUDA_BACKEND)
