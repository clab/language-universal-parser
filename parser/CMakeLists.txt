PROJECT(dynet:parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(lstm-parse lstm-parse.cc)
target_link_libraries(lstm-parse dynet ${Boost_LIBRARIES})

  if (WITH_CUDA_BACKEND)
    add_dependencies(lstm-parse dynetcuda)
    target_link_libraries(lstm-parse dynetcuda)
    CUDA_ADD_CUBLAS_TO_TARGET(lstm-parse)
  endif (WITH_CUDA_BACKEND)
