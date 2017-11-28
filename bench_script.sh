#!/bin/zsh

function run {
  git checkout $1
  mkdir -p build
  cd build
  cmake .. 1>/dev/null 2>/dev/null
  make appperlin cudaappperlin appall_modules cudaappall_modules 1>/dev/null 2>/dev/null
  echo "Make finished" >&2
  echo $2
  echo SHA-1 $1
  cd ../bin
  echo "./appperlin 32 $3 $4 $5 20 50 300 16 false 200 0 1000000"
  {time ./appperlin 32 $3 $4 $5 20 50 300 16 false 200 0 1000000 1>/dev/null 2>/dev/null} 2>&1
  echo "./cudaappperlin 32 $3 $4 $5 20 50 300 16 false 200 0 1000000"
  {time ./cudaappperlin 32 $3 $4 $5 20 50 300 16 false 200 0 1000000 1>/dev/null 2>/dev/null} 2>&1
  echo "./appall_modules 32 $3 $4 $5 20 50 300 16 false 200 0 1000000"
  {time ./appall_modules 32 $3 $4 $5 20 50 300 16 false 200 0 1000000 1>/dev/null 2>/dev/null} 2>&1
  echo "./cudaappall_modules 32  $3 $4 $5 20 50 300 16 false 200 0 1000000"
  {time ./cudaappall_modules 32 $3 $4 $5 20 50 300 16 false 200 0 1000000 1>/dev/null 2>/dev/null} 2>&1
  cd ..
}

run 46d5587c "Initial Version" 8 2 16
run b72d8d9f "openmp" 8 2 16
run 4e0426a2 "streams and pinned memory" 8 2 16
run b5165997 "memory pool" 8 2 16
run 3b9b85df "limit number of registers" 8 2 32
run 5b4cad2d "force inlining" 8 2 32
