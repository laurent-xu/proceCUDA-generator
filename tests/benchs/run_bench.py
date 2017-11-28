import os
import time

def main():
  os.chdir("../../bin/")
  #range_voxels_per_grid = [4, 8, 16, 32, 64, 128]
  range_voxels_per_grid = [4, 8]
  range_voxels_total = [8 ** e for e in range(len(range_voxels_per_grid))][::-1]
  nb_camera_position = 100
  min_position = -100
  max_position = 100

  best_times = [0] * len(range_voxels_per_grid)
  best_values = [(0, 0, 0)] * len(range_voxels_per_grid)
  for it in range(len(range_voxels_per_grid)):
    print("============================")
    nb_voxels = range_voxels_per_grid[it]
    print("nb voxels per grid: %d" % nb_voxels)
    time1 = time.time()
    os.system("./appsphere %d 0 0 0 %d 0 %d false %d %d %d" % (
          range_voxels_per_grid[it], range_voxels_total[it], range_voxels_total[it], 
          nb_camera_position, min_position, max_position))
    time2 = time.time()
    res = (time2 - time1) * 1000
    print("CPU: %d" % res)

    range_threads = [1, 2, 4, 8, 16, 32, 64, 128]
    best_times[it] = 100000000
    best_values[it] = (0, 0, 0)
    for i in range_threads:
      for j in range_threads:
        for k in range_threads:
          if (i * j * k < 2024 and i < nb_voxels and j < nb_voxels and k < nb_voxels):
            time1 = time.time()
            os.system("./cudaappsphere %d %d %d %d %d 0 %d false %d %d %d" % (
                range_voxels_per_grid[it], i, j, k,
                range_voxels_total[it], range_voxels_total[it],
                nb_camera_position, min_position, max_position))
            time2 = time.time()
            res = (time2 - time1) * 1000
            print("(%d, %d, %d): %d" % (i, j, k, res))
            if (res < best_times[it]):
              best_values[it] = (i, j, k)
              best_times[it] = res

    print(best_values)
    print(best_times)

if __name__ == "__main__":
  main()
