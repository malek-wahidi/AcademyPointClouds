## Pipeline Summary
- **Preprocessing:** Voxel downsampling, normal estimation, FPFH feature extraction
- **Global Registration:** Fast Global Registration (FGR)
- **Local Refinement:** Multiscale Point-to-Plane ICP
- **Evaluation:** Fitness, RMSE, correspondence count

## Runs Evaluation
Run 1:
------
Registration took 259.8699 seconds.
[RESULT] Final transformation matrix:
 [[ 0.46120604  0.4231363  -0.77990042 -0.12933523]
 [-0.39819621  0.88418752  0.24423803  0.03304724]
 [ 0.79292419  0.19790933  0.57628389 -3.94284983]
 [ 0.          0.          0.          1.        ]]
Registration accuracy metrics:
  Fitness: 62.11 %
  Inlier RMSE: 0.0066
  Correspondences found: 123494


Run 2: Using fast global registration:
--------------------------------------
Registration took 0.1978 seconds.
[RESULT] Final transformation matrix:
 [[ 0.46118943  0.42306343 -0.77994977 -0.12896458]
 [-0.39801669  0.88425383  0.24429056  0.03188349]
 [ 0.79302398  0.19776881  0.57619482 -3.94266438]
 [ 0.          0.          0.          1.        ]]
Registration accuracy metrics:
  Fitness: 62.11 %
  Inlier RMSE: 0.0066
  Correspondences found: 123489
  

Run 3: Using multiscale ICP:
----------------------------
[Multiscale ICP] Level 1 - voxel: 0.05, max_iter: 50
  -> Fitness: 65.92%, RMSE: 0.022416

[Multiscale ICP] Level 2 - voxel: 0.03, max_iter: 30
  -> Fitness: 64.68%, RMSE: 0.013948

[Multiscale ICP] Level 3 - voxel: 0.01, max_iter: 14
  -> Fitness: 61.30%, RMSE: 0.006702
Registration took 0.5566 seconds.
[RESULT] Final transformation matrix:
 [[ 0.46162643  0.4225525  -0.77996822 -0.13004424]
 [-0.39859787  0.88430232  0.24316485  0.03584365]
 [ 0.79247762  0.19864235  0.57664585 -3.94463257]
 [ 0.          0.          0.          1.        ]]
Registration accuracy metrics:
  Fitness: 62.09 %
  Inlier RMSE: 0.0066
  Correspondences found: 123449

## Final Metrics
- Fitness: 62.11%
- Inlier RMSE: 0.0066
- Runtime: ~0.2s
- Method: FGR + Standard ICP
