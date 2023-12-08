## Environment
- Simply run `pip install -r requirements.txt` in a new conda environment

## Notice
- When doing calibration, select the parameter for your sensor in `per_sensor_params.py` first.
- When running reconstruction code, you need to modify the `root` path in the script to the parent folder of your `<netid>` folder.
- The `reconstruction_tool.py` visualization part may not work on some OS due to an Open3D bug. You can visualize the generated point cloud and mesh with other tools like `pcl_viewer` or MeshLab.
