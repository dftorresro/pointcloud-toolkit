# pointcloud-segmentation-viewer

**pointcloud-segmentation-viewer** is a modular and extensible toolkit for semantic segmentation experiments on 3D point clouds.  
It provides interactive visualization of segmentation results and allows users to track predictions and progress over epochs.

This project starts as a Python-based tool leveraging [Open3D](http://www.open3d.org/) for local interactive visualization and is designed to evolve into a web-based application for accessible, browser-based exploration of segmentation results.

Key features:
- Load and visualize segmented 3D point clouds interactively
- Inspect segmentation results across epochs to monitor training evolution
- Modular architecture for integration with any deep learning pipeline
- Future-ready design for deploying a web frontend (Three.js + FastAPI backend)

This toolkit is ideal for researchers and engineers working on 3D semantic segmentation who want an integrated tool to both train models and analyze results visually.
