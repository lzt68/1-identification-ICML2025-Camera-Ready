# 1-identification-ICML2025-Camera-Ready
Code Implementation for the ICML 2025 paper "Near Optimal Non-asymptotic Sample Complexity of 1-Identification"

## File Structure

icml2025_camera_ready.pdf: Camera-ready paper  "Near Optimal Non-asymptotic Sample Complexity of 1-Identification"

icml2025.zip: Latex template of the Camera-ready paper 

"source": Implement SEE and other benchmark algorithms

"Figure": All the figure used in the ICML paper

Notebooks for conducting numeric experiments:

+ Experiment_Benchmark_APGAI.ipynb: run numeric experiments for APGAI
+ Experiment_Benchmark_Kano.ipynb: run numeric experiments for lilHDoC, HDoC and LUCB\_G
+ Experiment_Benchmark_TaS_MS.ipynb: run numeric experiments for MS, TaS, whose stopping rule is GLR
+ Experiment_SEE_recycle.ipynb: run numeric experiments for SEE

Visualize-Demo-Delta_0p15.ipynb: Notebook for plotting the figures used in the paper

## Reproduce the Result

The numeric results for plotting figures is store in the folder "Numeric-Record-Delta_0p15".

> If you want to reproduce the existing result, please run the corresponding jupyter notebooks mentioned above. **You may need to reset the file path.** 

All the figures used in the current paper are stored in the folder "Figure".

> If you want to plot the figure with the numeric results, please run the jupyter notebook Visualize-Demo-Delta_0p15.ipynb
