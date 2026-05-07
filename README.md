# ONNX-Predict: Predict the latency of ONNX Models using static features

This was a course project for [ECE 208: The Art of Machine Learning](https://hajim.rochester.edu/ece/sites/zduan/teaching/ece408/index.html) at the University of Rochester.

The project involves training a series of machine learning model to predict the latency of ONNX Models on x86 Linux machine using the ONNX Runtime CPU Provider.

# Structure

- data_collection: code to download ONNX models from the ONNX Model Zoo and collect static features
- inference: run slurm jobs or python script to collect inference data
- training: scripts that were used to train models
- tool: python script that performs inference and predicts latency given an ONNX Model
