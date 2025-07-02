# Spatio-Temporal Segmentation of Left Ventricular Wall in Murine Echocardiography

This repository contains the codebase for training a modular deep learning pipeline for the automatic segmentation of the **left ventricular wall (LVW)** in **murine echocardiography (mmECHO)**. The pipeline is based on a combination of spatial and temporal models: **Res-SE-U-Net** and **ConvLSTM**, as presented in the manuscript:

📄 **Spatio-Temporal Deep Learning-based Segmentation of Left Ventricular Wall in Murine Model Echocardiography**

---

## Overview
The proposed approach performs segmentation in two stages:
1. **Spatial Segmentation** using `Res-SE-U-Net`:
   - A U-Net backbone enhanced with residual connections (ResNet18) and Squeeze-and-Excitation (SE) blocks to emphasize meaningful spatial features.
2. **Temporal Refinement** using `ConvLSTM`:
   - A recurrent module that refines the segmentation masks by capturing temporal coherence across frame sequences.

---

## Repository Structure

