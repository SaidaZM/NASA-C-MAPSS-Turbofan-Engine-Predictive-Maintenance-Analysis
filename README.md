# Predictive Maintenance Project (Python & MATLAB) – NASA C-MAPSS

This work was developed **using both Python and MATLAB** for predictive maintenance and reliability analysis of turbofan engines based on the **NASA C-MAPSS FD001 dataset**.

---

## Project Overview

This project focuses on **predictive maintenance**, **Remaining Useful Life (RUL) estimation**, and **reliability analysis** of aircraft turbofan engines using real industrial simulation data provided by NASA.

The objective is to anticipate failures before they occur by combining:
- Statistical reliability methods (Weibull, Kaplan–Meier, Nelson–Aalen)
- Machine learning models (Linear Regression, Random Forest, SVM)
- Degradation modeling and availability analysis (FMDS concepts)

---

## Technologies Used

### Python
- Data analysis: `NumPy`, `Pandas`
- Visualization: `Matplotlib`, `Seaborn`
- Machine Learning: `Scikit-learn`
- Reliability & Survival Analysis: `SciPy`, `lifelines`
- GUI for data loading: `Tkinter`

### MATLAB
- Signal processing
- Degradation modeling
- Reliability and maintenance-oriented simulations
- FMDS-oriented analysis

---

## Dataset

- **NASA C-MAPSS FD001**
- Simulated degradation data for multiple turbofan engines
- Each engine operates until failure
- Includes operational settings and sensor measurements

---

## Project Structure

```text
NASA-C-MAPSS-Turbofan-Engine-Predictive-Maintenance-Analysis/
│
├── matlab/
│   └── MATLAB scripts for modeling and analysis
│
├── python/
│   └── fiabilite/
│       ├── CMAPSSData/
│       │   └── FD001 data files
│       └── Full Python analysis script
│
└── README.md
