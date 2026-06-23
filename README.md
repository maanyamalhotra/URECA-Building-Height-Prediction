# URECA-Building-Height-Prediction

# Building Height Prediction Using Urban Planning Features in Singapore

## Overview

This repository contains the code developed for a URECA research project at Nanyang Technological University (NTU).

The project investigates the use of machine learning methods to predict building heights in Singapore using building footprint geometry, official URA Master Plan zoning information, and neighbourhood-level spatial context features.

---

## Project Workflow

### 1. Coordinate Transformation

Building coordinates are converted from the Singapore SVY21 coordinate system (EPSG:3414) to WGS84 latitude/longitude coordinates (EPSG:4326).

**Script:**

* `convert_coordinates.py`

---

### 2. URA Zoning Assignment

Building locations are spatially matched with official URA Master Plan land-use polygons using point-in-polygon spatial joins.

**Script:**

* `match_buildings_to_ura_zones.py`

---

### 3. Neighbourhood Feature Generation

Neighbourhood-level contextual features are generated using surrounding buildings within a fixed radius.


**Script:**

* `create_neighbourhood_features.py`

---

### 4. Model Training

An XGBoost regression model is trained using:

* Building geometry features
* URA zoning information
* Neighbourhood context features

**Script:**

* `train_xgboost_neighbourhood.py`

---

### 5. Model Evaluation and Visualisation

Performance evaluation and feature importance analysis are conducted to assess model behaviour and interpretability.

**Scripts:**

* `plot_importance.py`
* `make_final_model_graphs.py`

---

## Repository Structure

```text
.
├── convert_coordinates.py
├── match_buildings_to_ura_zones.py
├── create_neighbourhood_features.py
├── train_xgboost_neighbourhood.py
├── plot_importance.py
├── make_final_model_graphs.py
├── figures/
├── results/
└── README.md
```

---

## Author

**Maanya Malhotra**
URECA Researcher
Nanyang Technological University
