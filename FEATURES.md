# Proposed Features for Cellular Dynamics Analysis

Here are three feature ideas to enhance the functionality and maintainability of this project.

## 1. Implement Parametric Growth Models

**Concept:**
Currently, the analysis focuses on non-parametric measures like AUC. This feature would involve implementing various parametric growth models (e.g., Logistic, Gompertz, Richards) to fit the time-series data.

**Benefits:**
- **Quantitative Insights:** Extract key biological parameters such as:
    - `λ` (lambda): Length of the lag phase.
    - `μ` (mu): Maximum specific growth rate.
    - `A`: Maximum cell growth (carrying capacity).
- **Model Selection:** Use statistical criteria like the Akaike Information Criterion (AIC) to automatically select the best-fitting model for a given curve.
- **Deeper Comparison:** Allow for more nuanced comparisons between different experimental conditions based on these quantitative parameters.

**Implementation Sketch:**
- Use the existing `growth` function in `scripts/functions/functions.py` as a starting point.
- Add more model functions (Gompertz, etc.).
- Use `scipy.optimize.curve_fit` to fit the models to the data.
- Calculate AIC for each model and identify the best fit.
- Add the calculated parameters (`λ`, `μ`, `A`) and the best model to the output `Summary.csv` file.
- Create new plots to visualize the model fits against the raw data.

## 2. Add Multivariate Analysis (PCA)

**Concept:**
Introduce multivariate analysis techniques to provide a higher-level overview of the data. Principal Component Analysis (PCA) would be an excellent first step.

**Benefits:**
- **Dimensionality Reduction:** Condense the complex time-series data into a few principal components, making it easier to visualize and interpret.
- **Pattern Discovery:** Identify clusters, outliers, and trends in the data that might not be obvious from individual growth curves.
- **Hypothesis Generation:** Help researchers generate new hypotheses by revealing unexpected relationships between samples.

**Implementation Sketch:**
- Add a new optional step in the analysis pipeline.
- Use a library like `scikit-learn` to perform PCA on the growth curve data (or the derived parameters from Feature 1).
- Generate a PCA plot (a "scores plot") to visualize the samples in the space of the first two principal components.
- Color the points in the PCA plot based on experimental variables from the `Design.xlsx` file (e.g., strain, treatment).
- Save the PCA plot as a new output file.

## 3. Refactor into a Class-Based Structure

**Concept:**
Refactor the existing functional code into a more organized, object-oriented structure. This was suggested by the `TODO` comment in `scripts/functions/functions.py`.

**Benefits:**
- **Improved Maintainability:** A class-based structure makes the code easier to understand, debug, and modify.
- **Enhanced Extensibility:** Adding new features (like the ones proposed above) becomes simpler and less error-prone.
- **Better Testability:** Code organized into classes is easier to unit test, leading to more robust and reliable software.

**Implementation Sketch:**
- Create a set of classes to encapsulate different parts of the workflow. For example:
    - `ExperimentReader`: For reading and parsing the `Design.xlsx` file and input data files.
    - `DataProcessor`: For cleaning, smoothing, and transforming the data.
    - `GrowthAnalyzer`: For performing the core analysis (AUC, parametric models, PCA).
    - `Plotter`: For generating all the output plots.
- Refactor the code from `growth.py` and `functions.py` into methods within these classes.
- The main `growth.py` script would then become a simpler script that instantiates these classes and calls their methods in the correct order.
