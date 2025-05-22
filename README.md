# GeneticPCA: Unsupervised Clustering with Genetic Algorithm Optimization

This project provides a modular structure for unsupervised clustering and anomaly detection using PCA and DBSCAN. It features a genetic algorithm optimization approach to automatically select the optimal number of PCA components for clustering quality.

The file `main.py` serves as the main entry point for the pipeline execution, orchestrated through a YAML configuration file.

## Project Structure

```
GeneticPCA/
├── data/                     # Folder for datasets (CSV files)
├── models/                   # Folder for saved outputs (plots, models)
├── notebooks/                # Folder for Jupyter notebooks (for exploratory analysis)
├── data_loader/              # Module for loading data
│   ├── __init__.py
│   └── data_loader.py
├── preprocessor/             # Module for data preprocessing
│   ├── __init__.py
│   └── preprocessor.py
├── clustering/               # Module for clustering algorithms
│   ├── __init__.py
│   └── clustering_utils.py
├── genetic_algorithm/        # Module for genetic algorithm optimization
│   ├── __init__.py
│   └── genetic_pca_optimizer.py
├── bayesian_methods/         # Module for bayesian methods (placeholder)
│   ├── __init__.py
│   └── bayesian_methods.py
├── main.py                   # Main script to execute the pipeline
├── config.yaml               # Configuration file for the pipeline
├── requirements.txt          # Python dependencies
└── README.md                 
```

## Core Features

- **Unsupervised Learning Pipeline**: Complete workflow for anomaly detection using dimensional reduction and clustering.
- **Genetic Algorithm for PCA Optimization**: Automatically determines the optimal number of principal components to maximize clustering quality.
- **Configurable via YAML**: All parameters can be adjusted through a simple configuration file.
- **Modular Architecture**: Each key functionality is encapsulated in its own Python module:
    - `data_loader`: Reads data from CSV files.
    - `preprocessor`: Handles data cleaning and scaling for unsupervised learning.
    - `clustering`: Implements PCA and DBSCAN algorithms with evaluation metrics.
    - `genetic_algorithm`: Contains the genetic algorithm for optimizing PCA component selection.
- **Visual Output**: Generates plots of clusters and anomalies, along with GA fitness evolution.

## Getting Started

1. **Clone or download the repository**
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare your data:**
   - Place your CSV file(s) in the `data/` folder
   - Modify `config.yaml` to specify your data source
5. **Configure the pipeline:**
   - Adjust DBSCAN parameters in `config.yaml`
   - Choose whether to use GA for PCA optimization or manual settings
6. **Run the pipeline:**
   ```bash
   python main.py
   ```

## Configuration Options

The pipeline is configured through `config.yaml`, which offers the following settings:

- **Data loading options**: Set folder path and specify a single file or use all CSV files
- **Preprocessing**: Choose between StandardScaler or MinMaxScaler
- **PCA optimization**: Enable/disable genetic algorithm, or set manual PCA components
- **Genetic algorithm parameters**: Configure population size, generations, mutation rates
- **Clustering parameters**: Set DBSCAN eps and min_samples values

## Modules in Detail

- **data_loader**: Handles loading CSV files from a directory or specific file
- **preprocessor**: Performs data cleaning, selects numeric features, and applies scaling  
- **clustering**: Implements PCA for dimensionality reduction, DBSCAN for clustering, and evaluation metrics
- **genetic_algorithm**: Contains the GeneticPCAOptimizer class that optimizes PCA components based on clustering quality

The generated results include annotated data with cluster labels and anomaly flags, as well as visualizations of the clusters and potential anomalies.

## Output

The pipeline produces:
- CSV file with original data plus cluster labels and anomaly flags
- 2D visualization of clusters with highlighted anomalies
- GA fitness evolution plot (when genetic algorithm is enabled)

## Next Steps

- Expand the functionality of the `bayesian_methods` module
- Add command line argument support in `main.py`
- Develop unit tests and integration tests
- Implement additional clustering algorithms beyond DBSCAN
- Add more advanced anomaly detection techniques

## Contributions

Feel free to contribute to the project by opening issues or pull requests.