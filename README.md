# VirtuDockDL: Automated Virtual Screening

VirtuDockDL is a comprehensive solution for streamlining the process of drug discovery and molecular analysis. With VirtuDockDL, you can harness the power of deep learning to perform virtual screening, evaluate molecular activities, and predict binding affinities with unprecedented accuracy and speed.

## Features

- **Graph Neural Network-Based Ligand Prioritization:** Streamline drug discovery with our GNN model, prioritizing ligands for speed and accuracy.
- **Descriptor Analysis:** Analyze molecular descriptors to predict pharmacological profiles and drug-likeness.
- **Re-screening:** Refine your ligand search iteratively, utilizing new data for targeted identification.
- **Protein Refinement:** Facilitates protein refinement by uploading PDB files.
- **Molecular Docking:** Predict ligand interactions with state-of-the-art simulations, focusing on optimal compounds.
- **Scalable Data Processing:** Efficiently process and analyze data across all scales, ensuring fast, reliable drug discovery results.

## Installations

### Prerequisites
- Python 3.8 or higher
- [PyTorch](https://pytorch.org/)
- [RDKit](https://www.rdkit.org/)
- [OpenMM](https://openmm.org/)
- [Flask](https://flask.palletsprojects.com/)

### Installing Required Libraries
```sh
pip install flask torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib rdkit biopython dgl
pip install openmm
Usage
Running the Application
Clone the repository:

sh
Copy code
git clone https://github.com/FatimaNoor74/VirtuDockDL/.git
cd VirtuDockDL
Set up your environment and install dependencies:

## Usage

### Running the Application

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/VirtuDockDL.git
    cd VirtuDockDL
    ```

2. **Set up your environment and install dependencies**
3. **Run the Flask application**:
    ```sh
    python app.py
    ```
    Open your web browser and navigate to `http://127.0.0.1:5000` to access VirtuDockDL.

### Uploading Files

#### 1. Upload CSV File for Ligand Prioritization
Navigate to the Ligand Prioritization tab and upload your CSV file containing data of active and inactive molecules.
```html
<form method="POST" enctype="multipart/form-data">
    <label for="file">Select a CSV File:</label>
    <input type="file" name="file" id="file" accept=".csv" required>
    <button type="submit">Upload</button>
</form>
