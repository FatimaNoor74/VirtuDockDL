# VirtuDockDL: An Automated Python- based Pipeline for Streamlined Virtual Screening and Drug Discovery

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

```

# Usage
## Running the Application
Clone the repository:
```sh
git clone https://github.com/yourusername/VirtuDockDL.git
cd VirtuDockDL
```
Set up your environment and install dependencies

Run the Flask application
```sh
python app.py
```
Open your web browser and navigate to http://127.0.0.1:5000 to access VirtuDockDL.

# Uploading Files
## Upload CSV File for Ligand Prioritization
Navigate to the Ligand Prioritization tab and upload your CSV file containing data of active and inactive molecules.
```sh
<form method="POST" enctype="multipart/form-data">
    <label for="file">Select a CSV File:</label>
    <input type="file" name="file" id="file" accept=".csv" required>
    <button type="submit">Upload</button>
</form>
```
## Upload PDB File for Protein Refinement
Navigate to the Protein Refinement tab and upload your PDB file.
```sh
<form method="post" enctype="multipart/form-data">
    <label for="proteinFile">Select a PDB File:</label>
    <input type="file" name="file" id="proteinFile" accept=".pdb" required>
    <button type="submit">Upload</button>
</form>
```
## Upload ZIP File for Molecular Docking
Navigate to the Molecular Docking tab and upload your ZIP file containing ligand structures.
```sh
<form id="docking-form" method="POST" enctype="multipart/form-data">
    <label for="protein_file">Select a Protein File (.pdb):</label>
    <input type="file" name="protein_file" id="protein_file" accept=".pdb" required>
    <label for="ligand_zip">Select Ligand Zip File (.zip):</label>
    <input type="file" name="ligand_zip" id="ligand_zip" accept=".zip" required>
    <button type="submit">Upload and Start Docking</button>
</form>
```
# Main Functionalities
## Ligand Prioritization
```sh

def predict():
    smiles = request.form['smiles']
    model = GNN()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    mol_data = mol_to_graph(smiles)
    with torch.no_grad():
        output = model(mol_data.x, mol_data.edge_index, mol_data.batch)
        activity = torch.sigmoid(output)
    
    return jsonify({'activity': activity.item()})
```
# Protein Refinement
def refine_protein():
    file = request.files['file']
    pdb = PDBFile(file)
    forcefield = ForceField('amber99sb.xml')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)
    system = forcefield.createSystem(modeller.topology)
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picosecond)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    
    refined_file = 'refined_protein.pdb'
    with open(refined_file, 'w') as f:
        PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
    
    return send_file(refined_file, as_attachment=True)

```
# Molecular Docking
```sh
from flask import Flask, request, send_file
import subprocess

app = Flask(__name__)

@app.route('/dock', methods=['POST'])
def dock():
    protein_file = request.files['protein_file']
    ligand_file = request.files['ligand_file']
    output_dir = 'docking_results'
    
    protein_path = f"{output_dir}/protein.pdb"
    ligand_path = f"{output_dir}/ligand.pdbqt"
    protein_file.save(protein_path)
    ligand_file.save(ligand_path)
    
    subprocess.run(['vina', '--receptor', protein_path, '--ligand', ligand_path, '--out', f'{output_dir}/out.pdbqt'])
    
    return send_file(f'{output_dir}/out.pdbqt', as_attachment=True)
```
# Tips for Success

- Ensure your input files are correctly formatted and contain all necessary information.
- Utilize the "De Novo Molecule Generation" feature to explore new ligands based on specified criteria.
- Take advantage of our re-screening feature to iteratively refine your search for the optimal ligand.
  
# Contributing
We welcome contributions! Please fork the repository and submit pull requests for any enhancements or bug fixes.

# Contact
For any questions or issues, please open an issue on this repository or contact us at tahirulqamar@gcuf.edu.pk.









