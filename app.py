import random
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory, after_this_request
from flask import send_from_directory, jsonify
import os
from flask import session
import logging
import json
from openmm.app import PDBFile, Modeller, ForceField, Simulation, PME, HBonds
from openmm import LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, picoseconds, nanometer
from pathlib import Path
from Bio.PDB import PDBIO
from torch_geometric.data import Data
from flask import Flask, request, jsonify
from rdkit.Chem import PandasTools
import zipfile
import uuid
import subprocess
import shutil
import time
import requests
from werkzeug.utils import secure_filename
from flask import send_from_directory
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from Bio.PDB import PDBParser
import numpy as np
from werkzeug.utils import secure_filename
import csv
import dgl
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, RDKFingerprint
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from flask import Flask
from gnn_model import GNN
from flask import Flask, request, render_template, flash, send_file
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from gnn_model import GNN  # Import GNN class from gnn_model.py
from gnn_utils import mol_to_graph, MoleculeDataset, collate
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from gnn_model import GNN
from gnn_utils import collate, mol_to_graph, MoleculeDataset
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser, CaPPBuilder
import mdtraj as md
import openmm
from openmm.app import *  # This will import the necessary 'app' module classes and functions
from openmm import *
from openmm.unit import *
from openmm.app import PDBFile, Modeller, ForceField
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, send_from_directory, url_for, current_app, flash, redirect, render_template
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOADED_FILES_DIR'] = 'uploaded_files'
app.config['GENERATED_FILES_DIR'] = 'generated_files'
app.config['uploaded_files_dir'] = 'uploaded_files'
app.config['generated_files_dir'] = 'generated_files'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOCKING_RESULTS_DIR'] = 'docking_results'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'zip', 'pdb', 'sdf'}

# Ensure directories exist
for directory in [app.config['GENERATED_FILES_DIR'], app.config['UPLOADED_FILES_DIR'], app.config['generated_files_dir'], app.config['uploaded_files_dir'], app.config['UPLOAD_FOLDER'], app.config['DOCKING_RESULTS_DIR']]:
    os.makedirs(directory, exist_ok=True)

# Directory setup
for directory in [app.config['GENERATED_FILES_DIR'], app.config['UPLOADED_FILES_DIR'], app.config['generated_files_dir'], app.config['uploaded_files_dir'],app.config['UPLOAD_FOLDER']]:
    os.makedirs(directory, exist_ok=True)

def save_data_to_csv(data, filename):
    """Save data to CSV format."""
    with open(filename, 'w', newline='') as csv_file:
        fieldnames = ['SMILES', 'Activity']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for smiles, activity in data:
            writer.writerow({'SMILES': smiles, 'Activity': activity})


def preprocess_csv(file):
    """Preprocess the uploaded CSV file."""
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Check if the CSV file contains the required columns
        if 'SMILES' not in df.columns or 'Activity' not in df.columns:
            flash('The CSV file must have "SMILES" and "Activity" columns.', 'error')
            return None

        # Canonicalize and validate SMILES strings
        df['SMILES'] = df['SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), canonical=True) if Chem.MolFromSmiles(x) is not None else None)

        # Drop rows with None (invalid SMILES) values
        df.dropna(subset=['SMILES'], inplace=True)

        # Save the preprocessed data in CSV format
        timestamp = int(time.time())
        filename = f'uploaded_data_{timestamp}.csv'
        save_data_to_csv(df.values.tolist(), filename)
        #flash('CSV file uploaded and saved successfully.', 'success')

        return df
    except Exception as e:
        flash(f'Error processing the CSV file: {str(e)}', 'error')
        return None


def train_and_evaluate_model(train_dataloader, test_dataloader, model, optimizer, criterion, scheduler):
    """Train and evaluate the GNN model."""
    best_val_loss = float('inf')
    patience = 10
    stop_counter = 0
    checkpoint_path = 'best_model.pth'
    for epoch in range(50):
        model.train()
        train_loss = 0
        for batched_graph, batched_features, batched_labels in train_dataloader:
            if batched_graph is None:
                continue
            optimizer.zero_grad()
            outputs = model(batched_graph, batched_features)
            loss = criterion(outputs, batched_labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        model.eval()
        val_loss = 0
        for batched_graph, batched_features, batched_labels in test_dataloader:
            if batched_graph is None:
                continue
            with torch.no_grad():
                outputs = model(batched_graph, batched_features)
            loss = criterion(outputs, batched_labels)
            val_loss += loss.item()
        val_loss /= len(test_dataloader)
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stop_counter += 1
        if stop_counter >= patience:
            print("Early stopping triggered.")
            break
    model.load_state_dict(torch.load(checkpoint_path))
    return model
def download_clusters():
    return send_file(os.path.join(app.config['GENERATED_FILES_DIR'], 'final_clusters.csv'), as_attachment=True,
                     attachment_filename='final_clusters.csv')

@app.route('/download', methods=['GET'])
def download():
    return send_file(os.path.join(app.config['GENERATED_FILES_DIR'], 'generated_molecules.csv'), as_attachment=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    final_compounds_filename = f'final_compounds_{timestamp}.csv'
    final_clusters_filename = f'final_clusters_{timestamp}.csv'
    cluster_plot_filename = f'cluster_plot_{timestamp}.png'
    model = GNN(1, 64, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    virtual_screening = False
    uploaded_file_path = None
    generated_file_path = None
    generated_molecules = None
    plot_file_path = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = f'Molecules_{timestamp}.csv'
                uploaded_file_path = os.path.join(app.config['UPLOADED_FILES_DIR'], filename)
                file.save(uploaded_file_path)
                #flash('CSV file uploaded and saved successfully.', 'success')

                # Load data and preprocess
                data = pd.read_csv(uploaded_file_path)
                smiles = data["SMILES"].tolist()
                labels = data["Activity"].astype(int).tolist()
                full_dataset = MoleculeDataset(smiles, labels)

                splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
                train_indices, test_indices = next(splitter.split(smiles, labels))
                train_dataset = [full_dataset[i] for i in train_indices]
                test_dataset = [full_dataset[i] for i in test_indices]

                train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
                test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)

                #flash('Dataset prepared and split into training and testing sets successfully.', 'success')

                # Train the model
                model = train_and_evaluate_model(train_dataloader, test_dataloader, model, optimizer, criterion,
                                                 scheduler)

                all_predictions, all_targets = [], []
                for batched_graph, batched_features, batched_labels in test_dataloader:
                    if batched_graph is None:
                        continue

                    with torch.no_grad():
                        outputs = model(batched_graph, batched_features)
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(batched_labels.cpu().numpy())

                accuracy = accuracy_score(all_targets, all_predictions)
                precision = precision_score(all_targets, all_predictions)
                recall = recall_score(all_targets, all_predictions)
                f1 = f1_score(all_targets, all_predictions)

                print(f"Test Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")

                # Evaluate the model on test data
                model.eval()
                all_probabilities = []

                for batched_graph, batched_features, batched_labels in test_dataloader:
                    if batched_graph is None:
                        continue
                    with torch.no_grad():
                        outputs = model(batched_graph, batched_features)
                        probabilities = torch.softmax(outputs, dim=1)
                    all_probabilities.extend(probabilities.cpu().numpy())
                # Calculate AUC
                all_probabilities = np.array(all_probabilities)
                true_labels = np.array(all_targets)
                class_1_probs = all_probabilities[:, 1]

                auc = roc_auc_score(true_labels, class_1_probs)

                print(f"AUC: {auc:.4f}")
                final_compounds = [(smiles[idx], class_1_probs[i]) for i, idx in enumerate(test_indices) if
                                   class_1_probs[i] >= 0.7]
                sorted_compounds = sorted(final_compounds, key=lambda x: x[1], reverse=True)
                final_df = pd.DataFrame(sorted_compounds, columns=['Compound', 'Probability'])
                generated_file_path = os.path.join(app.config['GENERATED_FILES_DIR'], final_compounds_filename)
                final_df.to_csv(generated_file_path, index=False)

                print("File saved successfully!")
                final_df = pd.read_csv(generated_file_path)
                # Extract the probabilities for clustering
                X = final_df[['Probability']].values

                # Fit the Gaussian Mixture Model
                n_clusters = 3  # you can change this to the desired number of clusters
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                final_df['Cluster'] = gmm.fit_predict(X)
                # Evaluate clustering performance
                silhouette_avg = silhouette_score(X, final_df['Cluster'])
                davies_bouldin = davies_bouldin_score(X, final_df['Cluster'])
                print(f"Silhouette Score: {silhouette_avg:.4f}")
                print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
                # Save final results in a file
                final_clusters_file_path = os.path.join(app.config['GENERATED_FILES_DIR'], final_clusters_filename)
                final_df.to_csv(final_clusters_file_path, index=False)
                print("Final clusters saved successfully!")

                # Extract the probabilities and clusters for plotting
                X = final_df[['Probability']].values
                clusters = final_df['Cluster'].values

                # Create a scatter plot
                plt.figure(figsize=(10, 6))
                for cluster in range(n_clusters):
                    cluster_points = X[clusters == cluster]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 0], label=f"Cluster {cluster}")

                # Plot the centroids
                centroids = gmm.means_
                plt.scatter(centroids[:, 0], centroids[:, 0], c='red', marker='X', label='Centroids')

                # Add labels and legend
                plt.xlabel('Probability')
                plt.ylabel('Probability')
                plt.title('Cluster Plot')
                plt.legend()

                # Save the plot as an image file
                plot_file_path = os.path.join('static', cluster_plot_filename)  # Assuming your static folder is set up correctly
                plt.savefig(plot_file_path)
                plt.close()
                # Perform virtual screening
                file_path = os.path.join(app.config['GENERATED_FILES_DIR'], final_clusters_filename)
                if not os.path.exists(file_path):
                    flash('File "final_clusters.csv" does not exist. Please generate the clusters first.', 'warning')
                else:
                    # Read CSV files into pandas dataframes
                    compounds_df = pd.read_csv(os.path.join(app.config['GENERATED_FILES_DIR'], 'final_compounds.csv'))
                    clusters_df = pd.read_csv(file_path)
                    # Convert dataframes to HTML tables
                    compounds_table = compounds_df.to_html(classes='table table-striped table-bordered', index=False)
                    clusters_table = clusters_df.to_html(classes='table table-striped table-bordered', index=False)

                    virtual_screening = True
                    # Pass data to template
                    return render_template('upload.html', virtual_screening=virtual_screening,
                                           final_clusters_filename=final_clusters_filename,
                                           final_compounds_filename=final_compounds_filename,
                                           compounds_table=compounds_table, clusters_table=clusters_table,
                                           plot_file_path=plot_file_path[len('static/'):],
                                           generated_file_path=generated_file_path,
                                           final_clusters_file_path=final_clusters_file_path)  # Added clusters_table
                    # Add return statement for GET request
    return render_template('upload.html', virtual_screening=virtual_screening,
                           uploaded_file_path=uploaded_file_path,
                           generated_file_path=generated_file_path,
                           generated_molecules=generated_molecules,
                           plot_file_path=plot_file_path)

def allow_files(filename):
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Ensure the static file serving route can handle the new library cluster plot image
@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory('path to/PycharmProjects/pythonProject3/generated_files', filename)
@app.route('/download/sdf_zip')
def download_sdf_zip():
    return send_from_directory(app.config['GENERATED_FILES_DIR'], 'compounds_sdf.zip', as_attachment=True)
def get_compound_name_from_pubchem(smiles_string):
    # URL for the PubChem PUG-REST service
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles_string}/synonyms/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        name = data['InformationList']['Information'][0]['Synonym'][0]
        return name
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

@app.route('/rescreening', methods=['POST'])
def rescreening():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allow_files(file.filename):
                filename = 'New_Library.csv'
                uploaded_file_path = os.path.join(app.config['UPLOADED_FILES_DIR'], filename)
                file.save(uploaded_file_path)
                #flash('CSV file uploaded and saved successfully.', 'success')
                # Load the trained model
                model = GNN(1, 64, 2)
                model.load_state_dict(torch.load("best_model.pth"))
                model.eval()
                # Create a dataset for the new library of compounds
                new_data = pd.read_csv(uploaded_file_path)
                new_smiles = new_data["SMILES"].tolist()
                new_dataset = MoleculeDataset(new_smiles, [0] * len(new_smiles))  # labels are not used in prediction
                # Use the model to predict the drug-like potential for each compound in the library
                new_dataloader = DataLoader(new_dataset, batch_size=32, shuffle=False, collate_fn=collate)

                all_probabilities = []
                for batched_graph, batched_features, _ in new_dataloader:
                    if batched_graph is None:
                        continue

                    with torch.no_grad():
                        outputs = model(batched_graph, batched_features)
                        probabilities = torch.softmax(outputs, dim=1)

                    all_probabilities.extend(probabilities.cpu().numpy())
                # Evaluate the results and perform clustering if needed
                all_probabilities = np.array(all_probabilities)
                class_1_probs = all_probabilities[:, 1]
                # Save final compounds with their respective predicted probabilities
                final_compounds = [(new_smiles[i], class_1_probs[i]) for i in range(len(new_smiles))if class_1_probs[i] > 0.7]
                print(final_compounds)
                sorted_compounds = sorted(final_compounds, key=lambda x: x[1], reverse=True)
                final_df = pd.DataFrame(sorted_compounds, columns=['Compound', 'Probability'])
                generated_file_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'new_library_predictions.csv')
                final_df.to_csv(generated_file_path, index=False)

                # Extract the probabilities for clustering
                X = final_df[['Probability']].values

                # Fit the Gaussian Mixture Model
                n_clusters = 3  # you can change this to the desired number of clusters
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                final_df['Cluster'] = gmm.fit_predict(X)

                # Save final results in a file
                final_clusters_file_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'new_library_clusters.csv')
                final_df.to_csv(final_clusters_file_path, index=False)
                # Convert SMILES to SDF
                compounds_sdf_dir = os.path.join(app.config['GENERATED_FILES_DIR'], 'compounds_sdf')
                if not os.path.exists(compounds_sdf_dir):
                    os.makedirs(compounds_sdf_dir)

                    # Save compounds to SDF with PubChem names
                for index, row in final_df.iterrows():
                    mol = Chem.MolFromSmiles(row['Compound'])
                    compound_name = get_compound_name_from_pubchem(
                        row['Compound']) or f"Compound_{index}"  # Fetch compound name
                    if mol:
                        mol.SetProp("_Name", compound_name)
                        mol.SetProp("Probability", str(row['Probability']))
                        mol.SetProp("Cluster", str(row['Cluster']))

                        # Create a filename from the compound name
                        safe_filename = secure_filename(
                            compound_name)  # Use secure_filename to ensure it's safe for file paths
                        sdf_filename = f"{safe_filename}.sdf" if safe_filename else f"Compound_{index}.sdf"
                        sdf_path = os.path.join(compounds_sdf_dir, sdf_filename)

                        with Chem.SDWriter(sdf_path) as writer:
                            writer.write(mol)

                        # Convert the molecule to SDF format
                        sdf_path = os.path.join(compounds_sdf_dir, f"Compound_{index}.sdf")
                        with Chem.SDWriter(sdf_path) as writer:
                            writer.write(mol)

                # Zip the SDF directory
                sdf_zipfile_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'compounds_sdf.zip')
                with zipfile.ZipFile(sdf_zipfile_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(compounds_sdf_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, arcname=os.path.relpath(file_path, compounds_sdf_dir))

                # Clean up the individual SDF files after zipping by removing the directory
                shutil.rmtree(compounds_sdf_dir)

                # Extract the probabilities and clusters for plotting
                clusters = final_df['Cluster'].values

                # Create a scatter plot
                plt.figure(figsize=(10, 6))
                for cluster in range(n_clusters):
                    cluster_points = X[clusters == cluster]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 0], label=f"Cluster {cluster}")

                # Plot the centroids
                centroids = gmm.means_
                plt.scatter(centroids[:, 0], centroids[:, 0], c='red', marker='X', label='Centroids')

                # Add labels and legend
                plt.xlabel('Probability')
                plt.ylabel('Probability')
                plt.title('Cluster Plot')
                plt.legend()

                # Save the plot as an image file
                new_plot_file_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'new_library_cluster_plot.png')
                plt.savefig(new_plot_file_path)
                plt.close()

                # Convert dataframes to HTML tables
                compound_table = final_df.to_html(classes='table table-striped table-bordered', index=False)
                cluster_table = final_df.to_html(classes='table table-striped table-bordered', index=False)
                #clusters_table = final_df[['Compound', 'Cluster']].to_html(classes='table table-striped table-bordered',
                                                                           #index=False)

                return render_template('upload.html', success=True, compound_table=compound_table,
                                       cluster_table=cluster_table, plot_file_p='new_library_cluster_plot.png',
                                       sdf_zip_file='compounds_sdf.zip')
            else:
                return render_template('upload.html')

def generate_de_novo_molecules(num_molecules, apply_lipinski=True):
    generated_mol = []
    elements = ['C', 'H', 'O', 'N']  # You can expand this list with more elements

    # Correctly obtain the directory path from app.config
    generated_files_dir = app.config['GENERATED_FILES_DIR']

    while len(generated_mol) < num_molecules:
        compound = ''.join(random.choice(elements) for _ in range(5, 20))  # Generate a compound
        mol = Chem.MolFromSmiles(compound)
        if mol is None:
            continue

        activity = 0

        if apply_lipinski:
            molecular_weight = Descriptors.MolWt(mol)
            logP = Descriptors.MolLogP(mol)
            num_h_donors = Descriptors.NumHDonors(mol)
            num_h_acceptors = Descriptors.NumHAcceptors(mol)
            # Check if the molecule meets desired property criteria
            if 150 <= molecular_weight <= 500 and -2 <= logP <= 5 and num_h_donors <= 5 and num_h_acceptors <= 10:
                generated_mol.append((Chem.MolToSmiles(mol, canonical=True), activity))
        else:
            generated_mol.append((Chem.MolToSmiles(mol, canonical=True), activity))

    # Use the corrected directory path
    filename = 'Molecules.csv'
    generated_file_p = os.path.join(generated_files_dir, filename)
    save_data_to_csv(generated_mol, generated_file_p)

    return generated_mol

@app.route('/generate', methods=['POST'])
def generate_molecules():
    num_molecules = int(request.form['num_molecules'])
    apply_lipinski = request.form.get('options') == 'lipinski'
    generated_molecules = generate_de_novo_molecules(num_molecules, apply_lipinski)

    # Assuming save_data_to_csv handles the saving process correctly
    filename = 'Molecules.csv'
    file_path = os.path.join(app.config['GENERATED_FILES_DIR'], filename)
    save_data_to_csv(generated_molecules, file_path)

    return send_file(file_path, as_attachment=True, download_name=filename)


@app.route('/downloads/<filename>')
def downloads(filename):
    directory = app.config['GENERATED_FILES_DIR']
    try:
        return send_from_directory(directory, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404

# Create a logger to capture the output typically sent to Flask's app.logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def perform_protein_refinement(protein_file_path):
    timestamp = int(time.time())
    # Updated file names with timestamp
    stripped_pdb_filename = f'protein_stripped_{timestamp}.pdb'
    fixed_pdb_filename = f'fixed_output_{timestamp}.pdb'
    minimized_pdb_filename = f'minimized_protein_{timestamp}.pdb'
    ramachandran_plot_filename = f'ramachandran_plot_{timestamp}.png'
    sasa_per_residue_plot_filename = f'sasa_per_residue_plot_{timestamp}.png'
    logger.debug(f"Starting protein refinement for: {protein_file_path}")
    traj = md.load(protein_file_path)
    protein = traj.topology.select('protein')
    stripped_traj = traj.atom_slice(protein)
    stripped_traj.save(stripped_pdb_filename)

    fixer = PDBFixer(stripped_pdb_filename)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    with open(fixed_pdb_filename, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    logger.debug("Protein fixed with PDBFixer. Saved to " + fixed_pdb_filename)

    pdb = PDBFile(fixed_pdb_filename)
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    try:
        modeller.addHydrogens(forcefield)
        logger.debug("Added hydrogens to the model.")
    except Exception as e:
        logger.error(f"An error occurred while adding hydrogens: {e}")
        raise

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.004 * picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=500)
    with open(minimized_pdb_filename, 'w') as f:
        state = simulation.context.getState(getPositions=True)
        PDBFile.writeFile(modeller.topology, state.getPositions(), f)
    logger.debug("Minimized protein structure saved to " + minimized_pdb_filename)

    traj = md.load(minimized_pdb_filename)  # Corrected variable name
    # Generate and save Ramachandran plot
    phi, psi = md.compute_phi(traj), md.compute_psi(traj)
    phi_angles, psi_angles = np.rad2deg(md.compute_dihedrals(traj, phi[0])), np.rad2deg(
        md.compute_dihedrals(traj, psi[0]))

    plt.figure(figsize=(8, 6))
    plt.scatter(phi_angles, psi_angles, s=2, c='blue', alpha=0.5)
    # For alpha helices
    plt.fill_betweenx(np.arange(-180, 50, 1), -100, -45, color='orange', alpha=0.25)
    plt.fill_betweenx(np.arange(-100, 180, 1), 45, 100, color='orange', alpha=0.25)

    # For beta sheets
    plt.fill_between(np.arange(-180, 180, 1), 135, 180, color='green', alpha=0.25)
    plt.fill_between(np.arange(-180, 180, 1), -180, -135, color='green', alpha=0.25)

    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel('Phi (φ) angles (degrees)')
    plt.ylabel('Psi (ψ) angles (degrees)')
    plt.title('Ramachandran Plot with Highlighted Secondary Structure Regions')
    plt.grid(True)

    # Annotations for secondary structure types
    plt.text(-75, 150, 'β-sheet', horizontalalignment='center', verticalalignment='center', color='green',
             alpha=0.75)
    plt.text(-60, -60, 'α-helix', horizontalalignment='center', verticalalignment='center', color='orange',
             alpha=0.75)
    plt.text(60, 60, 'α-helix', horizontalalignment='center', verticalalignment='center', color='orange',
             alpha=0.75)
    plt.text(100, -160, 'β-sheet', horizontalalignment='center', verticalalignment='center', color='green',
             alpha=0.75)

    plt.savefig(f'static/{ramachandran_plot_filename}')
    plt.close()
    # Compute SASA and plot average SASA per residue
    sasa = md.shrake_rupley(traj, mode='residue')
    # Plot SASA for each residue
    plt.plot(np.mean(sasa, axis=0))
    plt.title('Average Solvent Accessible Surface Area (SASA) per residue')
    plt.xlabel('Residue')
    plt.ylabel('SASA (nm²)')
    plt.savefig(f'static/{sasa_per_residue_plot_filename}')
    plt.close()

    return {
        'stripped_pdb': stripped_pdb_filename,
        'fixed_pdb': fixed_pdb_filename,
        'minimized_pdb': minimized_pdb_filename,
        'ramachandran_plot': f'static/{ramachandran_plot_filename}',
        'sasa_per_residue_plot': f'static/{sasa_per_residue_plot_filename}'
    }


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdb'}  # Add or remove file extensions as needed.
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/protein_refinement', methods=['GET', 'POST'])
def protein_refinement():
    try:
        if request.method == 'POST':
            # Check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                protein_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(protein_file_path)
                #flash('Protein file uploaded successfully.', 'success')
                perform_protein_refinement(protein_file_path)
                file2 = 'ramachandran_plot.png'
                file3 = 'sasa_per_residue_plot.png'
                result_files = perform_protein_refinement(protein_file_path)
                # Generate download links and visualization data for the protein refinement results
                download_links = {
                    'stripped_protein': url_for('uploa', filename=result_files['stripped_pdb']),
                    'fixed_protein': url_for('uploa', filename=result_files['fixed_pdb']),
                    'minimized_protein': url_for('uploa', filename=result_files['minimized_pdb']),
                    'ramachandran_plot': url_for('static', filename=os.path.basename(result_files['ramachandran_plot'])),
                    'sasa_per_residue_plot': url_for('static', filename=os.path.basename(result_files['sasa_per_residue_plot']))
                }

                return render_template('upload.html', download_links=download_links, random=int(time.time()), active_tab='protein_refinement')
    except Exception as e:
        app.logger.error(f"An error occurred during protein refinement: {str(e)}")
        flash('An error occurred during processing.', 'error')
        return redirect(request.url)
    return render_template('upload.html', active_tab='protein_refinement')
@app.route('/files/<filename>')
def uploa(filename):
    # This sets the directory to your app's root directory
    directory = current_app.root_path
    return send_from_directory(directory, filename)


def allowed_fil(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'zip', 'pdb'}
def convert_sdf_to_pdbqt(sdf_path, output_directory):
    # Function to convert SDF files in a specified directory to PDBQT format
    for root, dirs, files in os.walk(output_directory):
        for file in files:
            if file.endswith(".sdf"):  # Check for .sdf files
                sdf_path = os.path.join(root, file)
                pdbqt_filename = file.replace('.sdf', '.pdbqt')
                pdbqt_path = os.path.join(root, pdbqt_filename)
                # Prepare the obabel command
                obabel_command = [
                    'obabel', sdf_path, '-O', pdbqt_path,
                    '--gen3d', '-h'  # The -h flag adds hydrogens
                ]
                # Run the obabel command
                try:
                    subprocess.run(obabel_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"Conversion successful for {file}")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred while converting {file}: {e.stderr.decode()}")

def convert_protein(protein_pdb_path, protein_pdbqt_path):
    # Function to convert a PDB file to PDBQT
    obabel_command = [
        'obabel', protein_pdb_path, '-xr', '-O', protein_pdbqt_path  # The -xr flag removes residues not recognized by AutoDock
    ]
    try:
        subprocess.run(obabel_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Conversion successful for {protein_pdb_path}")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else 'An error occurred.'
        print(f"An error occurred while converting {protein_pdb_path}: {error_message}")


def clear_workspace(workspace_path):
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)
    os.makedirs(workspace_path)


@app.route('/upload', methods=['POST'])
def upload_files():
    # Generate a unique job ID for this particular user's session or job
    job_id = uuid.uuid4().hex
    job_workspace = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    job_results_dir = os.path.join(app.config['DOCKING_RESULTS_DIR'], job_id)

    # Create directories for the job
    clear_workspace(job_workspace)  # Clear previous data and create new workspace
    clear_workspace(job_results_dir)  # Clear previous results and create new results directory

    # Save uploaded protein and ligand files
    protein_file = request.files.get('protein_file')
    ligand_zip = request.files.get('ligand_zip')

    if protein_file and allowed_fil(protein_file.filename) and ligand_zip and allowed_fil(ligand_zip.filename):
        protein_filename = secure_filename(protein_file.filename)
        ligand_zip_filename = secure_filename(ligand_zip.filename)

        protein_file_path = os.path.join(job_workspace, protein_filename)
        ligand_zip_path = os.path.join(job_workspace, ligand_zip_filename)

        protein_file.save(protein_file_path)
        ligand_zip.save(ligand_zip_path)

        # Unzip ligands and start conversion
        output_directory_path = os.path.join(job_workspace, 'refined_ligands')
        Path(output_directory_path).mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(ligand_zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_directory_path)

        # Convert and dock
        convert_sdf_to_pdbqt(sdf_path=ligand_zip_path, output_directory=output_directory_path)
        protein_pdbqt_path = protein_file_path.replace('.pdb', '.pdbqt')
        convert_protein(protein_file_path, protein_pdbqt_path)
        run_docking(protein_pdbqt_path, output_directory_path, job_results_dir)

        return jsonify({'job_id': job_id, 'message': 'Files uploaded, conversion started, and docking initiated!'})
    else:
        return jsonify({'error': 'Invalid file type or missing files.'}), 400


def run_docking(protein_pdbqt, ligand_directory_path, results_directory_path):
    print("Starting the docking process...")  # Debug print
    for ligand_file in Path(ligand_directory_path).glob('*.pdbqt'):
        ligand_pdbqt = str(ligand_file)
        result_file_path = os.path.join(results_directory_path, ligand_file.stem + '_docked.pdbqt')
        # Extract docking parameters from the form
        center_x = request.form.get('center_x', type=float)
        center_y = request.form.get('center_y', type=float)
        center_z = request.form.get('center_z', type=float)
        size_x = request.form.get('size_x', type=float)
        size_y = request.form.get('size_y', type=float)
        size_z = request.form.get('size_z', type=float)
        exhaustiveness = request.form.get('exhaustiveness', type=int)
        num_modes = request.form.get('num_modes', type=int)
        energy_range = request.form.get('energy_range', type=int)

        # Configuration text for docking
        # Use these parameters in the docking configuration
        config_text = f"""receptor = {protein_pdbqt}
ligand = {ligand_pdbqt}

center_x = {center_x}
center_y = {center_y}
center_z = {center_z}
size_x = {size_x}
size_y = {size_y}
size_z = {size_z}

out = {result_file_path}
exhaustiveness = {exhaustiveness}
num_modes = {num_modes}
energy_range = {energy_range}
    """
        # Write configuration to a file
        config_file_path = os.path.join(results_directory_path, ligand_file.stem + '_config.txt')
        with open(config_file_path, 'w') as config_file:
            config_file.write(config_text)

        # Run Vina with output capture
        vina_command = ['vina', '--config', config_file_path]
        try:
            result = subprocess.run(vina_command, capture_output=True, text=True)
            if result.returncode != 0:  # Check if the command was not successful
                print(f"Error in docking: {result.stderr}")  # Log any errors
            else:
                print(f"Docking completed for {ligand_file.stem}. Output:\n{result.stdout}")  # Log the success output
        except Exception as e:
            print(f"An exception occurred: {e}")  # Log any exceptions
        finally:
            # Clean up the config file after docking
            os.remove(config_file_path)
        # Initialize an empty list to collect docking data
        docking_data = []
        for file_name in Path(results_directory_path).glob('*_docked.pdbqt'):
            with open(file_name, 'r') as file:
                lines = file.readlines()
                # Extract data for all poses
                for line in lines:
                    if line.startswith("REMARK VINA RESULT:"):
                        # Parse out the binding affinity and RMSD
                        parts = line.split()
                        binding_affinity = float(parts[3])  # The fourth item on this line is the affinity
                        rmsd_lb = float(parts[4])  # RMSD lower bound
                        rmsd_ub = float(parts[5])  # RMSD upper bound
                        # Store in the list with the 'file_name' key
                        docking_data.append({
                            'file_name': os.path.basename(file_name),  # Use basename to get the file name only
                            'binding_affinity': binding_affinity,
                            'rmsd_lb': rmsd_lb,
                            'rmsd_ub': rmsd_ub
                        })

        # Check if docking data was collected
        if docking_data:
            # Convert list to DataFrame
            df = pd.DataFrame(docking_data)
            df_second_poses = df.groupby('file_name').nth(1)  # This selects the second pose for each ligand
            df_second_poses['final_rmsd'] = df_second_poses['rmsd_ub'] - df_second_poses['rmsd_lb']
            df_best_poses = df_second_poses
            print(df_best_poses)
            csv_file_path = os.path.join(results_directory_path, 'docking_results.csv')
            df_best_poses.to_csv(csv_file_path, index=False)
        else:
            print("No docking data to process.")

def validate_docking_output(docked_file_path):
    if os.path.exists(docked_file_path) and os.path.getsize(docked_file_path) > 0:
        with open(docked_file_path, 'r') as file:
            for i in range(10):
                line = file.readline()
                if not line:
                    break
                print(line.strip())  # Process line or check if it's as expected
    else:
        print(f"Docked file {docked_file_path} not found or is empty.")
@app.route('/docking', methods=['GET'])
def docking():
    protein_file_path = request.args.get('protein_file_path', type=str)
    protein_pdbqt_path = os.path.join(app.config['UPLOADED_FILES_DIR'], protein_file_path)
    ligand_directory_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'refined_ligands')
    results_directory_path = os.path.join(app.config['DOCKING_RESULTS_DIR'])

    run_docking(protein_pdbqt_path, ligand_directory_path, results_directory_path)
    return jsonify({'message': 'Docking completed!'})

@app.route('/list_docking_results')
def list_docking_results():
    results_files = Path(app.config['DOCKING_RESULTS_DIR']).glob('*_docked.pdbqt')
    results_list = [str(result) for result in results_files if result.is_file() and result.stat().st_size > 0]
    return jsonify(results_list)
@app.route('/results/<filename>')
def download_results(filename):
    results_directory_path = os.path.join(app.config['DOCKING_RESULTS_DIR'])
    return send_from_directory(directory=results_directory_path, filename=filename, as_attachment=True)
@app.route('/analyze_results/<job_id>', methods=['GET'])
def analyze_results(job_id):
    # Directory where the results are stored
    results_directory = os.path.join(app.config['DOCKING_RESULTS_DIR'], job_id)
    filepath = os.path.join(results_directory, 'docking_results.csv')

    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        return send_file(filepath, as_attachment=True)  # Send the file for download
    else:
        return jsonify({'message': 'Results not ready'}), 202


@app.route('/chart_data/<job_id>')  # URL pattern includes job_id
def chart_data(job_id):
    # Construct the file path using the job_id provided in the URL
    job_results_dir = os.path.join(app.config['DOCKING_RESULTS_DIR'], job_id)
    filepath = os.path.join(job_results_dir, 'docking_results.csv')

    if os.path.isfile(filepath):
        df = pd.read_csv(filepath)
        # Create a barplot of binding affinities
        binding_affinities = df['binding_affinity'].tolist()
        file_names = df['file_name'].tolist()
        chart_data = {
            'labels': file_names,
            'datasets': [{
                'label': 'Binding Affinity',
                'data': binding_affinities,
                'backgroundColor': 'rgba(0, 123, 255, 0.5)',
                'borderColor': 'rgba(0, 123, 255, 1)',
                'borderWidth': 1
            }]
        }
        return jsonify(chart_data)
    else:
        return jsonify({'message': 'Results not ready for job ' + job_id}), 202


@app.route('/download_complexes/<job_id>')
def download_complexes(job_id):
    job_results_dir = os.path.join(app.config['DOCKING_RESULTS_DIR'], job_id)

    # Check if the job results directory exists
    if not os.path.isdir(job_results_dir):
        return abort(404, description="Job results not found.")

    # Create a BytesIO object to write the zip file in memory
    zip_in_memory = BytesIO()
    with zipfile.ZipFile(zip_in_memory, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(job_results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, job_results_dir))
    zip_in_memory.seek(0)
    zip_filename = f'{job_id}_results.zip'
    return send_file(zip_in_memory, download_name=zip_filename, as_attachment=True, mimetype='application/zip')


if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOADED_FILES_DIR']):
        os.makedirs(app.config['UPLOADED_FILES_DIR'])
    if not os.path.exists(app.config['GENERATED_FILES_DIR']):
        os.makedirs(app.config['GENERATED_FILES_DIR'])
    if not os.path.exists(app.config['generated_files_dir']):
        os.makedirs(app.config['generated_files_dir'])
    if not os.path.exists(app.config['uploaded_files_dir']):
        os.makedirs(app.config['uploaded_files_dir'])
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['DOCKING_RESULTS_DIR']):
        os.makedirs(app.config['DOCKING_RESULTS_DIR'])
    app.run(debug=True)