# %%
# function file
from data_structures import PatientData
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from sklearn.model_selection import train_test_split
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from decoders import generate_pseudopopulations, DecodingResult

# %%
class MultiPatientDataset:
    """
    Class to combine concept data from multiple patients for decoding.
    Similar to ConceptPairDataset but works across multiple patients.
    """
    def __init__(self, patient_data_list: List[PatientData], concept_pair: Tuple[str, str], 
                epoch: str, min_samples: int = 10, neurons_list=None):
        self.patient_data_list = patient_data_list
        self.c1, self.c2 = concept_pair
        self.epoch = epoch
        self.min_samples = min_samples
        
        # If neurons_list is not provided, use all neurons from each patient
        if neurons_list is None:
            self.neurons_list = [p.neurons for p in patient_data_list]
        else:
            self.neurons_list = neurons_list
            
    def create_dataset_normal(self, test_size=0.3):
        """
        Create a combined dataset from multiple patients without pseudopopulations.
        """
        all_c1_data = []
        all_c2_data = []
        
        # Collect data from each patient
        for patient_data, neurons in zip(self.patient_data_list, self.neurons_list):
            try:
                c1_data, c2_data = patient_data.get_concept_data(
                    c1=self.c1, c2=self.c2, epoch=self.epoch, neurons=neurons
                )
                all_c1_data.append(c1_data)
                all_c2_data.append(c2_data)
            except Exception as e:
                print(f"Skipping patient {patient_data.pid} for {self.c1} vs {self.c2}: {e}")
                
        # Ensure we have data from at least one patient
        if not all_c1_data or not all_c2_data:
            raise ValueError(f"No valid data for {self.c1} vs {self.c2} across patients")
        
        # Combine data across patients (along neuron dimension)
        # Each onset is paired with all neurons from all patients
        combined_c1_data = np.hstack([c1_data for c1_data in all_c1_data])
        combined_c2_data = np.hstack([c2_data for c2_data in all_c2_data])
        
        if len(combined_c1_data) < self.min_samples or len(combined_c2_data) < self.min_samples:
            raise ValueError(f"Insufficient samples for {self.c1} vs {self.c2}")
            
        X = np.vstack([combined_c1_data, combined_c2_data])
        y = np.concatenate([np.zeros(len(combined_c1_data)), np.ones(len(combined_c2_data))])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        info = {
            'n_patients': len(self.patient_data_list),
            'patient_ids': [p.pid for p in self.patient_data_list],
            'total_neurons': sum(len(neurons) for neurons in self.neurons_list)
        }
        
        res_dict = {
            'X_train': X_train, 'X_test': X_test, 'y_test': y_test, 'y_train': y_train
        }
        
        return res_dict, info
        
    def create_dataset_pseudo(self, test_size=0.3, train_size_total=100, test_size_total=50):
        """
        Create a combined dataset from multiple patients with pseudopopulations.
        """
        # Similar implementation to create_dataset_normal, but with pseudopopulation generation
        # First collect all data
        all_c1_data = []
        all_c2_data = []
        
        for patient_data, neurons in zip(self.patient_data_list, self.neurons_list):
            try:
                c1_data, c2_data = patient_data.get_concept_data(
                    c1=self.c1, c2=self.c2, epoch=self.epoch, neurons=neurons
                )
                all_c1_data.append(c1_data)
                all_c2_data.append(c2_data)
            except Exception as e:
                print(f"Skipping patient {patient_data.pid} for {self.c1} vs {self.c2}: {e}")
        
        # Ensure we have data
        if not all_c1_data or not all_c2_data:
            raise ValueError(f"No valid data for {self.c1} vs {self.c2} across patients")
            
        # Combine data across patients
        combined_c1_data = np.hstack([c1_data for c1_data in all_c1_data])
        combined_c2_data = np.hstack([c2_data for c2_data in all_c2_data])
        
        # Split into train/test
        c1_train_real, c1_test_real = train_test_split(combined_c1_data, test_size=test_size)
        c2_train_real, c2_test_real = train_test_split(combined_c2_data, test_size=test_size)
        
        # Calculate needed pseudo samples
        n_pseudo_train_c1 = max(0, train_size_total - len(c1_train_real))
        n_pseudo_test_c1 = max(0, test_size_total - len(c1_test_real))
        n_pseudo_train_c2 = max(0, train_size_total - len(c2_train_real))
        n_pseudo_test_c2 = max(0, test_size_total - len(c2_test_real))
        
        # Generate pseudopopulations using your existing function
        X_train_c1 = generate_pseudopopulations(c1_train_real, n_pseudo=n_pseudo_train_c1) if n_pseudo_train_c1 > 0 else c1_train_real
        X_test_c1 = generate_pseudopopulations(c1_test_real, n_pseudo=n_pseudo_test_c1) if n_pseudo_test_c1 > 0 else c1_test_real
        X_train_c2 = generate_pseudopopulations(c2_train_real, n_pseudo=n_pseudo_train_c2) if n_pseudo_train_c2 > 0 else c2_train_real
        X_test_c2 = generate_pseudopopulations(c2_test_real, n_pseudo=n_pseudo_test_c2) if n_pseudo_test_c2 > 0 else c2_test_real
        
        # Create labels and combine
        y_train = np.concatenate([np.zeros(len(X_train_c1)), np.ones(len(X_train_c2))])
        y_test = np.concatenate([np.zeros(len(X_test_c1)), np.ones(len(X_test_c2))])
        
        X_train = np.vstack([X_train_c1, X_train_c2])
        X_test = np.vstack([X_test_c1, X_test_c2])
        
        # Shuffle
        train_shuffle_idx = np.random.permutation(len(y_train))
        X_train = X_train[train_shuffle_idx]
        y_train = y_train[train_shuffle_idx]
        
        test_shuffle_idx = np.random.permutation(len(y_test))
        X_test = X_test[test_shuffle_idx]
        y_test = y_test[test_shuffle_idx]
        
        info = {
            'n_patients': len(self.patient_data_list),
            'patient_ids': [p.pid for p in self.patient_data_list],
            'total_neurons': sum(len(neurons) for neurons in self.neurons_list),
            'n_pseudo_train_c1': n_pseudo_train_c1,
            'n_pseudo_test_c1': n_pseudo_test_c1,
            'n_pseudo_train_c2': n_pseudo_train_c2,
            'n_pseudo_test_c2': n_pseudo_test_c2,
            'n_real_train_c1': len(c1_train_real),
            'n_real_test_c1': len(c1_test_real),
            'n_real_train_c2': len(c2_train_real),
            'n_real_test_c2': len(c2_test_real),
        }
        
        return {'X_train': X_train, 'X_test': X_test, 'y_test': y_test, 'y_train': y_train}, info

# %%
class MultiPatientDecoder:
    """Handles decoding across multiple patients for a single concept pair"""
    
    def __init__(self, patient_data_list: List[PatientData], c1: str, c2: str, epoch: str, 
                 classifier: BaseEstimator = LinearSVC(), standardize: bool=False, 
                 min_samples=20, neurons_list=None):
        self.patient_data_list = patient_data_list
        self.c1 = c1
        self.c2 = c2
        self.epoch = epoch
        self.classifier = classifier
        self.min_samples = min_samples
        self.neurons_list = neurons_list
        
        self.scaler = StandardScaler() if standardize else None
        self.enough_data = True
        
        self.dataset = MultiPatientDataset( # add option to input dataset? not high priority
            patient_data_list=self.patient_data_list,
            concept_pair=(self.c1, self.c2),
            epoch=self.epoch,
            min_samples=self.min_samples,
            neurons_list=self.neurons_list
        )
        
        # Check if we have enough data
        try:
            _, _ = self.dataset.create_dataset_normal()
        except ValueError as e:
            print(f"Skipping concept pair {self.c1}, {self.c2}: {e}")
            self.enough_data = False
            
    def decode_normal(self, test_size=0.3):
        try:
            data_dict, info = self.dataset.create_dataset_normal(test_size=test_size)
        except ValueError as e:
            print(f"Skipping concept pair {self.c1}, {self.c2}: {e}")
            return None
        return self._decode(data_dict=data_dict)
        
    def decode_pseudo(self, train_size_total=200, test_size_total=67, test_size=0.3):
        try:
            data_dict, info = self.dataset.create_dataset_pseudo(
                test_size=test_size, 
                train_size_total=train_size_total, 
                test_size_total=test_size_total
            )
        except ValueError as e:
            print(f"Skipping concept pair {self.c1}, {self.c2}: {e}")
            return None
        return self._decode(data_dict=data_dict)
        
    def _decode(self, data_dict) -> DecodingResult:
        """Same as ConceptDecoder._decode"""
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']

        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        self.classifier.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = self.classifier.predict(X_train)
        y_pred = self.classifier.predict(X_test)

        # Calculate metrics for train and test
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_pred)
        test_roc_auc = roc_auc_score(y_test, y_pred)
        
        train_samples = {
            self.c1: np.sum(y_train == 0),
            self.c2: np.sum(y_train == 1)
        }
        test_samples = {
            self.c1: np.sum(y_test == 0),
            self.c2: np.sum(y_test == 1)
        }

        return DecodingResult(
            train_accuracy=train_accuracy,
            train_roc_auc=train_roc_auc,
            test_accuracy=test_accuracy,
            test_roc_auc=test_roc_auc,
            train_samples=train_samples,
            test_samples=test_samples,
            predictions=y_pred,
            true_labels=y_test,
            classifier=self.classifier,
            data=data_dict
        )
    

# %%
class MultiResultsManager:
    """
    Manages decoding results for multiple patients and multiple concept pairs.
    Similar to SingleResultsManager but for multiple patients.
    """
    def __init__(self, patient_data_list: List[PatientData], concept_pairs: List[Tuple[str, str]], 
                 epoch: str, classifier: BaseEstimator = LinearSVC(), 
                 standardize: bool = False, pseudo=False, neurons_list=None, **pseudo_kwargs):
        self.patient_data_list = patient_data_list
        self.concept_pairs = concept_pairs
        self.epoch = epoch
        self.classifier = classifier
        self.standardize = standardize
        self.results = {}
        self.pseudo = pseudo
        self.pseudo_params = pseudo_kwargs
        self.neurons_list = neurons_list
        
    def run_decoding_for_pairs(self, num_iter: int = 1) -> None:
        """
        Runs decoding for all concept pairs provided in the constructor.
        """
        self.results = {}
        for c1, c2 in tqdm(self.concept_pairs, desc=f"Decoding for {[p.pid for p in self.patient_data_list]}"):
            decoder = MultiPatientDecoder(
                patient_data_list=self.patient_data_list,
                c1=c1,
                c2=c2,
                epoch=self.epoch,
                classifier=self.classifier,
                standardize=self.standardize,
                neurons_list=self.neurons_list
            )
            
            if decoder.enough_data:
                for i in range(num_iter):
                    if self.pseudo:
                        if self.pseudo_params:
                            result = decoder.decode_pseudo(**self.pseudo_params)
                        else:
                            result = decoder.decode_pseudo()
                    else:
                        result = decoder.decode_normal()
                        
                    if result is not None:
                        if (c1, c2) not in self.results:
                            self.results[(c1, c2)] = [result]
                        else:
                            self.results[(c1, c2)].append(result)
                            
    # Use the same plot_train_test_performance_heatmap method as in SingleResultsManager
    def plot_train_test_performance_heatmap(self, metric='test_roc_auc', figsize=(20, 10)):
        """
        Generates and displays a combined heatmap of training and testing performance for all concept pairs.
        For multiple iterations, shows mean performance with standard deviation in parentheses.
        
        Args:
            metric (str): One of 'test_accuracy', 'train_accuracy', 'test_roc_auc', 'train_roc_auc'
            figsize (tuple): Figure size for the plot
        """
        concepts = sorted(list(set([c for pair in self.concept_pairs for c in pair])))
        n_concepts = len(concepts)
        
        # Initialize matrices for means and standard deviations
        train_mean_matrix = np.full((n_concepts, n_concepts), np.nan)
        test_mean_matrix = np.full((n_concepts, n_concepts), np.nan)
        train_std_matrix = np.full((n_concepts, n_concepts), np.nan)
        test_std_matrix = np.full((n_concepts, n_concepts), np.nan)

        concept_to_idx = {concept: i for i, concept in enumerate(concepts)}

        for concept_pair, results in self.results.items():
            if results:  # Check if results exist for this pair
                c1, c2 = concept_pair
                i, j = concept_to_idx[c1], concept_to_idx[c2]
                
                # Extract values for all iterations
                if 'roc_auc' in metric:
                    train_values = [r.train_roc_auc for r in results]
                    test_values = [r.test_roc_auc for r in results]
                else:  # accuracy
                    train_values = [r.train_accuracy for r in results]
                    test_values = [r.test_accuracy for r in results]
                
                # Calculate mean and std
                train_mean = np.mean(train_values)
                test_mean = np.mean(test_values)
                train_std = np.std(train_values)
                test_std = np.std(test_values)
                
                # Fill matrices symmetrically
                for matrix, value in [(train_mean_matrix, train_mean), 
                                    (test_mean_matrix, test_mean),
                                    (train_std_matrix, train_std),
                                    (test_std_matrix, test_std)]:
                    matrix[i, j] = value
                    matrix[j, i] = value

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        def annotate_heatmap(ax, mean_matrix, std_matrix):
            """Helper function to annotate heatmap with mean ± std"""
            for i in range(mean_matrix.shape[0]):
                for j in range(mean_matrix.shape[1]):
                    if not np.isnan(mean_matrix[i, j]):
                        text = f'{mean_matrix[i, j]:.3f}\n(±{std_matrix[i, j]:.3f})'
                        ax.text(j + 0.5, i + 0.5, text,
                            ha='center', va='center',
                            color='white' if mean_matrix[i, j] > 0.5 else 'black',
                            fontsize=8)
                    else:
                        ax.text(j + 0.5, i + 0.5, 'N/A',
                            ha='center', va='center',
                            color='gray')

        # Plot heatmaps
        for ax, mean_matrix, std_matrix, title in [
            (ax1, train_mean_matrix, train_std_matrix, 'Training'),
            (ax2, test_mean_matrix, test_std_matrix, 'Test')
        ]:
            sns.heatmap(mean_matrix, ax=ax,
                    xticklabels=concepts,
                    yticklabels=concepts,
                    cmap='viridis',
                    vmin=0.0,
                    vmax=1.0,
                    center=0.4,
                    annot=False)  # We'll add custom annotations
            
            annotate_heatmap(ax, mean_matrix, std_matrix)
            
            ax.set_title(f'{title} {metric.replace("test_", "").replace("_", " ").title()}', fontsize=15)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle(
            f'Train vs Test Performance for Concept Decoding\nPatients {[p.pid for p in self.patient_data_list]}, Epoch: {self.epoch}\n(mean ± std across {len(next(iter(self.results.values())))} iterations)',
            y=1.05
        )
        plt.tight_layout()
        #plt.show()

        # Return statistics for analysis if needed
        stats = {
            'train_mean': train_mean_matrix,
            'train_std': train_std_matrix,
            'test_mean': test_mean_matrix,
            'test_std': test_std_matrix
        }
        return stats
# %%
def plot_multi_patient_heatmap(results_manager, metric='test_roc_auc', figsize=(20, 10), 
                              vmin=0.0, vmax=1.0, center=0.4, cmap='viridis',
                              return_stats=True, save_path=None, 
                              selected_concepts=None, show_numbers=True):
    """
    Generates a heatmap of decoding performance for a MultiResultsManager.
    
    Args:
        results_manager: An instance of MultiResultsManager with results
        metric (str): One of 'test_accuracy', 'train_accuracy', 'test_roc_auc', 'train_roc_auc'
        figsize (tuple): Figure size for the plot
        vmin, vmax, center: Color scale parameters for the heatmap
        cmap: Colormap for the heatmap
        return_stats (bool): Whether to return statistics dictionary
        save_path (str): Optional path to save the figure
        selected_concepts (list): List of specific concepts to include in heatmap (None = all)
        show_numbers (bool): Whether to display performance numbers on the heatmap
        
    Returns:
        stats (dict): Dictionary with performance matrices if return_stats is True
    """
    # Get all unique concepts from the concept pairs
    all_concepts = sorted(list(set([c for pair in results_manager.concept_pairs for c in pair])))
    
    # Filter concepts if selected_concepts is provided
    if selected_concepts is not None:
        # Ensure all selected concepts are valid
        invalid_concepts = [c for c in selected_concepts if c not in all_concepts]
        if invalid_concepts:
            print(f"Warning: These concepts are not in the results: {invalid_concepts}")
        
        # Only keep concepts that are in both all_concepts and selected_concepts
        concepts = sorted([c for c in all_concepts if c in selected_concepts])
        
        # Ensure we have at least 2 concepts
        if len(concepts) < 2:
            raise ValueError("Need at least 2 valid concepts to create a heatmap")
    else:
        concepts = all_concepts
    
    n_concepts = len(concepts)
    
    # Initialize matrices for means and standard deviations
    train_mean_matrix = np.full((n_concepts, n_concepts), np.nan)
    test_mean_matrix = np.full((n_concepts, n_concepts), np.nan)
    train_std_matrix = np.full((n_concepts, n_concepts), np.nan)
    test_std_matrix = np.full((n_concepts, n_concepts), np.nan)

    concept_to_idx = {concept: i for i, concept in enumerate(concepts)}

    # Fill matrices with results
    for concept_pair, results in results_manager.results.items():
        c1, c2 = concept_pair
        # Skip if either concept is not in our selected list
        if c1 not in concepts or c2 not in concepts:
            continue
            
        if results:  # Check if results exist for this pair
            i, j = concept_to_idx[c1], concept_to_idx[c2]
            
            # Extract values for all iterations
            if 'roc_auc' in metric:
                train_values = [r.train_roc_auc for r in results]
                test_values = [r.test_roc_auc for r in results]
            else:  # accuracy
                train_values = [r.train_accuracy for r in results]
                test_values = [r.test_accuracy for r in results]
            
            # Calculate mean and std
            train_mean = np.mean(train_values)
            test_mean = np.mean(test_values)
            train_std = np.std(train_values)
            test_std = np.std(test_values)
            
            # Fill matrices symmetrically
            for matrix, value in [(train_mean_matrix, train_mean), 
                                (test_mean_matrix, test_mean),
                                (train_std_matrix, train_std),
                                (test_std_matrix, test_std)]:
                matrix[i, j] = value
                matrix[j, i] = value

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    def annotate_heatmap(ax, mean_matrix, std_matrix):
        """Helper function to annotate heatmap with mean ± std"""
        if not show_numbers:
            return  # Skip annotation if show_numbers is False
            
        for i in range(mean_matrix.shape[0]):
            for j in range(mean_matrix.shape[1]):
                if not np.isnan(mean_matrix[i, j]):
                    text = f'{mean_matrix[i, j]:.3f}\n(±{std_matrix[i, j]:.3f})'
                    ax.text(j + 0.5, i + 0.5, text,
                        ha='center', va='center',
                        color='white' if mean_matrix[i, j] > 0.5 else 'black',
                        fontsize=8)
                else:
                    ax.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center',
                        color='gray')

    # Plot heatmaps
    for ax, mean_matrix, std_matrix, title in [
        (ax1, train_mean_matrix, train_std_matrix, 'Training'),
        (ax2, test_mean_matrix, test_std_matrix, 'Test')
    ]:
        sns.heatmap(mean_matrix, ax=ax,
                xticklabels=concepts,
                yticklabels=concepts,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                annot=False)  # We'll add custom annotations
        
        annotate_heatmap(ax, mean_matrix, std_matrix)
        
        ax.set_title(f'{title} {metric.replace("test_", "").replace("_", " ").title()}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Create title with patient information
    if hasattr(results_manager, 'patient_data_list'):
        # Multi-patient case
        patient_ids = [p.pid for p in results_manager.patient_data_list]
        patients_str = ", ".join(patient_ids)
        title = f'Multi-Patient Decoding Performance\nPatients: {patients_str}, Epoch: {results_manager.epoch}'
    else:
        # Single patient case (for compatibility)
        title = f'Train vs Test Performance for Concept Decoding\nPatient {results_manager.patient_data.pid}, Epoch: {results_manager.epoch}'
    
    # Add iteration info if available
    if results_manager.results and next(iter(results_manager.results.values())):
        first_result = next(iter(results_manager.results.values()))
        title += f'\n(mean ± std across {len(first_result)} iterations)'
    
    # Add concept selection info if applicable
    if selected_concepts is not None:
        title += f'\n(Showing {len(concepts)} selected concepts)'
        
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Return statistics if requested
    if return_stats:
        stats = {
            'train_mean': train_mean_matrix,
            'train_std': train_std_matrix,
            'test_mean': test_mean_matrix,
            'test_std': test_std_matrix,
            'concepts': concepts,
            'concept_to_idx': concept_to_idx
        }
        return stats
