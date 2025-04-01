# function file
from data_structures import PatientData
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union
from sklearn.model_selection import train_test_split
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

@dataclass
class DecodingResult:
    """Container for decoding results and metrics"""
    test_accuracy: float
    train_accuracy: float
    test_roc_auc: float
    train_roc_auc: float
    train_samples: Dict[str, int]  # Number of samples for each concept in training
    test_samples: Dict[str, int]   # Number of samples for each concept in testing
    predictions: np.ndarray
    true_labels: np.ndarray
    classifier: BaseEstimator
    data: Dict

def generate_pseudopopulations(
    responses: np.ndarray,
    n_pseudo: int
) -> np.ndarray:
    """
    Generate random pseudopopulation responses and concatenate with original data.
    
    Args:
        responses: Matrix of shape (n_onsets, n_neurons) containing neural responses
        n_pseudo: Number of pseudopopulation responses to generate
        random_state: Random seed for reproducibility
    
    Returns:
        Array of shape (n_onsets + n_pseudo, n_neurons) containing original responses
        concatenated with pseudopopulation responses
    
    Example:
        If responses is shape (5, 10) with 5 onsets and 10 neurons, and n_pseudo=3,
        the output will be shape (8, 10) containing the original 5 responses plus
        3 new pseudopopulation responses.
    """
    n_onsets, n_neurons = responses.shape
    pseudo_responses = np.zeros((n_pseudo, n_neurons))    
    for i in range(n_pseudo): # For each pseudopopulation response 
        for j in range(n_neurons): # For each neuron in psuedoresponse

            random_onset = np.random.randint(0, n_onsets) # using random library instead of np in case we seeded random
            # Use random onset response for this neuron
            pseudo_responses[i, j] = responses[random_onset, j]
    combined_responses = np.vstack([responses, pseudo_responses])
    
    return combined_responses


class ConceptDataset:
    """
    Flexible dataset class for decoding between concepts or concept groups.
    
    Can handle:
    1. Traditional pair decoding (c1 vs c2)
    2. Group decoding ((a,b,c,d) vs (e,f,g,h))
    
    Maintains concept identity tracking for visualization while using binary labels for training.
    """
    def __init__(self, patient_data: PatientData, 
                 concepts_groups: Tuple[Union[str, Tuple[str, ...]], Union[str, Tuple[str, ...]]],
                 epoch: str, min_samples: int = 10, neurons=None):
        """
        Initialize ConceptDataset with either a pair of concepts or a pair of concept groups.
        
        Args:
            patient_data: PatientData object containing neural recordings
            concepts_groups: Either (c1, c2) for pairs or ((a,b,c,d), (e,f,g,h)) for groups
            epoch: Data epoch ('movie', 'preSleep_recall', etc.)
            min_samples: Minimum samples required per concept
            neurons: List of neurons to use (optional)
        """
        self.patient_data = patient_data
        self.epoch = epoch
        self.min_samples = min_samples
        self.neurons = neurons
        
        # Handle both concept pairs and concept groups
        self.group1, self.group2 = self._normalize_concepts_input(concepts_groups)
        
        # Flag to track if we're in pair mode or group mode
        self.is_pair_mode = len(self.group1) == 1 and len(self.group2) == 1
        
    def _normalize_concepts_input(self, concepts_groups):
        """Convert input to standardized format of (tuple, tuple)"""
        group1, group2 = concepts_groups
        
        # Convert single concepts to tuples
        if isinstance(group1, str):
            group1 = (group1,)
        if isinstance(group2, str):
            group2 = (group2,)
            
        return group1, group2
    
    def _get_exclusive_onsets_for_concept(self, concept, other_concepts):
        """
        Get onsets where a concept appears and ALL other_concepts are absent.
        
        Args:
            concept: Target concept to find onsets for
            other_concepts: Concepts that should all be absent
            
        Returns:
            Array of onset times
        """
        # Start with all onsets for this concept against first opposing concept
        if not other_concepts:
            raise ValueError("Need at least one concept in other_concepts")
            
        # Get initial onsets against first opposing concept
        try:
            current_onsets = set(self.patient_data.exclusive_movie_times(
                c1=concept, c2=other_concepts[0]
            ))
        except ValueError:
            return np.array([])  # Return empty if insufficient data
        
        # Filter by exclusivity against each remaining opposing concept
        for other_concept in other_concepts[1:]:
            try:
                # Get onsets where this concept is also excluded
                more_onsets = set(self.patient_data.exclusive_movie_times(
                    c1=concept, c2=other_concept
                ))
                # Keep only onsets that exclude all concepts so far
                current_onsets &= more_onsets
            except ValueError:
                return np.array([])  # Return empty if insufficient data
        
        return np.array(list(current_onsets))
    
    def _gather_data_for_group(self, group, other_group):
        """
        Gather data for all concepts in one group, ensuring exclusivity with other group.
        
        Returns:
            all_data: Combined neural data for the group
            concept_ids: List identifying which concept each sample came from
        """
        all_data = []
        concept_ids = []
        
        for concept in group:
            # Get onsets where this concept appears and ALL concepts from other_group are absent
            onset_times = self._get_exclusive_onsets_for_concept(concept, other_group)
            
            if len(onset_times) < self.min_samples:
                print(f"Warning: Insufficient exclusive onsets for {concept} vs {other_group}")
                continue
                
            # Create neural response bins for these onset times
            concept_data = self.patient_data._bin_times(
                times=list(onset_times), 
                neurons=self.neurons or self.patient_data.neurons
            )
            
            # Add data and track concept identity
            if len(concept_data) >= self.min_samples:
                all_data.append(concept_data)
                concept_ids.extend([concept] * len(concept_data))
        
        if not all_data:
            raise ValueError(f"Insufficient data for any concept in {group} vs {other_group}")
            
        return np.vstack(all_data), concept_ids
    
    def create_dataset_normal(self, test_size=0.3):
        """
        Create dataset without pseudopopulations.
        
        Returns:
            data_dict: Dictionary containing X_train, X_test, y_train, y_test
            info: Dictionary with additional information including concept tracking
        """
        # Handle pair mode using the original implementation
        if self.is_pair_mode:
            c1, c2 = self.group1[0], self.group2[0]
            try:
                c1_data, c2_data = self.patient_data.get_concept_data(
                    c1=c1, c2=c2, epoch=self.epoch, neurons=self.neurons
                )
                
                if len(c1_data) < self.min_samples or len(c2_data) < self.min_samples:
                    raise ValueError(f"Insufficient samples for {c1} vs {c2}")
                    
                X = np.vstack([c1_data, c2_data])
                y = np.concatenate([np.zeros(len(c1_data)), np.ones(len(c2_data))])
                
                # Track concept identity
                concept_ids = [c1] * len(c1_data) + [c2] * len(c2_data)
            
            except ValueError as e:
                raise ValueError(f"Error in pair mode: {e}")
                
        # Handle group mode
        else:
            try:
                # Gather data for group 1 (concepts labeled 0)
                group1_data, group1_concept_ids = self._gather_data_for_group(
                    self.group1, self.group2
                )
                
                # Gather data for group 2 (concepts labeled 1)
                group2_data, group2_concept_ids = self._gather_data_for_group(
                    self.group2, self.group1
                )
                
                X = np.vstack([group1_data, group2_data])
                y = np.concatenate([np.zeros(len(group1_data)), np.ones(len(group2_data))])
                
                # Track concept identity
                concept_ids = group1_concept_ids + group2_concept_ids
                
            except ValueError as e:
                raise ValueError(f"Error in group mode: {e}")
        
        # Create stratified train/test split
        # We split by group (y), but we also track concept_ids
        # sklearn's train_test_split can handle multiple arrays
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, np.array(concept_ids), test_size=test_size, stratify=y
        )
        
        # Return results
        info = {
            'concept_ids_train': ids_train,
            'concept_ids_test': ids_test,
            'group1': self.group1,
            'group2': self.group2
        }
        
        data_dict = {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test
        }
        
        return data_dict, info
    
    def create_dataset_pseudo(self, test_size=0.3, train_size_total=100, test_size_total=50):
        """
        Create dataset with pseudopopulations to balance and augment data.
        
        For group mode, train_size_total and test_size_total are per group,
        distributed evenly among concepts in the group.
        
        Returns:
            data_dict: Dictionary containing X_train, X_test, y_train, y_test
            info: Dictionary with additional information including concept tracking
        """
        # First separate data by concepts, following pair and group modes
        concept_data_dict = {}  # Will hold data for each concept
        
        if self.is_pair_mode:
            # Get data for the single concepts in pair mode
            c1, c2 = self.group1[0], self.group2[0]
            try:
                c1_data, c2_data = self.patient_data.get_concept_data(
                    c1=c1, c2=c2, epoch=self.epoch, neurons=self.neurons
                )
                concept_data_dict[c1] = c1_data
                concept_data_dict[c2] = c2_data
            except ValueError as e:
                raise ValueError(f"Error in pair mode: {e}")
                
        else:
            # Get data for all concepts in both groups
            for group, other_group in [(self.group1, self.group2), (self.group2, self.group1)]:
                for concept in group:
                    onset_times = self._get_exclusive_onsets_for_concept(concept, other_group)
                    if len(onset_times) < self.min_samples:
                        print(f"Warning: Insufficient exclusive onsets for {concept} vs {other_group}")
                        continue
                        
                    concept_data = self.patient_data._bin_times(
                        times=list(onset_times), 
                        neurons=self.neurons or self.patient_data.neurons
                    )
                    
                    if len(concept_data) >= self.min_samples:
                        concept_data_dict[concept] = concept_data
        
        # Check if we have enough concepts with enough data
        if not concept_data_dict:
            raise ValueError("No concepts with sufficient data")
            
        if self.is_pair_mode and (len(concept_data_dict) < 2 or 
                                 self.group1[0] not in concept_data_dict or 
                                 self.group2[0] not in concept_data_dict):
            raise ValueError(f"Insufficient data for pair {self.group1[0]} vs {self.group2[0]}")
            
        # Now we have all concept data, proceed with train/test splits and pseudopopulations
        all_train_data = []
        all_test_data = []
        all_train_labels = []
        all_test_labels = []
        all_train_concept_ids = []
        all_test_concept_ids = []
        
        # Process group 1 concepts
        valid_group1_concepts = [c for c in self.group1 if c in concept_data_dict]
        if valid_group1_concepts:
            # Calculate per-concept targets for group 1
            train_size_per_concept = train_size_total // len(valid_group1_concepts)
            test_size_per_concept = test_size_total // len(valid_group1_concepts)
            
            for concept in valid_group1_concepts:
                data = concept_data_dict[concept]
                # Split data for this concept
                concept_train, concept_test = train_test_split(data, test_size=test_size)
                
                # Calculate needed pseudopopulations
                n_pseudo_train = max(0, train_size_per_concept - len(concept_train))
                n_pseudo_test = max(0, test_size_per_concept - len(concept_test))
                
                # Generate pseudopopulations for this concept
                if n_pseudo_train > 0:
                    concept_train = generate_pseudopopulations(concept_train, n_pseudo=n_pseudo_train)
                if n_pseudo_test > 0:
                    concept_test = generate_pseudopopulations(concept_test, n_pseudo=n_pseudo_test)
                
                # Add to combined arrays
                all_train_data.append(concept_train)
                all_test_data.append(concept_test)
                all_train_labels.extend([0] * len(concept_train))  # Group 1 = 0
                all_test_labels.extend([0] * len(concept_test))    # Group 1 = 0
                all_train_concept_ids.extend([concept] * len(concept_train))
                all_test_concept_ids.extend([concept] * len(concept_test))
        else:
            raise ValueError(f"No valid concepts with sufficient data in group 1: {self.group1}")
            
        # Process group 2 concepts
        valid_group2_concepts = [c for c in self.group2 if c in concept_data_dict]
        if valid_group2_concepts:
            # Calculate per-concept targets for group 2
            train_size_per_concept = train_size_total // len(valid_group2_concepts)
            test_size_per_concept = test_size_total // len(valid_group2_concepts)
            
            for concept in valid_group2_concepts:
                data = concept_data_dict[concept]
                # Split data for this concept
                concept_train, concept_test = train_test_split(data, test_size=test_size)
                
                # Calculate needed pseudopopulations
                n_pseudo_train = max(0, train_size_per_concept - len(concept_train))
                n_pseudo_test = max(0, test_size_per_concept - len(concept_test))
                
                # Generate pseudopopulations for this concept
                if n_pseudo_train > 0:
                    concept_train = generate_pseudopopulations(concept_train, n_pseudo=n_pseudo_train)
                if n_pseudo_test > 0:
                    concept_test = generate_pseudopopulations(concept_test, n_pseudo=n_pseudo_test)
                
                # Add to combined arrays
                all_train_data.append(concept_train)
                all_test_data.append(concept_test)
                all_train_labels.extend([1] * len(concept_train))  # Group 2 = 1
                all_test_labels.extend([1] * len(concept_test))    # Group 2 = 1
                all_train_concept_ids.extend([concept] * len(concept_train))
                all_test_concept_ids.extend([concept] * len(concept_test))
        else:
            raise ValueError(f"No valid concepts with sufficient data in group 2: {self.group2}")
        
        # Combine and shuffle
        X_train = np.vstack(all_train_data)
        X_test = np.vstack(all_test_data)
        y_train = np.array(all_train_labels)
        y_test = np.array(all_test_labels)
        concept_ids_train = np.array(all_train_concept_ids)
        concept_ids_test = np.array(all_test_concept_ids)
        
        # Shuffle training data
        train_shuffle_idx = np.random.permutation(len(y_train))
        X_train = X_train[train_shuffle_idx]
        y_train = y_train[train_shuffle_idx]
        concept_ids_train = concept_ids_train[train_shuffle_idx]
        
        # Shuffle test data
        test_shuffle_idx = np.random.permutation(len(y_test))
        X_test = X_test[test_shuffle_idx]
        y_test = y_test[test_shuffle_idx]
        concept_ids_test = concept_ids_test[test_shuffle_idx]
        
        # Create info dictionary with details about the dataset
        info = {
            'concept_ids_train': concept_ids_train,
            'concept_ids_test': concept_ids_test,
            'group1': self.group1,
            'group2': self.group2,
            'valid_group1_concepts': valid_group1_concepts,
            'valid_group2_concepts': valid_group2_concepts,
            'real_samples_per_concept': {c: len(concept_data_dict.get(c, [])) for c in 
                                         list(self.group1) + list(self.group2)},
        }
        
        data_dict = {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test
        }
        
        return data_dict, info



class ConceptDecoder:
    
    """
    Handles decoding for a single concept pair

    - design choice - will call dataset method in each decoder call?

    - problem is that generally we want an object associated with one dataset - this would require an input for pseudo or not
    in the class instantiation. however, we want fine control over pseudopops parameters, so this is less good.
    one potential solution is a params* dict, but thats complicated. 

    for consistent stuff

    maybe separate classes - instantiate dataset, get training into dict, input training dict into concept decoder?

    - add method for PCA visualization in 2D/3D
    """
    def __init__(self, patient_data: PatientData, c1: str, c2: str, epoch: str, classifier: BaseEstimator = LinearSVC(), 
                 dataset: ConceptPairDataset = None, standardize: bool=False, min_samples=20, neurons=None):
        self.patient_data = patient_data
        self.c1 = c1
        self.c2 = c2
        self.epoch = epoch
        self.classifier = classifier
        self.min_samples=min_samples
        self.neurons=neurons

        self.scaler = StandardScaler() if standardize else None
        self.metrics = {}
        self.enough_data = True # for efficiency


    
        if not dataset:
            self.dataset = ConceptPairDataset( #type: ignore
                patient_data=self.patient_data,
                concept_pair=(self.c1, self.c2),
                epoch=self.epoch, 
                min_samples = self.min_samples,
                neurons=self.neurons
            )
        else: # if we're passed a dataset
            self.dataset = dataset

        try:
            _, _ = self.dataset.create_dataset_normal()
        except ValueError as e:
            #print(f"Skipping concept pair {self.c1}, {self.c2}: {e}") # Inform user of skipped pair and reason
            self.enough_data = False
        
    def decode_normal(self, test_size=0.3):
        try:
            data_dict, info = self.dataset.create_dataset_normal(test_size=test_size)
        except ValueError as e:
            #print(f"Skipping concept pair {self.c1}, {self.c2}: {e}") # Inform user of skipped pair and reason
            return None # Return None to indicate decoding failure for this pair
        return self._decode(data_dict=data_dict)


    def decode_pseudo(self, train_size_total=300, test_size_total=100, test_size=0.3):
        try:
            data_dict, info = self.dataset.create_dataset_pseudo(test_size=test_size, train_size_total=train_size_total, test_size_total=test_size_total)
        except ValueError as e:
            #print(f"Skipping concept pair {self.c1}, {self.c2}: {e}") # Inform user of skipped pair and reason
            return None # Return None to indicate decoding failure for this pair
        return self._decode(data_dict=data_dict)



    def _decode(self, data_dict) -> DecodingResult: 
        """
        Performs decoding on the concept pair using normal dataset
        
        Args:
            test_size: Fraction of data to use for testing
            
        Returns:
            DecodingResult containing metrics and predictions
        """

        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        #print(f"split sizes: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")


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
        train_roc_auc = roc_auc_score(y_train, y_train_pred) # or use decision_function for prob based ROC AUC if needed
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
class SingleResultsManager:
    """
    Manages decoding results for a single patient and multiple concept pairs for a single patient and epoch.
    """
    def __init__(self, patient_data: PatientData, concept_pairs: List[Tuple[str, str]], 
                 epoch: str, classifier: BaseEstimator = LinearSVC(), 
                 standardize: bool = False, pseudo=False, neurons=None, **pseudo_kwargs):
        self.patient_data = patient_data
        self.concept_pairs = concept_pairs
        self.epoch = epoch
        self.classifier = classifier # Default classifier for all decoders, can be overridden
        self.standardize = standardize # Default standardization for all decoders
        self.results: Dict[List[Tuple[str, str]], DecodingResult] = {} # Store results here, key is concept pair
        self.pseudo = pseudo
        self.pseudo_params = pseudo_kwargs
        self.neurons = neurons

    def run_decoding_for_pairs(self, num_iter: int = 1) -> None:
        """
        Runs decoding for all concept pairs provided in the constructor.
        Stores the DecodingResult in the self.results dictionary.
        """
        self.results = {} # reset results every time
        for c1, c2 in tqdm(self.concept_pairs, desc=f"Decoding for {self.patient_data.pid}"):
            decoder = ConceptDecoder(
                    patient_data=self.patient_data,
                    c1=c1,
                    c2=c2,
                    epoch=self.epoch,
                    classifier=self.classifier,
                    standardize=self.standardize,
                    neurons=self.neurons
                )
            if decoder.enough_data: # efficient way of not rechecking for sufficient samples
                for i in range(num_iter):
                    #print(f"concept decoding: {c1} vs {c2}, iteration #{i}")
                    if self.pseudo:
                        if self.pseudo_params:
                            result = decoder.decode_pseudo(**self.pseudo_params)
                        else:
                            result = decoder.decode_pseudo()
                    else:
                        result = decoder.decode_normal()
                    if result is not None: # Only store if decode was successful (not None)
                        if (c1, c2) not in self.results:
                            self.results[(c1, c2)] = [result]
                        else:
                                self.results[(c1, c2)].append(result)


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
            
            ax.set_title(f'{title} {metric.replace("test_", "").replace("_", " ").title()}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle(
            f'Train vs Test Performance for Concept Decoding\nPatient {self.patient_data.pid}, Epoch: {self.epoch}\n(mean ± std across {len(next(iter(self.results.values())))} iterations)',
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
