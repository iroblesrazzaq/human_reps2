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

# %%
class ConceptPairDataset():
    """
    Class to turn both concept bins - np.ndarrays shape (n_onsets, n_neurons) (each row is a response) into dataset with 

    2 methods - one with psuedopopulations

    
    """
    def __init__(self, patient_data: PatientData, concept_pair: Tuple[str, str], 
                 epoch: str, min_samples: int = 10, neurons=None):
        self.patient_data = patient_data
        self.c1, self.c2 = concept_pair
        self.epoch = epoch
        self.min_samples = min_samples
        self.neurons = neurons
        
        # one optimization I could do - currently instantiating new concept pair dataset class every iteration

    def create_dataset_normal(self, test_size = 0.3):
        """
        Method to create dataset without pseudopops, liable to make unbalanced dataset

        Returns X_train, X_test, y_train, y_test, info: dict
        """
        c1_data, c2_data = self.patient_data.get_concept_data(c1=self.c1, c2=self.c2, epoch=self.epoch, neurons=self.neurons)

        #print(f"c1 shape: {c1_data.shape[0]}, c2 shape: {c2_data.shape[0]}")

        if len(c1_data) < self.min_samples or len(c2_data) < self.min_samples:
            raise ValueError(f"Insufficient samples for {self.c1} vs {self.c2}")

        X = np.vstack([c1_data, c2_data])
        y = np.concatenate([np.zeros(len(c1_data)), np.ones(len(c2_data))])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        info = {} #for any extra stuff we wanna pass through
        res_dict = {
            'X_train': X_train, 'X_test': X_test, 'y_test': y_test, 'y_train': y_train
        }

        return res_dict, info




    def create_dataset_pseudo(self, test_size=0.3, train_size_total=100, test_size_total=50):
        """
        Method to create dataset with pseudopopulations to balance and augment data.
        """
        c1_data, c2_data = self.patient_data.get_concept_data(c1=self.c1, c2=self.c2, epoch=self.epoch, neurons=self.neurons)
        
        #print(f"c1 shape: {c1_data.shape[0]}, c2 shape: {c2_data.shape[0]}")
        
        if len(c1_data) < self.min_samples or len(c2_data) < self.min_samples:
            raise ValueError(f"Insufficient samples for {self.c1} vs {self.c2}")
            
        # Split real data into train and test sets
        c1_train_real, c1_test_real = train_test_split(c1_data, test_size=test_size)
        c2_train_real, c2_test_real = train_test_split(c2_data, test_size=test_size)
        #print(f"c1 real train size: {c1_train_real.shape[0]}")
        
        # Calculate needed pseudo samples
        n_pseudo_train_c1 = max(0, train_size_total - len(c1_train_real))
        n_pseudo_test_c1 = max(0, test_size_total - len(c1_test_real))
        n_pseudo_train_c2 = max(0, train_size_total - len(c2_train_real))
        n_pseudo_test_c2 = max(0, test_size_total - len(c2_test_real))
        
        # Generate pseudopopulations
        X_train_c1 = generate_pseudopopulations(c1_train_real, n_pseudo=n_pseudo_train_c1) if n_pseudo_train_c1 > 0 else np.empty((0, c1_train_real.shape[1]))
        X_test_c1 = generate_pseudopopulations(c1_test_real, n_pseudo=n_pseudo_test_c1) if n_pseudo_test_c1 > 0 else np.empty((0, c1_test_real.shape[1]))
        X_train_c2 = generate_pseudopopulations(c2_train_real, n_pseudo=n_pseudo_train_c2) if n_pseudo_train_c2 > 0 else np.empty((0, c2_train_real.shape[1]))
        X_test_c2 = generate_pseudopopulations(c2_test_real, n_pseudo=n_pseudo_test_c2) if n_pseudo_test_c2 > 0 else np.empty((0, c2_test_real.shape[1]))
        
        
        # Create labels (0 for c1, 1 for c2)
        y_train = np.concatenate([np.zeros(len(X_train_c1)), np.ones(len(X_train_c2))])
        y_test = np.concatenate([np.zeros(len(X_test_c1)), np.ones(len(X_test_c2))])
        
        # Combine features
        X_train = np.vstack([X_train_c1, X_train_c2])
        X_test = np.vstack([X_test_c1, X_test_c2])

        # Shuffle training data
        train_shuffle_idx = np.random.permutation(len(y_train))
        X_train = X_train[train_shuffle_idx]
        y_train = y_train[train_shuffle_idx]
        
        # Shuffle test data
        test_shuffle_idx = np.random.permutation(len(y_test))
        X_test = X_test[test_shuffle_idx]
        y_test = y_test[test_shuffle_idx]
        
        info = {
            'test_size_real_data': test_size,
            'train_size_total_target': train_size_total,
            'test_size_total_target': test_size_total,
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