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
from collections import defaultdict
import random

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
        MODIFIED to return onset times.

        Returns:
            all_data: Combined neural data for the group
            concept_ids: List identifying which concept each sample came from
            all_onset_times: List of onset times corresponding to each sample
        """
        all_data = []
        concept_ids = []
        all_onset_times = [] # <<< Added

        for concept in group:
            # Assume _get_exclusive_onsets returns times
            onset_times = self._get_exclusive_onsets_for_concept(concept, other_group)

            if len(onset_times) < self.min_samples:
                print(f"Warning: Insufficient exclusive onsets for {concept} vs {other_group}")
                continue

            concept_data = self.patient_data._bin_times(
                times=list(onset_times),
                neurons=self.neurons or self.patient_data.neurons
            )

            if len(concept_data) >= self.min_samples:
                all_data.append(concept_data)
                concept_ids.extend([concept] * len(concept_data))
                all_onset_times.extend(list(onset_times)) # <<< Added: store the times

        if not all_data:
            raise ValueError(f"Insufficient data for any concept in {group} vs {other_group}")


        return np.vstack(all_data), concept_ids, np.array(all_onset_times)
    
    
    def _temporal_block_split(self, X, y, onset_times, test_size, T_separation, concept_ids=None):
        """
        Splits data into train/test sets ensuring temporal separation between sets.

        Args:
            X (np.ndarray): Feature matrix (samples x neurons).
            y (np.ndarray): Label vector (0 or 1).
            onset_times (np.ndarray): Onset time for each sample in X.
            test_size (float): Approximate fraction of samples for the test set.
            T_separation (float): Minimum time gap (in seconds) required between
                                   any train sample onset and any test sample onset.
            concept_ids (np.ndarray, optional): Specific concept IDs for each sample.

        Returns:
            tuple: (train_indices, test_indices) - Lists of original indices.
        """
        if len(onset_times) == 0:
             return np.array([], dtype=int), np.array([], dtype=int)

        n_samples = len(onset_times)
        original_indices = np.arange(n_samples)

        # 1. Sort onsets and keep track of original indices
        sorted_indices = np.argsort(onset_times)
        sorted_onsets = onset_times[sorted_indices]
        sorted_original_indices = original_indices[sorted_indices] # Map position in sorted list back to original index

        # 2. Identify Temporal Blocks (Connected Components)
        adj = defaultdict(list)
        # Build adjacency list based on T_separation violation
        # More efficient than N^2: only check nearby points in sorted list
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # If onset difference is less than T_separation, they are connected
                if sorted_onsets[j] - sorted_onsets[i] < T_separation:
                    adj[i].append(j)
                    adj[j].append(i)
                else:
                    # Since the list is sorted, no further 'j' will be close enough to 'i'
                    break

        # Find connected components (temporal blocks) using DFS/BFS
        visited = set()
        blocks = [] # List of blocks, each block is a list of *sorted* indices
        for i in range(n_samples):
            if i not in visited:
                component = []
                q = [i] # Use list as a queue/stack for BFS/DFS
                visited.add(i)
                while q:
                    u = q.pop(0) # BFS style
                    component.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                blocks.append(component) # Add the found component/block

        # Map block indices from sorted positions back to original indices
        original_indices_blocks = []
        for block in blocks:
            original_indices_blocks.append(list(sorted_original_indices[block]))

        # 3. Randomly Assign Blocks to Train/Test
        n_test_target = int(n_samples * test_size)
        block_sizes = [len(block) for block in original_indices_blocks]
        block_indices = list(range(len(original_indices_blocks)))

        random.shuffle(block_indices) # Shuffle the order of blocks

        test_indices_list = []
        train_indices_list = []
        current_test_size = 0

        for i in block_indices:
            block_original_idxs = original_indices_blocks[i]
            # Assign to test set if we haven't met the target size yet
            if current_test_size < n_test_target:
                test_indices_list.extend(block_original_idxs)
                current_test_size += len(block_original_idxs)
            else:
                # Assign to train set otherwise
                train_indices_list.extend(block_original_idxs)

        # Handle edge case: if the first N blocks overshoot test_size significantly,
        # it might be better to assign the last added block to train instead.
        # (Could add more sophisticated balancing later if needed)

        return np.array(train_indices_list, dtype=int), np.array(test_indices_list, dtype=int)

    # --- Modifications needed in data gathering ---
    # Assume patient_data.get_concept_data and _gather_data_for_group now return
    # a tuple like: (neural_data_matrix, concept_ids_or_labels, onset_times_vector)

    def create_dataset_normal(self, test_size=0.3, T_separation=21.0):
        """
        Create dataset without pseudopopulations, using temporal block split.
        """
        all_X_list = []
        all_y_list = []
        all_onsets_list = []
        all_concept_ids_list = [] # Only used in group mode

        if self.is_pair_mode:
            c1, c2 = self.group1[0], self.group2[0]
            try:
                # Assuming get_concept_data returns (data, labels, times) format
                # Needs adaptation if get_concept_data structure is different
                # Simplified: assume it returns c1_data, c2_data which are (samples, neurons)
                # We need the *times* associated with those samples.
                # Let's reconstruct:
                c1_onset_times = self.patient_data.exclusive_movie_times(c1=c1, c2=c2)
                c2_onset_times = self.patient_data.exclusive_movie_times(c1=c2, c2=c1)

                if len(c1_onset_times) < self.min_samples or len(c2_onset_times) < self.min_samples:
                    raise ValueError(f"Insufficient samples for {c1} vs {c2}")

                c1_data = self.patient_data._bin_times(list(c1_onset_times), self.neurons or self.patient_data.neurons)
                c2_data = self.patient_data._bin_times(list(c2_onset_times), self.neurons or self.patient_data.neurons)

                X = np.vstack([c1_data, c2_data])
                y = np.concatenate([np.zeros(len(c1_data)), np.ones(len(c2_data))])
                onset_times = np.concatenate([c1_onset_times, c2_onset_times])
                concept_ids = np.array([c1] * len(c1_data) + [c2] * len(c2_data)) # Track concepts

            except ValueError as e:
                raise ValueError(f"Error getting pair data: {e}")
        else:
            # Group mode - Use _gather_data_for_group
            try:
                # Gather data for group 1 (concepts labeled 0)
                group1_data, group1_concept_ids, group1_onsets = self._gather_data_for_group(
                    self.group1, self.group2
                )
                all_X_list.append(group1_data)
                all_y_list.append(np.zeros(len(group1_data)))
                all_onsets_list.append(group1_onsets)
                all_concept_ids_list.extend(group1_concept_ids)

                # Gather data for group 2 (concepts labeled 1)
                group2_data, group2_concept_ids, group2_onsets = self._gather_data_for_group(
                    self.group2, self.group1
                )
                all_X_list.append(group2_data)
                all_y_list.append(np.ones(len(group2_data)))
                all_onsets_list.append(group2_onsets)
                all_concept_ids_list.extend(group2_concept_ids)

                X = np.vstack(all_X_list)
                y = np.concatenate(all_y_list)
                onset_times = np.concatenate(all_onsets_list)
                concept_ids = np.array(all_concept_ids_list) # Track concepts

            except ValueError as e:
                raise ValueError(f"Error getting group data: {e}")

        # Perform the temporal split on the combined real data
        train_indices, test_indices = self._temporal_block_split(
            X, y, onset_times, test_size=test_size, T_separation=T_separation
        )

        # Create the final splits
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Prepare info dict, including concept IDs if available
        info = {
            'group1': self.group1,
            'group2': self.group2,
            'train_indices_original': train_indices, # Optional: for debugging
            'test_indices_original': test_indices   # Optional: for debugging
        }
        if 'concept_ids' in locals() and concept_ids is not None:
             info['concept_ids_train'] = concept_ids[train_indices]
             info['concept_ids_test'] = concept_ids[test_indices]


        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        return data_dict, info


    def create_dataset_pseudo(self, test_size=0.3, train_size_total=100, test_size_total=50, T_separation=21.0):
        """
        Create dataset with pseudopopulations, using temporal block split on real data first.
        """
        # 1. Gather all *real* data first (similar to start of create_dataset_normal)
        real_X_list = []
        real_y_list = []
        real_onsets_list = []
        real_concept_ids_list = [] # Store specific concept ID for each real sample

        # This part needs careful implementation based on how data is retrieved.
        # Let's assume we can get data per concept with associated times.
        concept_real_data = {} # Dict: concept_name -> (data_matrix, onset_times_vector)
        all_concepts_in_groups = list(self.group1) + list(self.group2)

        for concept in all_concepts_in_groups:
             # Determine which concepts this one needs to be exclusive against
             other_group = self.group2 if concept in self.group1 else self.group1
             try:
                 # Get onsets exclusive to the other group
                 onset_times = self._get_exclusive_onsets_for_concept(concept, other_group)
                 if len(onset_times) >= self.min_samples:
                     data = self.patient_data._bin_times(list(onset_times), self.neurons or self.patient_data.neurons)
                     if len(data) >= self.min_samples:
                          concept_real_data[concept] = (data, onset_times)
                 else:
                     print(f"Warning: Insufficient real onsets for {concept} during initial gathering.")
             except ValueError as e:
                 print(f"Warning: Error gathering real data for {concept}: {e}")
                 continue # Skip this concept if error

        # Combine into single arrays for splitting
        for concept, (data, times) in concept_real_data.items():
             real_X_list.append(data)
             label = 0 if concept in self.group1 else 1
             real_y_list.append(np.full(len(data), label))
             real_onsets_list.append(times)
             real_concept_ids_list.extend([concept] * len(data))

        if not real_X_list:
            raise ValueError("Insufficient real data across all concepts to proceed.")

        real_X = np.vstack(real_X_list)
        real_y = np.concatenate(real_y_list)
        real_onsets = np.concatenate(real_onsets_list)
        real_concept_ids = np.array(real_concept_ids_list)

        # 2. Perform temporal split on the *real* data
        train_indices, test_indices = self._temporal_block_split(
            real_X, real_y, real_onsets, test_size=test_size, T_separation=T_separation
        )

        # 3. Partition the real data based on the split
        real_X_train, real_X_test = real_X[train_indices], real_X[test_indices]
        real_y_train, real_y_test = real_y[train_indices], real_y[test_indices]
        real_concept_ids_train = real_concept_ids[train_indices]
        real_concept_ids_test = real_concept_ids[test_indices]

        # 4. Generate Pseudopopulations *Separately* for Train and Test
        final_X_train_list, final_y_train_list, final_concept_ids_train_list = [], [], []
        final_X_test_list, final_y_test_list, final_concept_ids_test_list = [], [], []

        # Determine target samples per group (adjust based on valid concepts)
        valid_group1_concepts = [c for c in self.group1 if c in concept_real_data]
        valid_group2_concepts = [c for c in self.group2 if c in concept_real_data]

        train_size_per_concept_g1 = train_size_total // len(valid_group1_concepts) if valid_group1_concepts else 0
        test_size_per_concept_g1 = test_size_total // len(valid_group1_concepts) if valid_group1_concepts else 0
        train_size_per_concept_g2 = train_size_total // len(valid_group2_concepts) if valid_group2_concepts else 0
        test_size_per_concept_g2 = test_size_total // len(valid_group2_concepts) if valid_group2_concepts else 0

        # Process Training Set Pseudos
        for concept in concept_real_data.keys(): # Iterate through concepts with real data
            is_group1 = concept in self.group1
            target_train_size = train_size_per_concept_g1 if is_group1 else train_size_per_concept_g2

            # Find real samples of this concept *in the training set*
            mask_train = (real_concept_ids_train == concept)
            concept_real_train_data = real_X_train[mask_train]

            if len(concept_real_train_data) == 0:
                 print(f"Warning: No real samples for concept '{concept}' landed in the training set after temporal split.")
                 # Optionally, add the existing real samples if any, even if 0
                 final_X_train_list.append(concept_real_train_data) # Append empty or few samples
                 final_y_train_list.append(real_y_train[mask_train])
                 final_concept_ids_train_list.extend(list(real_concept_ids_train[mask_train]))
                 continue

            n_pseudo_train = max(0, target_train_size - len(concept_real_train_data))

            if n_pseudo_train > 0:
                 concept_train_combined = generate_pseudopopulations(concept_real_train_data, n_pseudo=n_pseudo_train)
            else:
                 # If we have enough or too many real samples, just use them (or potentially subsample later if strict size needed)
                 concept_train_combined = concept_real_train_data[:target_train_size] # Simple truncation if over target

            final_X_train_list.append(concept_train_combined)
            label = 0 if is_group1 else 1
            final_y_train_list.append(np.full(len(concept_train_combined), label))
            final_concept_ids_train_list.extend([concept] * len(concept_train_combined))

        # Process Testing Set Pseudos (similarly)
        for concept in concept_real_data.keys():
            is_group1 = concept in self.group1
            target_test_size = test_size_per_concept_g1 if is_group1 else test_size_per_concept_g2

            mask_test = (real_concept_ids_test == concept)
            concept_real_test_data = real_X_test[mask_test]

            if len(concept_real_test_data) == 0:
                 print(f"Warning: No real samples for concept '{concept}' landed in the test set after temporal split.")
                 final_X_test_list.append(concept_real_test_data)
                 final_y_test_list.append(real_y_test[mask_test])
                 final_concept_ids_test_list.extend(list(real_concept_ids_test[mask_test]))
                 continue

            n_pseudo_test = max(0, target_test_size - len(concept_real_test_data))

            if n_pseudo_test > 0:
                 concept_test_combined = generate_pseudopopulations(concept_real_test_data, n_pseudo=n_pseudo_test)
            else:
                 concept_test_combined = concept_real_test_data[:target_test_size]

            final_X_test_list.append(concept_test_combined)
            label = 0 if is_group1 else 1
            final_y_test_list.append(np.full(len(concept_test_combined), label))
            final_concept_ids_test_list.extend([concept] * len(concept_test_combined))

        # 5. Combine final lists
        if not final_X_train_list or not final_X_test_list:
             raise ValueError("Failed to generate sufficient data for train/test splits after pseudo-population step.")

        X_train = np.vstack(final_X_train_list)
        y_train = np.concatenate(final_y_train_list)
        concept_ids_train = np.array(final_concept_ids_train_list)

        X_test = np.vstack(final_X_test_list)
        y_test = np.concatenate(final_y_test_list)
        concept_ids_test = np.array(final_concept_ids_test_list)

        # 6. Shuffle final sets
        train_shuffle_idx = np.random.permutation(len(y_train))
        X_train, y_train, concept_ids_train = X_train[train_shuffle_idx], y_train[train_shuffle_idx], concept_ids_train[train_shuffle_idx]

        test_shuffle_idx = np.random.permutation(len(y_test))
        X_test, y_test, concept_ids_test = X_test[test_shuffle_idx], y_test[test_shuffle_idx], concept_ids_test[test_shuffle_idx]

        # 7. Return results
        info = {
            'group1': self.group1,
            'group2': self.group2,
            'concept_ids_train': concept_ids_train,
            'concept_ids_test': concept_ids_test,
            'real_samples_in_train': len(train_indices), # Number of real samples before pseudos
            'real_samples_in_test': len(test_indices),   # Number of real samples before pseudos
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
    Handles decoding for either a single concept pair or groups of concepts.
    Preserves concept identity for visualization while using binary classification.
    """
    def __init__(self, patient_data: PatientData, 
                 c1: Union[str, Tuple[str, ...]], 
                 c2: Union[str, Tuple[str, ...]], 
                 epoch: str, 
                 classifier: BaseEstimator = LinearSVC(), 
                 dataset = None, 
                 standardize: bool=False, 
                 min_samples=20, 
                 neurons=None):
        
        self.patient_data = patient_data
        # Normalize inputs to tuples for consistent handling
        self.c1 = c1 if isinstance(c1, tuple) else (c1,)
        self.c2 = c2 if isinstance(c2, tuple) else (c2,)
        self.epoch = epoch
        self.classifier = classifier
        self.min_samples = min_samples
        self.neurons = neurons

        self.scaler = StandardScaler() if standardize else None
        self.metrics = {}
        self.enough_data = True
        
        # Detect if we're in pair mode or group mode
        self.is_pair_mode = len(self.c1) == 1 and len(self.c2) == 1
        
        # Create or use provided dataset
        if dataset is None:
            self.dataset = ConceptDataset(
                patient_data=self.patient_data,
                concepts_groups=(self.c1, self.c2),
                epoch=self.epoch,
                min_samples=self.min_samples,
                neurons=self.neurons
            )
        else:
            self.dataset = dataset
            
        # Check if we have enough data
        try:
            _, _ = self.dataset.create_dataset_normal()
        except ValueError as e:
            print(f"Skipping concept {'pair' if self.is_pair_mode else 'groups'} {self.c1} vs {self.c2}: {e}")
            self.enough_data = False
            
    def decode_normal(self, test_size=0.3):
        """Perform decoding without pseudopopulations"""
        try:
            data_dict, info = self.dataset.create_dataset_normal(test_size=test_size, T_separation=21)
        except ValueError as e:
            print(f"Skipping concept {'pair' if self.is_pair_mode else 'groups'} {self.c1} vs {self.c2}: {e}")
            return None
            
        # Transfer concept ID tracking from info to data_dict
        if info and 'concept_ids_train' in info:
            data_dict['concept_ids_train'] = info['concept_ids_train']
        if info and 'concept_ids_test' in info:
            data_dict['concept_ids_test'] = info['concept_ids_test']
            
        return self._decode(data_dict=data_dict)
    
    def decode_pseudo(self, train_size_total=200, test_size_total=67, test_size=0.3, T_separation=21):
        """Perform decoding with pseudopopulations for balancing"""
        try:
            data_dict, info = self.dataset.create_dataset_pseudo(
                test_size=test_size,
                train_size_total=train_size_total,
                test_size_total=test_size_total,
                T_separation=T_separation
            )
        except ValueError as e:
            print(f"Skipping concept {'pair' if self.is_pair_mode else 'groups'} {self.c1} vs {self.c2}: {e}")
            return None
            
        # Transfer concept ID tracking from info to data_dict
        if info and 'concept_ids_train' in info:
            data_dict['concept_ids_train'] = info['concept_ids_train']
        if info and 'concept_ids_test' in info:
            data_dict['concept_ids_test'] = info['concept_ids_test']
            
        return self._decode(data_dict=data_dict)
        
    def _decode(self, data_dict) -> DecodingResult:
        """
        Performs the actual decoding and metrics calculation.
        
        The crucial part here is that we:
        1. Train the model on binary labels (0/1 for group membership)
        2. Preserve concept identity information in the data dictionary
        3. Return everything in the standard DecodingResult format
        """
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        if len(y_train) == 0:
            print(f"ERROR: Empty training set resulted from split for {self.c1} vs {self.c2}. Cannot proceed.")
            # Return NaN for all metrics if training isn't possible
            # Determine keys for sample dictionaries based on mode
            train_sample_keys = {self.c1[0]: 0, self.c2[0]: 0} if self.is_pair_mode else {'group1': 0, 'group2': 0}
            test_sample_keys = {self.c1[0]: 0, self.c2[0]: 0} if self.is_pair_mode else {'group1': 0, 'group2': 0}
            if not self.is_pair_mode: # Add specific concept keys if group mode
                 for concept in self.c1 + self.c2:
                     train_sample_keys[concept] = 0
                     test_sample_keys[concept] = 0

            return DecodingResult(
                train_accuracy=np.nan, train_roc_auc=np.nan,
                test_accuracy=np.nan, test_roc_auc=np.nan,
                train_samples=train_sample_keys,
                test_samples=test_sample_keys,
                predictions=np.array([]), true_labels=np.array([]),
                classifier=self.classifier, # Untrained classifier instance
                data=data_dict
            )


        # Apply standardization if requested
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            data_dict['X_train'] = X_train # put scaled values back in data matrix
            data_dict['X_test'] = X_test
            
        # Train classifier on binary labels
        self.classifier.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = self.classifier.predict(X_train)
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # check for only one concept in train set - bad
        if len(np.unique(y_train)) > 1:
            train_roc_auc = roc_auc_score(y_train, y_train_pred)
        else:
            print(f"Warning: Only one class present in y_train for {self.c1} vs {self.c2}. Setting train_roc_auc to NaN.")
            train_roc_auc = np.nan
    
        test_set_valid_for_eval = len(y_test) > 0

        if test_set_valid_for_eval:
            # Standardize test set if necessary (using scaler fitted on train)
            if self.scaler:
                 X_test = self.scaler.transform(X_test)
                 data_dict['X_test'] = X_test # Keep scaled data

            # Make predictions on the test set
            y_pred = self.classifier.predict(X_test)

            # Calculate Test Accuracy (always possible if test set is not empty)
            test_accuracy = accuracy_score(y_test, y_pred)

            # Calculate Test ROC AUC (only if multiple classes present)
            if len(np.unique(y_test)) > 1:
                test_roc_auc = roc_auc_score(y_test, y_pred)
            else:
                print(f"Warning: Only one class present in y_test for {self.c1} vs {self.c2}. Setting test_roc_auc to NaN.")
                test_roc_auc = np.nan
        else:
            # Test set is empty
            print(f"Warning: Empty test set resulted from temporal split for {self.c1} vs {self.c2}. Setting test metrics to NaN.")
            y_pred = np.array([]) # No predictions possible
            test_accuracy = np.nan
            test_roc_auc = np.nan


        # --- Calculate Sample Counts ---
        # (Ensuring keys exist even if counts are 0)
        train_samples = {}
        test_samples = {}
        group1_concepts = self.c1
        group2_concepts = self.c2
        all_concepts = list(group1_concepts) + list(group2_concepts)

        if self.is_pair_mode:
            c1_key, c2_key = group1_concepts[0], group2_concepts[0]
            train_samples[c1_key] = np.sum(y_train == 0)
            train_samples[c2_key] = np.sum(y_train == 1)
            test_samples[c1_key] = np.sum(y_test == 0) if test_set_valid_for_eval else 0
            test_samples[c2_key] = np.sum(y_test == 1) if test_set_valid_for_eval else 0
        else: # Group mode
            train_samples['group1'] = np.sum(y_train == 0)
            train_samples['group2'] = np.sum(y_train == 1)
            test_samples['group1'] = np.sum(y_test == 0) if test_set_valid_for_eval else 0
            test_samples['group2'] = np.sum(y_test == 1) if test_set_valid_for_eval else 0
            # Add specific concepts
            concept_ids_train = data_dict.get('concept_ids_train', np.array([]))
            concept_ids_test = data_dict.get('concept_ids_test', np.array([]))
            for concept in all_concepts:
                train_samples[concept] = np.sum(concept_ids_train == concept)
                test_samples[concept] = np.sum(concept_ids_test == concept) if test_set_valid_for_eval else 0

        
        # Create and return standard result object
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
            data=data_dict  # Contains concept identity information
        )


        
    

# %%
class SingleResultsManager:
    """
    Manages decoding results for a single patient and multiple concept pairs/groups for a single patient and epoch.
    
    This enhanced version handles both:
    - Traditional concept pairs: ('J.Bauer', 'C.OBrian')
    - Concept groups: (('J.Bauer', 'T.Lennox'), ('C.OBrian', 'A.Fayed'))
    """
    def __init__(self, patient_data: PatientData, concept_items: List[Tuple], 
                 epoch: str, classifier: BaseEstimator = LinearSVC(), 
                 standardize: bool = False, pseudo=False, neurons=None, **pseudo_kwargs):
        """
        Initialize with concept pairs or groups.
        
        Args:
            patient_data: PatientData object for a single patient
            concept_items: List of concept pairs or concept groups to decode
                - For pairs: List of tuples like [('J.Bauer', 'C.OBrian'), ...]
                - For groups: List of tuples like [(('J.Bauer', 'T.Lennox'), ('C.OBrian', 'A.Fayed')), ...]
                - Can mix pairs and groups in the same list
            epoch: Data epoch ('movie', 'preSleep_recall', etc.)
            classifier: Classifier to use for decoding (default: LinearSVC)
            standardize: Whether to standardize features
            pseudo: Whether to use pseudopopulations
            neurons: List of neurons to use (optional)
            **pseudo_kwargs: Additional parameters for pseudopopulation generation
        """
        self.patient_data = patient_data
        self.concept_items = concept_items
        self.epoch = epoch
        self.classifier = classifier
        self.standardize = standardize
        self.results = {}  # Store results here
        self.pseudo = pseudo
        self.pseudo_params = pseudo_kwargs
        self.neurons = neurons
        
        # Detect if we're using the old format (all pairs) or mixed/new format
        self.all_pairs = all(isinstance(item[0], str) and isinstance(item[1], str) 
                             for item in concept_items)

    def run_decoding(self, num_iter: int = 1) -> None:
        """
        Runs decoding for all concept pairs or groups provided in the constructor.
        Stores the DecodingResult in the self.results dictionary.
        
        Args:
            num_iter: Number of iterations to run for each pair/group
        """
        self.results = {}  # Reset results every time
        
        for item in tqdm(self.concept_items, desc=f"Decoding for {self.patient_data.pid}"):
            # Process the item to extract c1 and c2 (either strings or tuples)
            if isinstance(item[0], tuple) or isinstance(item[1], tuple):
                # Group mode
                c1, c2 = item
            else:
                # Pair mode
                c1, c2 = item[0], item[1]
                
            # Create decoder
            decoder = ConceptDecoder(
                patient_data=self.patient_data,
                c1=c1,
                c2=c2,
                epoch=self.epoch,
                classifier=self.classifier,
                standardize=self.standardize,
                neurons=self.neurons
            )
            
            if decoder.enough_data:  # Efficient way of not rechecking for sufficient samples
                for i in range(num_iter):
                    result = None
                    if self.pseudo:
                        if self.pseudo_params:
                            result = decoder.decode_pseudo(**self.pseudo_params)
                        else:
                            result = decoder.decode_pseudo()
                    else:
                        result = decoder.decode_normal()
                        
                    if result is not None:  # Only store if decode was successful
                        # Use the original item as key to maintain the format
                        if item not in self.results:
                            self.results[item] = [result]
                        else:
                            self.results[item].append(result)
    
    # Keep for backward compatibility
    def run_decoding_for_pairs(self, num_iter: int = 1) -> None:
        """
        Backward compatibility method that calls run_decoding.
        
        Args:
            num_iter: Number of iterations to run for each pair/group
        """
        return self.run_decoding(num_iter)

    def plot_train_test_performance_heatmap(self, metric='test_roc_auc', figsize=(20, 10)):
        """
        Generates and displays a combined heatmap of training and testing performance.
        For multiple iterations, shows mean performance with standard deviation in parentheses.
        
        This method is optimized for concept pairs. For concept groups, consider using
        plot_group_performance instead.
        
        Args:
            metric (str): One of 'test_accuracy', 'train_accuracy', 'test_roc_auc', 'train_roc_auc'
            figsize (tuple): Figure size for the plot
        """
        # Only collect individual concepts from pairs, not from groups
        pair_items = [item for item in self.concept_items 
                     if isinstance(item[0], str) and isinstance(item[1], str)]
        
        if not pair_items:
            print("No concept pairs available for heatmap visualization.")
            return None
            
        concepts = sorted(list(set([c for pair in pair_items for c in pair])))
        n_concepts = len(concepts)
        
        # Initialize matrices for means and standard deviations
        train_mean_matrix = np.full((n_concepts, n_concepts), np.nan)
        test_mean_matrix = np.full((n_concepts, n_concepts), np.nan)
        train_std_matrix = np.full((n_concepts, n_concepts), np.nan)
        test_std_matrix = np.full((n_concepts, n_concepts), np.nan)

        concept_to_idx = {concept: i for i, concept in enumerate(concepts)}

        for item, results in self.results.items():
            # Skip group items
            if not (isinstance(item[0], str) and isinstance(item[1], str)):
                continue
                
            if results:  # Check if results exist for this pair
                c1, c2 = item
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

        # Return statistics for analysis if needed
        stats = {
            'train_mean': train_mean_matrix,
            'train_std': train_std_matrix,
            'test_mean': test_mean_matrix,
            'test_std': test_std_matrix
        }
        return stats
        
    def plot_group_performance(self, metric='test_roc_auc', figsize=(10, 6)):
        """
        Generates a bar chart showing performance for concept groups.
        
        Args:
            metric (str): One of 'test_accuracy', 'train_accuracy', 'test_roc_auc', 'train_roc_auc'
            figsize (tuple): Figure size for the plot
        """
        # Filter for group items
        group_items = [item for item in self.concept_items 
                      if isinstance(item[0], tuple) or isinstance(item[1], tuple)]
        
        if not group_items:
            print("No concept groups available for visualization.")
            return None
            
        # Prepare data for plotting
        group_labels = []
        performance_values = []
        error_values = []
        
        for item in group_items:
            if item in self.results and self.results[item]:
                # Create label for this group pair
                if isinstance(item[0], tuple) and isinstance(item[1], tuple):
                    group1_str = '+'.join(item[0])
                    group2_str = '+'.join(item[1])
                    label = f"{group1_str} vs {group2_str}"
                else:
                    # Mixed format (one group, one concept)
                    group1 = '+'.join(item[0]) if isinstance(item[0], tuple) else item[0]
                    group2 = '+'.join(item[1]) if isinstance(item[1], tuple) else item[1]
                    label = f"{group1} vs {group2}"
                    
                group_labels.append(label)
                
                # Get performance for this group
                results = self.results[item]
                if 'roc_auc' in metric:
                    values = [getattr(r, metric) for r in results]
                else:
                    values = [getattr(r, metric) for r in results]
                
                performance_values.append(np.mean(values))
                error_values.append(np.std(values))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(group_labels)), performance_values, 
                     yerr=error_values, align='center', alpha=0.7)
        
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(f'{metric.replace("test_", "").replace("_", " ").title()}')
        ax.set_title(f'Group Decoding Performance\nPatient {self.patient_data.pid}, Epoch: {self.epoch}')
        
        plt.tight_layout()
        
        return {'group_labels': group_labels, 'performance': performance_values, 'error': error_values}
    



    





# --- Other methods (__init__, _normalize_concepts_input, _get_exclusive_onsets_for_concept) remain largely the same ---

# End of ConceptDataset class    
