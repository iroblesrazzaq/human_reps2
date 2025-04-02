import unittest
import numpy as np
from unittest.mock import MagicMock
from sklearn.svm import LinearSVC

# Import the old and new implementations
from old_decoders import ConceptPairDataset, ConceptDecoder as OldConceptDecoder, DecodingResult
from decoders import ConceptDataset, ConceptDecoder

class TestConceptDatasetCompatibility(unittest.TestCase):
    """Tests for compatibility between the old ConceptPairDataset and new ConceptDataset classes."""
    
    def setUp(self):
        """Set up common test fixtures."""
        # Create mock PatientData
        self.mock_patient = MagicMock()
        
        # Generate fixed test data
        np.random.seed(42)
        self.random_state = 42
        self.c1_data = np.random.rand(20, 5)  # 20 samples, 5 neurons
        self.c2_data = np.random.rand(15, 5)  # 15 samples, 5 neurons
        
        # Configure mock to return our fixed data
        self.mock_patient.get_concept_data.return_value = (self.c1_data, self.c2_data)
        
        # Common test parameters
        self.c1 = "A.Fayed"
        self.c2 = "M.OBrian"
        self.epoch = "movie"
        self.min_samples = 10
        
        # Create instances of both dataset classes
        self.old_dataset = ConceptPairDataset(
            patient_data=self.mock_patient,
            concept_pair=(self.c1, self.c2),
            epoch=self.epoch,
            min_samples=self.min_samples
        )
        
        self.new_dataset = ConceptDataset(
            patient_data=self.mock_patient,
            concepts_groups=(self.c1, self.c2),
            epoch=self.epoch,
            min_samples=self.min_samples
        )
    
    def test_normal_dataset_structure(self):
        """
        Test that both dataset implementations return dictionaries with the same keys
        and arrays with compatible shapes when using normal dataset creation.
        
        This test is necessary to ensure the basic interface remains compatible.
        """
        # Use the same random seed for both to minimize randomness differences
        np.random.seed(42)
        old_data, _ = self.old_dataset.create_dataset_normal()
        
        np.random.seed(42)
        new_data, _ = self.new_dataset.create_dataset_normal()
        
        # Check that both return the same keys
        self.assertEqual(set(old_data.keys()), set(new_data.keys()),
                        "Dataset dictionaries should have the same keys")
        
        # Check that arrays have compatible shapes
        for key in old_data:
            old_shape = old_data[key].shape
            new_shape = new_data[key].shape
            
            # Check dimensionality
            self.assertEqual(len(old_shape), len(new_shape), 
                          f"Array {key} should have the same number of dimensions")
            
            # Check total size is similar (not exact due to randomness in train/test split)
            old_size = np.prod(old_shape)
            new_size = np.prod(new_shape)
            
            # Should be within 10% (allowing for some variance due to random splitting)
            self.assertLess(abs(old_size - new_size) / old_size, 0.1,
                          f"Total size of {key} should be similar")
    
    def test_pseudo_dataset_structure(self):
        """
        Test that both dataset implementations return dictionaries with the same keys
        and arrays with compatible shapes when using pseudopopulation datasets.
        
        This test is necessary to verify compatibility with the more complex pseudopopulation feature.
        """
        # Use the same random seed for both
        np.random.seed(42)
        old_data, _ = self.old_dataset.create_dataset_pseudo(
            train_size_total=100, test_size_total=50)
        
        np.random.seed(42)
        new_data, _ = self.new_dataset.create_dataset_pseudo(
            train_size_total=100, test_size_total=50)
        
        # Check that both return the same keys
        self.assertEqual(set(old_data.keys()), set(new_data.keys()),
                        "Dataset dictionaries should have the same keys")
        
        # For pseudopopulations, the sizes should be more controlled by the parameters
        # Check that first dimensions (number of samples) match more closely
        for key in old_data:
            old_shape = old_data[key].shape
            new_shape = new_data[key].shape
            
            # Check dimensionality
            self.assertEqual(len(old_shape), len(new_shape), 
                          f"Array {key} should have the same number of dimensions")
            
            # Check first dimension (samples) is within 10% difference
            # This is particularly important for pseudopopulation-augmented datasets
            self.assertLess(abs(old_shape[0] - new_shape[0]) / max(1, old_shape[0]), 0.1,
                          f"First dimension of {key} should be similar")
    
    def test_min_samples_enforcement(self):
        """
        Test that both implementations correctly enforce the min_samples constraint
        and raise similar errors when there's insufficient data.
        
        This test verifies error handling compatibility for a common edge case.
        """
        # Update mock to return insufficient data for c1
        small_c1_data = np.random.rand(5, 5)  # Only 5 samples (< min_samples)
        self.mock_patient.get_concept_data.return_value = (small_c1_data, self.c2_data)
        
        # Create new dataset instances with the updated mock
        old_dataset = ConceptPairDataset(
            patient_data=self.mock_patient,
            concept_pair=(self.c1, self.c2),
            epoch=self.epoch,
            min_samples=self.min_samples
        )
        
        new_dataset = ConceptDataset(
            patient_data=self.mock_patient,
            concepts_groups=(self.c1, self.c2),
            epoch=self.epoch,
            min_samples=self.min_samples
        )
        
        # Both should raise ValueError
        with self.assertRaises(ValueError):
            old_dataset.create_dataset_normal()
        
        with self.assertRaises(ValueError):
            new_dataset.create_dataset_normal()


class TestConceptDecoderCompatibility(unittest.TestCase):
    """Tests for compatibility between the old ConceptDecoder and new ConceptDecoder classes."""
    
    def setUp(self):
        """Set up common test fixtures."""
        # Create mock PatientData
        self.mock_patient = MagicMock()
        
        # Generate fixed test data
        np.random.seed(42)
        self.c1_data = np.random.rand(20, 5)  # 20 samples, 5 neurons
        self.c2_data = np.random.rand(15, 5)  # 15 samples, 5 neurons
        
        # Configure mock to return our fixed data
        self.mock_patient.get_concept_data.return_value = (self.c1_data, self.c2_data)
        
        # Common test parameters
        self.c1 = "concept1"
        self.c2 = "concept2"
        self.epoch = "movie"
        self.min_samples = 10
        
        # Create a consistent classifier to use for both decoders
        self.classifier = LinearSVC(random_state=42)
        
        # Create instances of both decoder classes
        self.old_decoder = OldConceptDecoder(
            patient_data=self.mock_patient,
            c1=self.c1,
            c2=self.c2,
            epoch=self.epoch,
            classifier=self.classifier,
            min_samples=self.min_samples
        )
        
        self.new_decoder = ConceptDecoder(
            patient_data=self.mock_patient,
            c1=self.c1,
            c2=self.c2,
            epoch=self.epoch,
            classifier=self.classifier,
            min_samples=self.min_samples
        )
    
    def test_decode_result_structure(self):
        """
        Test that both decoder implementations return results with the same structure
        and compatible attribute types when using the internal _decode method.
        
        This test verifies the core decoding logic produces compatible outputs.
        """
        # Create a simple data_dict to use with _decode
        np.random.seed(42)
        X_train = np.random.rand(10, 5)
        X_test = np.random.rand(5, 5)
        y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_test = np.array([0, 0, 1, 1, 1])
        
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Mock the classifier predict method for consistent outputs
        mock_classifier = MagicMock()
        mock_classifier.fit.return_value = None
        mock_classifier.predict.side_effect = [y_train, y_test]  # First call for train, second for test
        
        # Set both decoders to use the mock classifier
        self.old_decoder.classifier = mock_classifier
        self.new_decoder.classifier = mock_classifier
        
        # Get results from old decoder
        old_result = self.old_decoder._decode(data_dict)
        
        # Reset mock and get new result
        mock_classifier.predict.side_effect = [y_train, y_test]
        new_result = self.new_decoder._decode(data_dict)
        
        # Check both return DecodingResult objects or equivalent
        self.assertEqual(type(old_result).__name__, type(new_result).__name__,
                       "Both should return the same result type")
        
        # Check attributes exist and have the same types
        old_attrs = vars(old_result)
        new_attrs = vars(new_result)
        
        self.assertEqual(set(old_attrs.keys()), set(new_attrs.keys()),
                        "Results should have the same attribute names")
        
        # Check numeric attributes (values should match exactly with mocked inputs)
        for attr in ['test_accuracy', 'train_accuracy', 'test_roc_auc', 'train_roc_auc']:
            self.assertAlmostEqual(old_attrs[attr], new_attrs[attr], places=5,
                                 msg=f"Attribute {attr} should match")
        
        # Check dictionary attributes have the same structure
        self.assertEqual(set(old_result.train_samples.keys()), 
                       set(new_result.train_samples.keys()),
                       "train_samples should have the same keys")
                       
        self.assertEqual(set(old_result.test_samples.keys()), 
                       set(new_result.test_samples.keys()),
                       "test_samples should have the same keys")
    
    def test_normal_decode_interface(self):
        """
        Test that both decoders have compatible normal decode methods
        that return the same type of results.
        
        This test ensures the high-level interface remains compatible.
        """
        # Set fixed random seed for reproducibility
        np.random.seed(42)
        old_result = self.old_decoder.decode_normal()
        
        # Reset seed for consistent randomization
        np.random.seed(42)
        new_result = self.new_decoder.decode_normal()
        
        # Check both return non-None results
        self.assertIsNotNone(old_result)
        self.assertIsNotNone(new_result)
        
        # Check they return the same type
        self.assertEqual(type(old_result).__name__, type(new_result).__name__)
        
        # Verify key attributes exist and have compatible types
        self.assertIsInstance(old_result.test_accuracy, float)
        self.assertIsInstance(new_result.test_accuracy, float)
        self.assertIsInstance(old_result.train_samples, dict)
        self.assertIsInstance(new_result.train_samples, dict)
    
    def test_pseudo_decode_interface(self):
        """
        Test that both decoders have compatible pseudo decode methods.
        
        This test verifies the pseudopopulation decoding interface remains compatible.
        """
        # Set parameters for pseudopopulation generation
        train_size_total = 100
        test_size_total = 50
        
        # Set fixed random seed for reproducibility
        np.random.seed(42)
        old_result = self.old_decoder.decode_pseudo(
            train_size_total=train_size_total, 
            test_size_total=test_size_total
        )
        
        # Reset seed for consistent randomization
        np.random.seed(42)
        new_result = self.new_decoder.decode_pseudo(
            train_size_total=train_size_total, 
            test_size_total=test_size_total
        )
        
        # Check both return non-None results
        self.assertIsNotNone(old_result)
        self.assertIsNotNone(new_result)
        
        # Check they return the same type
        self.assertEqual(type(old_result).__name__, type(new_result).__name__)
        
        # Verify that the number of samples is roughly as requested
        # This is important for pseudopopulation-based methods
        self.assertGreaterEqual(len(old_result.predictions), test_size_total * 0.8)
        self.assertGreaterEqual(len(new_result.predictions), test_size_total * 0.8)
    
    def test_error_handling(self):
        """
        Test that both decoders handle errors similarly when there's insufficient data.
        
        This test verifies error handling consistency for a common edge case.
        """
        # Update mock to return insufficient data
        small_c1_data = np.random.rand(5, 5)  # Only 5 samples (< min_samples)
        self.mock_patient.get_concept_data.return_value = (small_c1_data, self.c2_data)
        
        # Create new decoder instances with the updated mock
        old_decoder = OldConceptDecoder(
            patient_data=self.mock_patient,
            c1=self.c1,
            c2=self.c2,
            epoch=self.epoch,
            classifier=self.classifier,
            min_samples=self.min_samples
        )
        
        new_decoder = ConceptDecoder(
            patient_data=self.mock_patient,
            c1=self.c1,
            c2=self.c2,
            epoch=self.epoch,
            classifier=self.classifier,
            min_samples=self.min_samples
        )
        
        # Both should return None when there's insufficient data
        self.assertIsNone(old_decoder.decode_normal())
        self.assertIsNone(new_decoder.decode_normal())

    def test_raw_data_compatibility(self):
        """
        Test that both implementations use exactly the same raw data before train/test splitting.
        
        While train/test splits may vary due to randomness, the combined data should be identical.
        This ensures fundamental compatibility between implementations.
        """
        # Use deterministic test data with distinct values for clear verification
        c1_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c2_data = np.array([[7.0, 8.0], [9.0, 10.0]])
        
        # Expected combined data
        expected_X = np.vstack([c1_data, c2_data])  # Shape (5, 2)
        expected_y = np.array([0, 0, 0, 1, 1])  # 3 samples of class 0, 2 of class 1
        
        # Configure mock to return our test data
        self.mock_patient.get_concept_data.return_value = (c1_data, c2_data)
        
        # Create instances with the mock
        old_dataset = ConceptPairDataset(
            patient_data=self.mock_patient,
            concept_pair=(self.c1, self.c2),
            epoch=self.epoch,
            min_samples=1  # Set low to accept our small test data
        )
        
        new_dataset = ConceptDataset(
            patient_data=self.mock_patient,
            concepts_groups=(self.c1, self.c2),
            epoch=self.epoch,
            min_samples=1
        )
        
        # Variables to capture the raw data before splitting
        captured_old_X = None
        captured_old_y = None
        captured_new_X = None
        captured_new_y = None
        
        # Create patch functions to capture the data
        def capture_old_data(X, y, **kwargs):
            nonlocal captured_old_X, captured_old_y
            captured_old_X = X.copy()  # Make a copy to avoid reference issues
            captured_old_y = y.copy()
            
            # Return any valid split for the test to continue
            train_idx = np.array([0, 2, 4])
            test_idx = np.array([1, 3])
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        
        def capture_new_data(X, y, **kwargs):
            nonlocal captured_new_X, captured_new_y
            captured_new_X = X.copy()
            captured_new_y = y.copy()
            
            # Return the same split pattern for consistency
            train_idx = np.array([0, 2, 4])
            test_idx = np.array([1, 3])
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        
        # Run old implementation with patched function
        with patch('sklearn.model_selection.train_test_split', capture_old_data):
            old_dataset.create_dataset_normal()
        
        # Run new implementation with patched function
        with patch('sklearn.model_selection.train_test_split', capture_new_data):
            new_dataset.create_dataset_normal()
        
        # Compare captured raw data from both implementations
        np.testing.assert_array_equal(
            captured_old_X, captured_new_X, 
            "Both implementations should pass identical feature data to train_test_split"
        )
        
        np.testing.assert_array_equal(
            captured_old_y, captured_new_y,
            "Both implementations should pass identical labels to train_test_split"
        )
        
        # Verify the data matches our expectations
        np.testing.assert_array_equal(
            captured_old_X, expected_X,
            "Data preparation should correctly combine concept features"
        )
        
        np.testing.assert_array_equal(
            captured_old_y, expected_y,
            "Data preparation should correctly generate concept labels"
        )

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    
    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"\nSummary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
