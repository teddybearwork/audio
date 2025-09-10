import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class AnimalSoundClassifier:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the animal sound classifier."""
        self.config = self._load_config(config_path)
        self.model = None
        self.classes = []
        self.sample_rate = self.config['audio']['sample_rate']
        self.confidence_threshold = self.config['model']['confidence_threshold']
        
        # Load YAMNet model
        self._load_model()
        
        logger.info("Animal Sound Classifier initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
    
    def _load_model(self):
        """Load the YAMNet model from TensorFlow Hub."""
        try:
            # Load YAMNet model
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            # Load YAMNet class names
            class_map_path = self.model.class_map_path().numpy()
            self.class_map = self._load_class_map(class_map_path)
            
            logger.info("YAMNet model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading YAMNet model: {e}")
            raise
    
    @staticmethod
    def _load_class_map(class_map_path: str) -> Dict[int, str]:
        """Load YAMNet's class map."""
        import csv
        class_map = {}
        try:
            # First try to read as CSV
            with open(class_map_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if len(row) >= 2:  # At least ID and display name
                        try:
                            class_id = int(row[0].split(',')[0])  # Handle '1,/m/...' format
                            class_name = row[1].strip('"')
                            class_map[class_id] = class_name
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            logger.warning(f"Error reading class map as CSV: {e}, trying fallback method")
            # Fallback to original method
            with open(class_map_path) as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) >= 2:
                        try:
                            class_id = int(parts[0].split(',')[0])
                            class_name = ' '.join(parts[1:]).strip('"')
                            class_map[class_id] = class_name
                        except (ValueError, IndexError):
                            continue
        
        if not class_map:
            logger.warning("No valid class mappings found, using default classes")
            # Default to some common animal sounds if no mapping is found
            class_map = {
                0: 'Animal',
                1: 'Bird',
                2: 'Dog',
                3: 'Cat',
                4: 'Rooster',
                5: 'Pig',
                6: 'Cow',
                7: 'Frog',
                8: 'Sheep',
                9: 'Chicken'
            }
            
        logger.info(f"Loaded {len(class_map)} class mappings")
        return class_map
    
    def predict(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Predict the class of the given audio data.
        
        Args:
            audio_data: Numpy array containing audio data (1D, mono, 16kHz)
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        try:
            # Ensure audio is in the correct format (1D array of float32)
            if isinstance(audio_data, np.ndarray):
                # Convert to 1D if needed
                if len(audio_data.shape) > 1:
                    audio_data = np.squeeze(audio_data)
                    if len(audio_data.shape) > 1:  # If still not 1D
                        audio_data = np.mean(audio_data, axis=-1)  # Average across channels
                
                # Ensure float32 dtype and normalize to [-1.0, 1.0] if needed
                if audio_data.dtype != np.float32:
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    else:
                        audio_data = audio_data.astype(np.float32)
                
                # Ensure values are in [-1.0, 1.0]
                max_val = np.max(np.abs(audio_data))
                if max_val > 1.0:
                    audio_data = audio_data / max_val
            
            # Ensure we have at least 1 second of audio (YAMNet expects this)
            min_samples = self.sample_rate  # 1 second of audio
            if len(audio_data) < min_samples:
                # Pad with zeros if audio is too short
                padding = np.zeros(min_samples - len(audio_data), dtype=np.float32)
                audio_data = np.concatenate([audio_data, padding])
            
            # Run inference
            scores, embeddings, spectrogram = self.model(audio_data)
            
            # Get the top prediction
            scores_np = scores.numpy()
            predicted_class_id = np.argmax(scores_np[0])
            confidence = float(scores_np[0][predicted_class_id])
            
            # Map to class name
            predicted_class = self.class_map.get(predicted_class_id, "unknown")
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                return "unknown", confidence
                
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "error", 0.0
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process a chunk of audio and return prediction results.
        
        Args:
            audio_chunk: Numpy array containing audio data
            
        Returns:
            dict: Dictionary containing prediction results
        """
        try:
            # Make prediction
            predicted_class, confidence = self.predict(audio_chunk)
            
            # Prepare result
            result = {
                'class': predicted_class,
                'confidence': float(confidence),
                'is_animal_sound': predicted_class != 'unknown' and confidence >= self.confidence_threshold
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                'class': 'error',
                'confidence': 0.0,
                'is_animal_sound': False,
                'error': str(e)
            }
    
    def get_audio_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Get the audio embedding from the model.
        
        Args:
            audio_data: Numpy array containing audio data
            
        Returns:
            np.ndarray: Audio embedding vector
        """
        try:
            # Ensure audio is in the correct format
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)  # Convert to mono if needed
            
            # Get embeddings
            scores, embeddings, spectrogram = self.model(audio_data)
            
            return embeddings.numpy()[0]  # Return first (and only) embedding
            
        except Exception as e:
            logger.error(f"Error getting audio embedding: {e}")
            raise
    
    def get_supported_classes(self) -> List[str]:
        """Get the list of supported animal/bird sound classes."""
        # This would be customized based on your specific classes
        return [
            'lion_roar', 'elephant_trumpet', 'bird_chirp',
            # Add more classes as needed
        ]
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for predictions."""
        if 0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold set to {threshold}")
        else:
            logger.warning(f"Invalid confidence threshold: {threshold}. Must be between 0 and 1.")
