import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

from src.model import AnimalSoundClassifier
from src.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class AudioMatcher:
    def __init__(self, classifier: AnimalSoundClassifier, data_dir: str = 'data'):
        """Initialize the audio matcher."""
        self.classifier = classifier
        self.data_dir = Path(data_dir)
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        self.audio_processor = AudioProcessor()
        self._build_reference_database()

    def _build_reference_database(self):
        """Build a database of embeddings from audio files in the data directory."""
        logger.info(f"Building reference audio database from: {self.data_dir}")
        audio_files = list(self.data_dir.glob('*.mp3'))
        if not audio_files:
            logger.warning(f"No audio files found in {self.data_dir}. The matcher will not work.")
            return

        for audio_file in audio_files:
            try:
                animal_name = audio_file.stem
                logger.debug(f"Processing reference file: {animal_name}")
                
                # Load audio file
                audio_data = self.audio_processor.load_audio_file(str(audio_file))
                
                # Process in chunks to get multiple embeddings
                chunk_size = int(self.audio_processor.chunk_duration * self.audio_processor.sample_rate)
                file_embeddings = []
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    if len(chunk) < chunk_size:
                        continue # Skip partial chunks

                    if not self.audio_processor._is_silent(chunk):
                        # For mock classifier, use a simple hash-based approach
                        result = self.classifier.process_audio_chunk(chunk)
                        if result['confidence'] > 0.3:  # Use result as "embedding"
                            # Create a simple hash-based embedding for the chunk
                            chunk_hash = hash(tuple(chunk[::100])) % 1000  # Simple hash
                            file_embeddings.append([chunk_hash, result['confidence']])
                
                if file_embeddings:
                    self.reference_embeddings[animal_name] = np.array(file_embeddings)
            except Exception as e:
                logger.error(f"Failed to process reference file {audio_file}: {e}", exc_info=True)
        
        logger.info(f"Reference database built with {len(self.reference_embeddings)} entries.")

    def match_sound(self, audio_chunk: np.ndarray, threshold: float = 0.5) -> Optional[Tuple[str, float]]:
        """Match an audio chunk against the reference database."""
        if not self.reference_embeddings:
            return None

        try:
            # Get prediction result for the new audio chunk
            result = self.classifier.process_audio_chunk(audio_chunk)
            if result['confidence'] < 0.3:
                return None
                
            # Create a simple hash-based embedding for the chunk
            chunk_hash = hash(tuple(audio_chunk[::100])) % 1000  # Simple hash
            chunk_embedding = np.array([chunk_hash, result['confidence']])
            
            # Normalize the chunk embedding
            chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding)

            best_match = None
            highest_similarity = -1

            # Find the best match using cosine similarity against all reference embeddings
            for name, ref_embeddings in self.reference_embeddings.items():
                # Normalize all reference embeddings for the current sound
                norm_ref_embeddings = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
                
                # Calculate similarities and find the max
                similarities = np.dot(norm_ref_embeddings, chunk_embedding)
                max_similarity_for_sound = np.max(similarities)

                if max_similarity_for_sound > highest_similarity:
                    highest_similarity = max_similarity_for_sound
                    best_match = name
            
            if best_match and highest_similarity >= threshold:
                return best_match, highest_similarity

        except Exception as e:
            logger.error(f"Error during sound matching: {e}", exc_info=True)
        
        return None
