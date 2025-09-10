import numpy as np
import sounddevice as sd
import librosa
import yaml
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import queue
import threading
import time

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the audio processor with configuration."""
        self.config = self._load_config(config_path)
        self.sample_rate = self.config['audio']['sample_rate']
        self.channels = self.config['audio']['channels']
        self.chunk_duration = self.config['audio']['chunk_duration']
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.silence_threshold = self.config['audio']['silence_threshold']
        self.min_silence_duration = self.config['audio']['min_silence_duration']
        self.sound_event_buffer_padding = self.config['audio']['sound_event_buffer']

        # State for sound event detection
        self.is_event_active = False
        self.silence_chunks_counter = 0
        self.event_audio_buffer = []
        self.chunks_for_silence = int(self.min_silence_duration / self.chunk_duration)
        
        # Audio buffer and processing queue
        self.audio_buffer = np.array([], dtype=np.float32)
        self.processing_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        
        logger.info("Audio Processor initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
    
    async def start_stream(self, callback=None, device_index=None):
        """Start audio stream from microphone."""
        import asyncio
        loop = asyncio.get_running_loop()
        
        # This inner callback will run in a separate thread managed by sounddevice
        def audio_callback_sync(indata, frames, time, status):
            if status:
                logger.warning(f"Audio stream status: {status}")
            
            # Convert to mono if needed
            audio_data = indata.flatten() if indata.shape[1] == 1 else np.mean(indata, axis=1)
            
            # Add to buffer
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_data.astype(np.float32)))
            
            # Process chunks if we have enough data
            while len(self.audio_buffer) >= self.chunk_samples:
                chunk = self.audio_buffer[:self.chunk_samples]
                self.audio_buffer = self.audio_buffer[self.chunk_samples:]
                
                is_silent = self._is_silent(chunk)

                if not is_silent:
                    # If sound is detected, start a new event or continue the current one
                    if not self.is_event_active:
                        logger.debug("Sound event started.")
                        self.is_event_active = True
                        self.event_audio_buffer = []
                    self.event_audio_buffer.append(chunk)
                    self.silence_chunks_counter = 0
                elif self.is_event_active:
                    # If a silent chunk is detected during an event, count it
                    self.silence_chunks_counter += 1
                    if self.silence_chunks_counter > self.chunks_for_silence:
                        # If enough silence has passed, the event is over
                        logger.debug(f"Sound event ended after {self.silence_chunks_counter} silent chunks.")
                        complete_event = np.concatenate(self.event_audio_buffer)
                        
                        # Add padding to the event
                        padding_samples = int(self.sound_event_buffer_padding * self.sample_rate)
                        padded_event = np.pad(complete_event, (padding_samples, padding_samples), 'constant')
                        
                        if callback:
                            asyncio.run_coroutine_threadsafe(callback(padded_event), loop)
                        
                        # Reset for the next event
                        self.is_event_active = False
                        self.event_audio_buffer = []
                        self.silence_chunks_counter = 0

        try:
            self.is_recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                device=device_index,
                callback=audio_callback_sync
            )
            self.stream.start()
            logger.info(f"Audio stream started on device {device_index or 'default'}")
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            raise
    
    def _open_stream(self, input_device=None):
        """Open the audio input stream with optimized settings."""
        try:
            # Calculate optimal blocksize for the given sample rate
            blocksize = int(self.chunk_duration * self.sample_rate)
            
            # Ensure blocksize is a power of 2 for better performance
            blocksize = 2 ** (int(np.log2(blocksize)) + 1)
            
            # Limit blocksize to reasonable bounds
            blocksize = min(max(blocksize, 256), 4096)
            
            # Basic stream settings - compatible with all sounddevice versions
            stream_params = {
                'samplerate': self.sample_rate,
                'channels': self.channels,
                'dtype': 'float32',
                'callback': self._audio_callback,
                'blocksize': blocksize,
                'latency': self.config.get('latency', 'low')
            }
            
            # Only add device if specified
            if input_device is not None:
                stream_params['device'] = input_device
            
            self.stream = sd.InputStream(**stream_params)
            logger.info(f"Audio stream opened successfully with blocksize: {blocksize}")
        except Exception as e:
            logger.error(f"Error opening audio stream: {e}")
            raise
    
    def _audio_callback(self, indata, frames, time_info, status):
        try:
            if status:
                logger.warning(f"Audio stream status: {status}")
            
            # Get the current timestamp
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Convert to mono if needed and ensure float32
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1, dtype=np.float32)
            else:
                audio_data = indata.flatten().astype(np.float32)
            
            # Calculate volume level for debugging
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Print volume level (for debugging)
            print(f"\r[{timestamp}] Audio level: {rms:.6f}", end='', flush=True)
            
            # Add to buffer
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_data))
            
            # Process chunks if we have enough data
            while len(self.audio_buffer) >= self.chunk_samples:
                chunk = self.audio_buffer[:self.chunk_samples]
                self.audio_buffer = self.audio_buffer[self.chunk_samples:]
                
                # Skip silent chunks
                if self._is_silent(chunk):
                    continue
                
                # Print when we're about to process a chunk
                print(f"\n[{timestamp}] Processing audio chunk (length: {len(chunk)} samples)")
                
                # Process chunk in a separate thread
                self.processing_queue.put(chunk)
                
        except Exception as e:
            print(f"\nError in audio callback: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_stream(self):
        """Stop the audio stream."""
        if self.stream:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped")
    
    def _is_silent(self, audio_chunk: np.ndarray) -> bool:
        """Check if the audio chunk is silent."""
        rms = np.sqrt(np.mean(np.square(audio_chunk)))
        return rms < self.silence_threshold
    
    def process_audio_file(self, file_path: str, callback: callable):
        """Process an audio file in chunks and call the callback for each chunk."""
        try:
            logger.info(f"Starting to process audio file: {file_path}")
            
            # Load the entire audio file
            logger.debug(f"Loading audio file with sample rate: {self.sample_rate} Hz")
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            logger.info(f"Loaded audio file - Duration: {len(y)/self.sample_rate:.2f}s, Samples: {len(y)}")
            
            # Calculate number of samples per chunk
            chunk_size = int(self.sample_rate * self.chunk_duration)
            total_chunks = int(np.ceil(len(y) / chunk_size))
            
            logger.info(f"Processing {file_path} in {total_chunks} chunks of {self.chunk_duration}s each")
            
            # Process each chunk
            for i in range(total_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk = y[start:end]
                
                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    padding = chunk_size - len(chunk)
                    logger.debug(f"Padding last chunk with {padding} zeros")
                    chunk = np.pad(chunk, (0, padding), 'constant')
                
                # Calculate RMS level to check for silence
                rms = np.sqrt(np.mean(np.square(chunk)))
                logger.debug(f"Processing chunk {i+1}/{total_chunks} - RMS: {rms:.6f}")
                
                # Only process non-silent chunks
                if rms > self.silence_threshold:
                    logger.debug(f"Processing chunk {i+1} (non-silent)")
                    callback(chunk)
                else:
                    logger.debug(f"Skipping silent chunk {i+1}")
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
            logger.info(f"Successfully processed {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}", exc_info=True)
            raise
    
    @staticmethod
    def load_audio_file(file_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load an audio file and resample if needed."""
        try:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    @staticmethod
    def extract_features(audio: np.ndarray, sr: int = 16000, n_mels: int = 64) -> np.ndarray:
        """Extract mel-spectrogram features from audio."""
        try:
            # Compute mel-scaled spectrogram
            S = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=n_mels,
                fmax=sr//2
            )
            
            # Convert to decibels
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Normalize to [0, 1]
            S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
            
            # Add channel dimension
            S_norm = np.expand_dims(S_norm, axis=-1)
            
            return S_norm
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for model input."""
        # Ensure audio is the correct length
        if len(audio) > self.chunk_samples:
            audio = audio[:self.chunk_samples]
        elif len(audio) < self.chunk_samples:
            audio = np.pad(audio, (0, max(0, self.chunk_samples - len(audio))))
        
        # Extract features
        features = self.extract_features(audio, self.sample_rate)
        
        return features
    
    def play_audio(self, audio: np.ndarray, sample_rate: Optional[int] = None):
        """Play audio through the default audio device."""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            raise
