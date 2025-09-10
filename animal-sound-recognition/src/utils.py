import os
import yaml
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        raise

def save_config(config: Dict[str, Any], config_path: str = 'config.yaml'):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the YAML configuration file
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        raise

def ensure_directory(directory: Union[str, Path]):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise

def list_audio_files(directory: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    """
    List all audio files in a directory with the specified extensions.
    
    Args:
        directory: Directory to search for audio files
        extensions: List of file extensions to include (e.g., ['.wav', '.mp3'])
        
    Returns:
        List of Path objects for the audio files
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(directory.glob(f'*{ext}'))
        audio_files.extend(directory.glob(f'**/*{ext}'))
    
    return sorted(audio_files)

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to the range [-1, 1].
    
    Args:
        audio: Input audio array
        
    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio
    
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to the target sample rate.
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        logger.warning("librosa not installed, using scipy for resampling (less accurate)")
        from scipy import signal
        ratio = target_sr / orig_sr
        return signal.resample(audio, int(len(audio) * ratio))

def time_to_samples(time_sec: float, sample_rate: int) -> int:
    """
    Convert time in seconds to number of samples.
    
    Args:
        time_sec: Time in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Number of samples
    """
    return int(round(time_sec * sample_rate))

def samples_to_time(samples: int, sample_rate: int) -> float:
    """
    Convert number of samples to time in seconds.
    
    Args:
        samples: Number of samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Time in seconds
    """
    return samples / sample_rate

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string (HH:MM:SS.mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    ms = (seconds % 1) * 1000
    
    if h > 0:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int(ms):03d}"
    else:
        return f"{int(m):02d}:{int(s):02d}.{int(ms):03d}"

def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Calculate the duration of an audio signal in seconds.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate in Hz
        
    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate if sample_rate > 0 else 0.0

def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get the size of a file in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in megabytes
    """
    file_path = Path(file_path)
    if file_path.is_file():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0

def get_available_microphones() -> List[Dict[str, Any]]:
    """
    Get a list of available audio input devices.
    
    Returns:
        List of dictionaries containing device information
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Only include input devices
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'sample_rate': device['default_samplerate'],
                    'input_channels': device['max_input_channels'],
                    'output_channels': device['max_output_channels'],
                    'default': (i == sd.default.device[0])
                })
        
        return input_devices
    except (ImportError, OSError) as e:
        logger.warning(f"SoundDevice not available, cannot list microphones: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting microphone list: {e}")
        return []

def set_log_level(level: str = 'INFO'):
    """
    Set the logging level for all loggers.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = level.upper()
    numeric_level = getattr(logging, level, None)
    
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(level=numeric_level)
    logger.setLevel(numeric_level)
    
    # Update log level for other loggers
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(numeric_level)
    
    logger.info(f"Log level set to {level}")
