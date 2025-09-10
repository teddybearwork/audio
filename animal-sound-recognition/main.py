#!/usr/bin/env python3
"""
Animal Sound Recognition System

This is the main entry point for the Animal Sound Recognition System.
It provides a command-line interface for real-time animal sound detection
from either a WebSocket stream or a local microphone.
"""

import argparse
import asyncio
import json
import logging
import numpy as np
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import local modules
from audio_processor import AudioProcessor
from logger import DetectionLogger
from model import AnimalSoundClassifier
from utils import (
    ensure_directory, format_time, get_available_microphones,
    load_config, set_log_level
)
from src.websocket_server import AudioWebSocketServer
from src.audio_matcher import AudioMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnimalSoundRecognizer:
    """Main class for the Animal Sound Recognition System."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the recognizer with configuration."""
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.audio_processor = AudioProcessor(config_path)
        self.classifier = AnimalSoundClassifier(config_path)
        self.audio_matcher = AudioMatcher(self.classifier)  # Initialize the matcher
        self.detection_logger = DetectionLogger(config_path)
        self.websocket_server = AudioWebSocketServer(config_path)
        
        # Set up WebSocket callback
        self.websocket_server.set_audio_callback(self.process_audio_chunk)
        
        # State variables
        self.is_running = False
        self.start_time = None
        self.processed_chunks = 0
        self.detection_count = 0
        
        # Create necessary directories
        for directory in ['logs', 'output', 'models']:
            ensure_directory(directory)
        
        logger.info("Animal Sound Recognizer initialized")
    
    async def start(self, input_source: str = 'microphone', **kwargs):
        """Start the recognition process."""
        if self.is_running:
            logger.warning("Recognizer is already running")
            return
        
        try:
            self.is_running = True
            self.start_time = time.time()
            self.processed_chunks = 0
            self.detection_count = 0
            
            # Set up audio processor
            self.audio_processor = AudioProcessor()
            
            # Set up callback for audio processing
            async def audio_callback(audio_data):
                await self.process_audio_chunk(audio_data)
            
            if input_source == 'microphone':
                # Start microphone input
                device = kwargs.get('device')
                await self.audio_processor.start_stream(audio_callback, device_index=device)
                
                # Keep the main thread alive
                while self.is_running:
                    await asyncio.sleep(0.1)
                    
            elif input_source == 'file':
                # Process audio file
                audio_file = kwargs.get('audio_file')
                if not audio_file:
                    raise ValueError("audio_file parameter is required for file input")
                
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Store the original callback
                original_callback = audio_callback
                
                # Create a synchronous callback that runs the coroutine in the thread's event loop
                def sync_callback(chunk):
                    try:
                        # Run the coroutine in this thread's event loop
                        loop.run_until_complete(original_callback(chunk))
                    except Exception as e:
                        logger.error(f"Error in audio callback: {e}", exc_info=True)
                
                # Process the file in a separate thread to avoid blocking
                def process_file():
                    try:
                        # Process the file in chunks using the synchronous callback
                        self.audio_processor.process_audio_file(audio_file, sync_callback)
                    except Exception as e:
                        logger.error(f"Error processing file: {e}", exc_info=True)
                    finally:
                        # Clean up the event loop
                        loop.close()
                
                # Start file processing in a thread
                import threading
                file_thread = threading.Thread(target=process_file, daemon=True)
                file_thread.start()
                
                # Keep the main thread alive while processing
                while file_thread.is_alive() and self.is_running:
                    await asyncio.sleep(0.1)
                    
            elif input_source == 'websocket':
                # Update config with host and port if provided
                host = kwargs.get('host', '0.0.0.0')
                port = kwargs.get('port', 8000)
                self.websocket_server.host = host
                self.websocket_server.port = port
                
                # Start WebSocket server
                await self.websocket_server.start()
                logger.info(f"WebSocket server started on {host}:{port}")
                
                # Keep the main thread alive
                while self.is_running:
                    await asyncio.sleep(1)
                    
            else:
                raise ValueError(f"Unsupported input source: {input_source}")
            
        except Exception as e:
            logger.error(f"Error in recognition loop: {e}", exc_info=True)
            raise
        
        finally:
            self.is_running = False
            logger.info("Recognition stopped")
    
    async def process_audio_chunk(self, audio_data):
        """Process a chunk of audio data."""
        try:
            # Perform matching
            match = self.audio_matcher.match_sound(audio_data)
            
            # Log if a match is found
            if match:
                sound_name, confidence = match
                self.detection_count += 1
                logger.info(f"Matched: {sound_name} (Confidence: {confidence:.2f})")
                
                # Log detection to file
                try:
                    self.detection_logger.log_detection(sound_name, confidence)
                except Exception as e:
                    print(f"[Logger] Error logging detection: {e}")
                    
            if self.processed_chunks % 10 == 0:
                elapsed = time.time() - self.start_time
                logger.debug(
                    f"Processed {self.processed_chunks} chunks "
                    f"({self.detection_count} detections) in {format_time(elapsed)}"
                )
        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
    
    def stop(self):
        """Stop the recognition process."""
        logger.info("Stopping recognition...")
        self.is_running = False
        
        # Stop audio processing
        if hasattr(self, 'audio_processor'):
            self.audio_processor.stop_stream()
            
        # Stop WebSocket server if running
        if hasattr(self, 'websocket_server'):
            try:
                import asyncio
                asyncio.get_event_loop().run_until_complete(self.websocket_server.stop())
            except RuntimeError:
                # If no event loop is running, create a new one
                asyncio.run(self.websocket_server.stop())
            
        # Save detections if logger exists and has a flush method
        if hasattr(self, 'detection_logger') and hasattr(self.detection_logger, 'flush'):
            self.detection_logger.flush()
        logger.info(f"Recognition stopped after {time.time() - self.start_time if self.start_time else 0} seconds")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the recognizer."""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time,
            'processed_chunks': self.processed_chunks,
            'detection_count': self.detection_count,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Animal Sound Recognition System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        choices=['microphone', 'websocket', 'file'],
        help='Input source for audio data (microphone, websocket, or file)'
    )
    
    # File input option
    file_group = parser.add_argument_group('File Input Options')
    file_group.add_argument(
        '--audio-file',
        type=str,
        help='Path to audio file (required when --input=file)'
    )
    
    # WebSocket options
    websocket_group = parser.add_argument_group('WebSocket Options')
    websocket_group.add_argument(
        '--host',
        default='0.0.0.0',
        help='WebSocket server host (default: 0.0.0.0)'
    )
    websocket_group.add_argument(
        '--port',
        type=int,
        default=8000,
        help='WebSocket server port (default: 8000)'
    )
    
    # General options
    parser.add_argument('--list-devices', action='store_true',
                      help='List available audio devices and exit')
    parser.add_argument('--device', type=int, default=None,
                      help='Audio device index to use (default: default device)')
    parser.add_argument('--model', type=str, default=None,
                      help='Path to custom model (default: use YAMNet)')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Minimum confidence threshold (0.0-1.0, default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Output directory for detections (default: output/)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level (default: INFO)')
    
    return parser.parse_args()


def list_audio_devices():
    """List available audio input devices and exit."""
    devices = get_available_microphones()
    
    if not devices:
        print("No audio input devices found.")
        return
    
    print("\nAvailable audio input devices:")
    print("ID\tDefault\tChannels\tSample Rate\tName")
    print("-" * 80)
    
    for device in devices:
        default = "*" if device['default'] else " "
        print(
            f"{device['id']}\t{default}\t"
            f"{device['input_channels']}\t\t"
            f"{int(device['sample_rate'])} Hz\t"
            f"{device['name']}"
        )
    
    print("\n* Default input device")


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    print("\nShutting down...")
    sys.exit(0)


def main():
    """Main entry point."""
    # Parse command line arguments
    try:
        args = parse_arguments()
        
        # Validate file input
        if args.input == 'file':
            if not args.audio_file:
                print("Error: --audio-file is required when --input=file")
                return
            if not os.path.isfile(args.audio_file):
                print(f"Error: Audio file not found: {args.audio_file}")
                return
        
        # Set up logging
        set_log_level(args.log_level)
        
        if args.list_devices:
            list_audio_devices()
            return
            
        # Initialize the recognizer
        recognizer = AnimalSoundRecognizer()
        
        # Start the recognition
        if args.input == 'microphone':
            print("Starting animal sound recognition from microphone...")
            print("Press Ctrl+C to stop")
            asyncio.run(recognizer.start(
                input_source='microphone',
                device=args.device
            ))
        elif args.input == 'file':
            print(f"Processing audio file: {args.audio_file}")
            asyncio.run(recognizer.start(
                input_source='file',
                audio_file=args.audio_file
            ))
        elif args.input == 'websocket':
            print(f"Starting WebSocket server on {args.host}:{args.port}")
            print("Press Ctrl+C to stop")
            asyncio.run(recognizer.start(
                input_source='websocket',
                host=args.host,
                port=args.port
            ))
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        if 'recognizer' in locals():
            recognizer.stop()


if __name__ == "__main__":
    main()
