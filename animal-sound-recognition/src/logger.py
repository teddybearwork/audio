import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml

class DetectionLogger:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the detection logger."""
        self.config = self._load_config(config_path)
        self.log_dir = Path(self.config['output']['directory'])
        self.log_file = self.log_dir / self.config['output']['csv_file']
        self.log_interval = self.config['output']['log_interval']
        
        # Create output directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file with header if it doesn't exist
        self._init_csv_file()
        
        # Buffer for batching writes
        self.buffer: List[Dict[str, Any]] = []
        self.last_write_time = datetime.now()
        
        logging.info(f"Detection logger initialized. Logging to {self.log_file}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            raise
    
    def _init_csv_file(self):
        """Initialize the CSV file with header if it doesn't exist."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'sound', 'confidence'])
    
    def log_detection(self, sound_name: str, confidence: float, timestamp: datetime = None):
        """Log a detection to the CSV file."""
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            # Create a new row as a dictionary
            row = {
                'timestamp': timestamp.isoformat(),
                'sound': sound_name,
                'confidence': float(confidence)
            }
            
            # Print detection to console for debugging
            print(f"\n[Detection] {row['timestamp']} - {sound_name} (Confidence: {confidence:.2f})")
            
            # Add to buffer
            self.buffer.append(row)
            
            # Always write immediately for now (for debugging)
            self._write_buffer()
            
            # Also print the file content to verify
            try:
                with open(self.log_file, 'r') as f:
                    print("\nCurrent log file content:")
                    print(f.read())
            except Exception as e:
                print(f"Error reading log file: {e}")
                
            print(f"[Logger] Logged detection: {row['sound']}")
        except Exception as e:
            print(f"Error in log_detection: {e}")
            import traceback
            traceback.print_exc()
    
    def _write_buffer(self):
        """Write buffered detections to the CSV file."""
        if not self.buffer:
            return
        
        try:
            # Write to CSV
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'sound', 'confidence'])
                writer.writerows(self.buffer)
            
            logging.debug(f"Wrote {len(self.buffer)} detections to {self.log_file}")
            
            # Clear buffer and update last write time
            self.buffer = []
            self.last_write_time = datetime.now()
            
        except Exception as e:
            logging.error(f"Error writing to log file: {e}")
    
    def get_recent_detections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent detections from the log file.
        
        Args:
            limit: Maximum number of detections to return
            
        Returns:
            List of detection dictionaries, most recent first
        """
        if not self.log_file.exists():
            return []
        
        try:
            # Read all rows
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Return most recent rows (skip header if present)
            return rows[-limit:]
            
        except Exception as e:
            logging.error(f"Error reading log file: {e}")
            return []
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the logged detections.
        
        Returns:
            Dictionary containing detection statistics
        """
        if not self.log_file.exists():
            return {}
        
        try:
            # Read all rows
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                return {}
            
            # Calculate statistics
            total_detections = len(rows)
            animal_detections = sum(1 for row in rows if row.get('is_animal_sound', '').lower() == 'true')
            
            # Count detections by class
            class_counts = {}
            for row in rows:
                class_name = row.get('class', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Get most common class
            most_common_class = max(class_counts.items(), key=lambda x: x[1]) if class_counts else ('none', 0)
            
            return {
                'total_detections': total_detections,
                'animal_detections': animal_detections,
                'noise_detections': total_detections - animal_detections,
                'most_common_class': most_common_class[0],
                'class_counts': class_counts,
                'first_detection': rows[0]['timestamp'] if rows else None,
                'last_detection': rows[-1]['timestamp'] if rows else None
            }
            
        except Exception as e:
            logging.error(f"Error calculating detection statistics: {e}")
            return {}
    
    def export_to_json(self, output_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Export detection logs to a JSON file.
        
        Args:
            output_path: Path to the output JSON file (default: log_dir/detections_<timestamp>.json)
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.log_dir / f'detections_{timestamp}.json'
        else:
            output_path = Path(output_path)
        
        try:
            # Read all rows from CSV
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Write to JSON
            with open(output_path, 'w') as f:
                json.dump(rows, f, indent=2)
            
            logging.info(f"Exported {len(rows)} detections to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error exporting to JSON: {e}")
            return False
    
    def clear_logs(self):
        """Clear all detection logs."""
        try:
            if self.log_file.exists():
                os.remove(self.log_file)
            self._init_csv_file()
            self.buffer = []
            logging.info("Cleared all detection logs")
        except Exception as e:
            logging.error(f"Error clearing logs: {e}")
