import asyncio
import json
import logging
import numpy as np
import websockets
from typing import Dict, Set, Optional, Callable, Any
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class AudioWebSocketServer:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the WebSocket server for audio streaming."""
        self.config = self._load_config(config_path)
        self.host = self.config['websocket']['host']
        self.port = self.config['websocket']['port']
        self.ping_interval = self.config['websocket']['ping_interval']
        self.ping_timeout = self.config['websocket']['ping_timeout']
        self.max_size = self.config['websocket']['max_size']
        
        # Client tracking
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server: Optional[asyncio.AbstractServer] = None
        
        # Audio processing callback
        self.audio_callback: Optional[Callable[[np.ndarray], None]] = None
        
        logger.info("WebSocket server initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
    
    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set the callback function for processing audio data."""
        self.audio_callback = callback
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle a new WebSocket client connection."""
        # Register client
        self.clients.add(websocket)
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"New client connected: {client_ip}")
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'status',
                'message': 'Connected to audio streaming server',
                'sample_rate': self.config['audio']['sample_rate'],
                'channels': self.config['audio']['channels'],
                'chunk_duration': self.config['audio']['chunk_duration']
            }))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    # Handle binary audio data
                    if isinstance(message, bytes):
                        if self.audio_callback:
                            # Convert bytes to numpy array (assuming float32)
                            audio_data = np.frombuffer(message, dtype=np.float32)
                            # Process audio in a separate task to avoid blocking
                            asyncio.create_task(self._process_audio(audio_data))
                    
                    # Handle text messages (JSON)
                    elif isinstance(message, str):
                        try:
                            data = json.loads(message)
                            await self._handle_json_message(websocket, data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON received from {client_ip}")
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': 'Invalid JSON format'
                            }))
                    
                except Exception as e:
                    logger.error(f"Error processing message from {client_ip}: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': f'Error processing message: {str(e)}'
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_ip}")
        
        finally:
            # Unregister client
            self.clients.discard(websocket)
    
    async def _process_audio(self, audio_data: np.ndarray):
        """Process incoming audio data using the registered callback."""
        try:
            if self.audio_callback:
                await asyncio.get_event_loop().run_in_executor(
                    None,  # Use default executor (thread pool)
                    self.audio_callback,
                    audio_data
                )
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
    
    async def _handle_json_message(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle incoming JSON messages from clients."""
        message_type = data.get('type', '').lower()
        
        if message_type == 'ping':
            await websocket.send(json.dumps({'type': 'pong'}))
        
        elif message_type == 'get_status':
            await websocket.send(json.dumps({
                'type': 'status',
                'clients_connected': len(self.clients),
                'sample_rate': self.config['audio']['sample_rate'],
                'channels': self.config['audio']['channels'],
                'chunk_duration': self.config['audio']['chunk_duration']
            }))
        
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {message_type}'
            }))
    
    async def broadcast_detection(self, detection: Dict[str, Any]):
        """Broadcast a detection result to all connected clients."""
        if not self.clients:
            return
        
        message = json.dumps({
            'type': 'detection',
            'timestamp': detection.get('timestamp'),
            'class': detection.get('class'),
            'confidence': detection.get('confidence'),
            'is_animal_sound': detection.get('is_animal_sound', False)
        })
        
        # Send to all connected clients
        tasks = [client.send(message) for client in self.clients]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            max_size=self.max_size
        )
        
        # Log server info
        for socket in self.server.sockets:
            host, port = socket.getsockname()[:2]
            logger.info(f"Server listening on {host}:{port}")
        
        # Keep the server running
        await self.server.wait_closed()
    
    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")
    
    def run(self):
        """Run the WebSocket server (blocking)."""
        try:
            asyncio.get_event_loop().run_until_complete(self.start())
        except KeyboardInterrupt:
            logger.info("Shutting down WebSocket server...")
            asyncio.get_event_loop().run_until_complete(self.stop())
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            raise
