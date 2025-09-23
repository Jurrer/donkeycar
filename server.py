
import asyncio
import base64
import json
import logging
import time
from io import BytesIO
import os
import donkeycar as dk
import numpy as np
import websockets
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RemoteControlServer:
    def __init__(self, host, port, model_path, model_type):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.cfg = None

    def load_model(self):
        """Load the Donkey Car model."""
        logging.info(f"Loading model: {self.model_path} of type: {self.model_type}")

        # conf = os.path.expanduser("./deeplearning_donkeycar")
        self.cfg = dk.load_config(myconfig='myconfig.py')

        # Ensure image dimensions are set for the model
        self.cfg.IMAGE_W = 160
        self.cfg.IMAGE_H = 120
        self.cfg.IMAGE_DEPTH = 3

        self.model = dk.utils.get_model_by_type(self.model_type, self.cfg)
        self.model.load(self.model_path)
        logging.info("Model loaded successfully.")

    async def handle_connection(self, websocket, path):
        """Handle a new WebSocket connection from a car."""
        logging.info(f"Car connected from {websocket.remote_address}")
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                image_b64 = data['image']
                
                # Decode image
                image_bytes = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_bytes))
                img_arr = np.array(image)

                # Perform inference
                start_time = time.time()
                steering, throttle = self.model.run(img_arr)
                end_time = time.time()

                logging.info(f"Inference time: {((end_time - start_time) * 1000):.2f}ms | Steering: {steering:.3f}, Throttle: {throttle:.3f}")

                # Send response back to the car
                response = {'steering': float(steering), 'throttle': float(throttle)}
                await websocket.send(json.dumps(response))

        except websockets.exceptions.ConnectionClosed as e:
            logging.warning(f"Connection closed by car: {e}")
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

    async def start(self):
        """Start the WebSocket server."""
        self.load_model()
        logging.info(f"Starting server on ws://{self.host}:{self.port}")
        server = await websockets.serve(self.handle_connection, self.host, self.port)
        await server.wait_closed()

if __name__ == "__main__":
    # Configuration
    SERVER_HOST = "0.0.0.0"  # Listen on all available network interfaces
    SERVER_PORT = 9000
    # Make sure this path points to your trained model
    MODEL_PATH = "models/pilot_25-09-22_0.tflite" 
    MODEL_TYPE = "tflite_linear"

    server = RemoteControlServer(SERVER_HOST, SERVER_PORT, MODEL_PATH, MODEL_TYPE)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logging.info("Server shutting down.")