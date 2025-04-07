# rave_osc_server.py

import argparse
import torch
import os
from pythonosc import dispatcher, osc_server, udp_client

# --- Configuration ---
DEFAULT_SERVER_IP = "127.0.0.1"
DEFAULT_SERVER_PORT = 9000
DEFAULT_CLIENT_IP = "127.0.0.1"
DEFAULT_CLIENT_PORT = 9001
OSC_ADDRESS_LOAD = "/rave/load/model"
OSC_ADDRESS_DIMENSIONS = "/rave/info/dimensions"
DEFAULT_SR = 48000 # Default Sample Rate for dummy signal

# --- Global Variables ---
client = None
inference_device = None # Device will be set in main block

# --- RAVE Model Handling (New Function) ---
def get_rave_dimensions(model_path, device, sr=DEFAULT_SR):
    """
    Load a RAVE model and determine its latent dimensions by encoding a test signal.
    Adapted from rave_latent_explorer.py.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model to determine latent dimensions: {model_path}")
    try:
        rave = torch.jit.load(model_path).to(device)
        rave.eval()

        # Create a test signal to determine encode output shape
        # Use a length that's likely safe for common RAVE models
        test_signal_length = max(sr // 2, 4096) # 0.5 sec or min 4096 samples
        test_signal = torch.zeros(1, 1, test_signal_length).float().to(device)
        print(f"Using test signal length: {test_signal_length} on device: {device}")


        with torch.no_grad():
            try:
                # Ensure model has encode method
                if not hasattr(rave, 'encode'):
                     raise AttributeError("Model does not have an 'encode' method.")
                z = rave.encode(test_signal) # Expected shape: (1, n_dims, time_steps)
                if not isinstance(z, torch.Tensor) or z.ndim != 3 or z.shape[0] != 1:
                     raise ValueError(f"Unexpected output shape from encode: {z.shape if isinstance(z, torch.Tensor) else type(z)}")
                latent_dim = z.shape[1] # Dimension is the second element
            except RuntimeError as e:
                # Handle potential kernel size errors if test signal is too short
                if "Kernel size can't be greater than actual input size" in str(e) or \
                   "pad value is out of bounds" in str(e): # Added check for padding errors
                    print("Initial test signal potentially too short or causing padding issues, trying with a longer signal...")
                    test_signal_length = sr * 2 # Try 2 seconds
                    test_signal = torch.zeros(1, 1, test_signal_length).float().to(device)
                    print(f"Retrying with test signal length: {test_signal_length}")
                    z = rave.encode(test_signal)
                    if not isinstance(z, torch.Tensor) or z.ndim != 3 or z.shape[0] != 1:
                         raise ValueError(f"Unexpected output shape from encode on retry: {z.shape if isinstance(z, torch.Tensor) else type(z)}")
                    latent_dim = z.shape[1]
                else:
                    raise e # Re-raise other runtime errors
            except AttributeError as e: # Catch if encode method doesn't exist
                print(f"AttributeError during encode: {e}")
                raise # Re-raise to be caught by the outer handler

        print(f"Model has {latent_dim} latent dimensions.")
        # We don't need the model itself anymore, just the dimension count
        del rave
        if 'mps' in str(device): # Small delay sometimes needed for MPS memory release
           torch.mps.synchronize()
        elif 'cuda' in str(device):
           torch.cuda.empty_cache()


        return latent_dim

    except FileNotFoundError: # Already handled above, but keep for clarity
         raise
    except AttributeError as e: # Catch attribute errors during loading or checking encode
         print(f"AttributeError loading or preparing model {model_path}: {e}")
         raise AttributeError(f"Model structure issue or missing method: {e}")
    except ValueError as e: # Catch value errors from shape checks
        print(f"ValueError during dimension extraction for {model_path}: {e}")
        raise ValueError(f"Could not determine dimensions due to unexpected encode output: {e}")
    except RuntimeError as e: # Catch other runtime errors (like memory issues, incompatible ops)
         print(f"RuntimeError during dimension extraction for {model_path}: {e}")
         raise RuntimeError(f"Torch runtime error during encode: {e}")
    except Exception as e:
        print(f"Unexpected error determining model dimensions for {model_path}: {type(e).__name__} - {e}")
        # import traceback # Optional: for debugging
        # traceback.print_exc()
        raise Exception(f"General error processing model: {e}") # Re-raise as a generic exception


# --- OSC Message Handler ---
def handle_load_model(address, *args):
    """Handles incoming OSC messages to load a RAVE model and get dimensions."""
    global inference_device # Ensure we use the globally set device

    print(f"Received message on {address}: {args}")
    if not args or not isinstance(args[0], str):
        print("Error: Expected a string argument (model path).")
        return
    if inference_device is None:
        print("Error: Inference device not set. Cannot process model.")
        # Optionally send OSC error
        # if client: client.send_message("/rave/error", "Server device not initialized")
        return

    raw_model_path = args[0]
    model_path = raw_model_path

    # Check for macOS absolute path format (e.g., "Macintosh HD:/Users/...")
    path_parts = raw_model_path.split(':', 1)
    if len(path_parts) == 2 and path_parts[1].startswith('/'):
        model_path = path_parts[1]
        print(f"Detected macOS path, using: {model_path}")
    else:
        print(f"Using path as received: {model_path}")

    try:
        print(f"Attempting to get dimensions for: {model_path} using device: {inference_device}")
        latent_dim = get_rave_dimensions(model_path, inference_device) # Use the new function
        print(f"Successfully determined model latent dimensions: {latent_dim}")

        # Send the dimension back via OSC
        if client:
            client.send_message(OSC_ADDRESS_DIMENSIONS, int(latent_dim))
            print(f"Sent dimension {latent_dim} to {client._address}:{client._port}{OSC_ADDRESS_DIMENSIONS}")
        else:
            print("OSC client not initialized, cannot send dimension back.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        # Optionally send OSC error
        # if client: client.send_message("/rave/error", f"File not found: {model_path}")
    except (AttributeError, ValueError, RuntimeError, Exception) as e: # Catch errors from get_rave_dimensions
        print(f"Error processing model: {e}")
        # Optionally send OSC error
        # if client: client.send_message("/rave/error", f"Error processing model: {e}")


# --- Main Server Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=DEFAULT_SERVER_IP,
                        help="The IP address the OSC Server will listen on.")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT,
                        help="The port the OSC Server will listen on.")
    parser.add_argument("--client-ip", default=DEFAULT_CLIENT_IP,
                        help="The IP address to send dimension results to.")
    parser.add_argument("--client-port", type=int, default=DEFAULT_CLIENT_PORT,
                        help="The port to send dimension results to.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "mps", "cpu"],
                        help="Device to use for model loading (cuda, mps, or cpu - default: cpu).")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR,
                        help=f"Sample rate for dummy signal used in dimension check (default: {DEFAULT_SR}).")
    args = parser.parse_args()

    # Set device globally
    if args.device == "cuda" and torch.cuda.is_available():
        inference_device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
         # Add extra check for MPS build
         try:
             test_tensor = torch.tensor([1.0, 2.0]).to(torch.device("mps"))
             if test_tensor.device.type != 'mps': raise RuntimeError("MPS check failed")
             inference_device = torch.device("mps")
             print("Using MPS device.")
         except Exception as e:
             print(f"Warning: MPS device requested but failed test ({e}). Falling back to CPU.")
             inference_device = torch.device("cpu")
    else:
        if args.device != "cpu":
            print(f"Warning: Device '{args.device}' not available or requested. Using CPU.")
        inference_device = torch.device("cpu")
    print(f"Selected inference device: {inference_device}")


    # Initialize OSC client
    client = udp_client.SimpleUDPClient(args.client_ip, args.client_port)
    print(f"OSC client configured to send to {args.client_ip}:{args.client_port}")


    # Initialize OSC dispatcher and server
    osc_dispatcher = dispatcher.Dispatcher()
    # Pass the sample rate to the handler if needed, though get_rave_dimensions uses its own default for now
    osc_dispatcher.map(OSC_ADDRESS_LOAD, handle_load_model)

    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), osc_dispatcher)
    print(f"Serving on {server.server_address}")
    print(f"Listening for messages on {OSC_ADDRESS_LOAD}")
    print(f"Using sample rate {args.sr} for dimension check test signal.")


    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping OSC Server.")
        server.server_close() 