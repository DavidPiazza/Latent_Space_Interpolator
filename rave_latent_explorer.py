#!/usr/bin/env python3
"""
RAVE Latent Space Explorer

Generates audio by sampling the latent space of a pre-trained RAVE model
and analyzes the resulting audio using librosa audio descriptors.
Outputs a JSON dataset suitable for fluid.dataset~ in Max/MSP.

Author: David Piazza (based on concepts from original script by Moisés Horta Valenzuela)
Date: 2024-06-05
"""

import argparse
import torch
import librosa as li
import numpy as np
from tqdm import tqdm
import os
import json
import time

# Import necessary tools for normalization and UMAP
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

print("Script executing... Imports successful.")

# Define the audio features to extract
# Using mean of features over the clip
FEATURE_EXTRACTORS = {
    "mfcc": lambda y, sr: np.mean(li.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1),
    "spectral_centroid": lambda y, sr: np.mean(li.feature.spectral_centroid(y=y, sr=sr)),
    "spectral_bandwidth": lambda y, sr: np.mean(li.feature.spectral_bandwidth(y=y, sr=sr)),
    "spectral_contrast": lambda y, sr: np.mean(li.feature.spectral_contrast(y=y, sr=sr), axis=1),
    "spectral_flatness": lambda y, sr: np.mean(li.feature.spectral_flatness(y=y)),
    "rms": lambda y, sr: np.mean(li.feature.rms(y=y)),
    "zero_crossing_rate": lambda y, sr: np.mean(li.feature.zero_crossing_rate(y=y)),
    "chroma_stft": lambda y, sr: np.mean(li.feature.chroma_stft(y=y, sr=sr), axis=1),
}

def get_model_dimensions(model_path, device, sr=48000):
    """Load a RAVE model and determine its latent dimensions."""
    try:
        print(f"Loading model to determine latent dimensions: {model_path}")
        rave = torch.jit.load(model_path).to(device)
        rave.eval()

        # Create a short test signal to determine encode output shape
        test_signal_length = max(sr // 4, 1024) # Ensure sufficient length
        test_signal = torch.zeros(1, 1, test_signal_length).float().to(device)

        with torch.no_grad():
            try:
                z = rave.encode(test_signal) # Shape (1, n_dims, time_steps)
                latent_dim = z.shape[1]
            except RuntimeError as e:
                 # Handle potential kernel size errors if test signal is too short despite checks
                if "Kernel size can't be greater than actual input size" in str(e):
                    print("Initial test signal too short, trying with a longer signal...")
                    test_signal = torch.zeros(1, 1, sr).float().to(device) # 1 second
                    z = rave.encode(test_signal)
                    latent_dim = z.shape[1]
                else:
                    raise e # Re-raise other runtime errors

        print(f"Model has {latent_dim} latent dimensions.")
        return rave, latent_dim

    except Exception as e:
        print(f"Error determining model dimensions: {str(e)}")
        raise # Re-raise after printing

def generate_latent_samples(n_samples, n_dims, min_val, max_val):
    """Generate random samples in the latent space."""
    print(f"Generating {n_samples} random samples for {n_dims} dimensions between {min_val} and {max_val}...")
    samples = np.random.uniform(min_val, max_val, size=(n_samples, n_dims))
    return samples

def decode_latent_vector(rave, z_sample, device, num_frames=1):
    """Decode a latent vector sample (repeated num_frames times) to an audio waveform."""
    # Ensure z_sample is a numpy array first if it isn't
    if isinstance(z_sample, torch.Tensor):
        z_sample = z_sample.cpu().numpy()

    # Reshape the latent vector and repeat it 'num_frames' times along the time axis
    # Original z_sample shape: (n_dims,)
    # Target z_tensor shape: (batch_size=1, n_dims, num_frames)
    z_sample_reshaped = z_sample.reshape(1, -1, 1) # Shape (1, n_dims, 1)
    z_repeated = np.tile(z_sample_reshaped, (1, 1, num_frames)) # Shape (1, n_dims, num_frames)
    z_tensor = torch.from_numpy(z_repeated).float().to(device)

    with torch.no_grad():
        start_time = time.time()
        audio_out = rave.decode(z_tensor)
        # print(f"Decoding time: {time.time() - start_time:.4f}s") # Optional: timing info

    # Reshape audio output and move to CPU
    audio_np = audio_out.squeeze().cpu().numpy()
    return audio_np

def extract_audio_features(audio_np, sr):
    """Extract pre-defined audio features from a waveform."""
    features = []
    feature_names = []

    if np.sum(np.abs(audio_np)) < 1e-6: # Check for near silence
        print("Warning: Generated audio is nearly silent. Features might be zero or NaN.")
        # Return zeros for all expected features if silent
        num_features = 0
        for name, func in FEATURE_EXTRACTORS.items():
             # Estimate output size based on typical feature shapes
            if name in ["mfcc", "spectral_contrast", "chroma_stft"]:
                 # These often return multiple values (e.g., 13 MFCCs)
                 # We need a way to know the expected size. Let's run on dummy data.
                 dummy_data = np.random.randn(sr // 2) # 0.5s dummy audio
                 try:
                    dummy_feat = func(dummy_data, sr)
                    num_features += len(dummy_feat) if isinstance(dummy_feat, np.ndarray) and dummy_feat.ndim > 0 else 1
                    feature_names.extend([f"{name}_{i}" for i in range(len(dummy_feat))] if isinstance(dummy_feat, np.ndarray) and dummy_feat.ndim > 0 else [name])
                 except Exception:
                    # Fallback if dummy calculation fails (should not happen often)
                    if name == "mfcc": num_features += 13; feature_names.extend([f"mfcc_{i}" for i in range(13)])
                    elif name == "spectral_contrast": num_features += 7; feature_names.extend([f"spectral_contrast_{i}" for i in range(7)]) # Typically 7 bands (6 contrast + DC)
                    elif name == "chroma_stft": num_features += 12; feature_names.extend([f"chroma_{i}" for i in range(12)])
                    else: num_features += 1; feature_names.append(name)
            else:
                 num_features += 1
                 feature_names.append(name)

        print(f"Estimated total features: {num_features}")
        return np.zeros(num_features).tolist(), feature_names # Return list of zeros

    all_feature_names = []
    for name, func in FEATURE_EXTRACTORS.items():
        try:
            feature_val = func(y=audio_np, sr=sr)
            # Flatten in case a feature extractor returns multiple values (like MFCCs)
            if isinstance(feature_val, np.ndarray):
                if feature_val.ndim > 0: # It's an array of values
                     features.extend(feature_val.flatten().tolist())
                     all_feature_names.extend([f"{name}_{i}" for i in range(len(feature_val.flatten()))])
                else: # It's a scalar wrapped in a 0-dim array
                    features.append(float(feature_val))
                    all_feature_names.append(name)
            else: # It's a scalar
                features.append(float(feature_val))
                all_feature_names.append(name)
        except Exception as e:
            print(f"Warning: Could not calculate feature '{name}': {str(e)}")
            # Append NaN or zero? Let's use NaN to indicate failure.
            # Need to know the expected size if it failed. This is tricky.
            # For now, let's just skip failed features, which might lead to inconsistent row sizes.
            # A better approach is to pre-determine size and fill with NaNs.
            # Let's try the dummy calculation approach again here.
            dummy_data = np.random.randn(sr // 2)
            try:
                dummy_feat = func(dummy_data, sr)
                feat_size = len(dummy_feat.flatten()) if isinstance(dummy_feat, np.ndarray) else 1
                features.extend([np.nan] * feat_size)
                if feat_size > 1:
                    all_feature_names.extend([f"{name}_{i}" for i in range(feat_size)])
                else:
                    all_feature_names.append(name)
            except Exception: # Fallback if dummy calculation also fails
                 # Use estimated sizes
                 if name == "mfcc": feat_size = 13; fn = [f"mfcc_{i}" for i in range(13)]
                 elif name == "spectral_contrast": feat_size = 7; fn = [f"spectral_contrast_{i}" for i in range(7)]
                 elif name == "chroma_stft": feat_size = 12; fn = [f"chroma_{i}" for i in range(12)]
                 else: feat_size = 1; fn = [name]
                 features.extend([np.nan] * feat_size)
                 all_feature_names.extend(fn)


    # Replace any potential NaNs or Infs resulting from calculations (e.g., division by zero in silent parts)
    features_cleaned = np.nan_to_num(np.array(features), nan=0.0, posinf=0.0, neginf=0.0)

    return features_cleaned.tolist(), all_feature_names


def main():
    print("Executing main block...")
    parser = argparse.ArgumentParser(description="Explore RAVE latent space by decoding samples and analyzing audio.")
    parser.add_argument("model_path", type=str, help="Path to the pre-trained RAVE model (.ts file).")
    parser.add_argument("output_json", type=str, help="Path stem for saving the output JSON datasets (e.g., 'output_data').")
    parser.add_argument("--num_samples", "-n", type=int, default=1000, help="Number of random latent vectors to sample (default: 1000).")
    parser.add_argument("--min_val", type=float, default=-2.0, help="Minimum value for latent dimensions during sampling (default: -2.0).")
    parser.add_argument("--max_val", type=float, default=2.0, help="Maximum value for latent dimensions during sampling (default: 2.0).")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate for audio generation and analysis (default: 48000).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "mps", "cpu"], help="Device to use (cuda, mps, or cpu - default: cpu).")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of identical latent frames to decode consecutively (default: 1).")

    args = parser.parse_args()

    # Ensure output base name doesn't end with .json yet
    output_base = args.output_json
    if output_base.lower().endswith('.json'):
         output_base = os.path.splitext(output_base)[0]

    # Define output filenames based on the stem
    output_features_json = f"{output_base}.json"
    output_latents_json = f"{output_base}_latent_vectors.json"
    output_main_metadata_json = f"{output_base}_metadata.json"
    output_umap_json = f"{output_base}_umap_2d.json"
    output_umap_metadata_json = f"{output_base}_umap_metadata.json"

    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return
    output_dir = os.path.dirname(output_base)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
         device = torch.device("mps")
         try:
             test_tensor = torch.tensor([1.0, 2.0]).to(device)
             if test_tensor.device.type != 'mps': raise RuntimeError("MPS check failed")
             print("Using MPS device.")
         except Exception as e:
             print(f"Warning: MPS device requested but failed test ({e}). Falling back to CPU.")
             device = torch.device("cpu")
    else:
        if args.device != "cpu":
            print(f"Warning: Device '{args.device}' not available. Using CPU.")
        device = torch.device("cpu")
    print(f"Using device: {device}")


    # --- Main Processing ---
    try:
        # 1. Load model and get dimensions
        rave_model, n_dims = get_model_dimensions(args.model_path, device, args.sr)

        # 2. Generate latent samples
        latent_samples = generate_latent_samples(args.num_samples, n_dims, args.min_val, args.max_val)

        # 3. Process samples
        results_data = {}
        num_features = None
        feature_names = None # Store the names from the first successful analysis

        print(f"Decoding {args.num_samples} samples (each {args.num_frames} frames) and extracting features...")
        for i in range(args.num_samples): # Removed tqdm here for clearer print output
            print(f"\n--- Processing sample {i+1}/{args.num_samples} ---")
            z_sample = latent_samples[i]

            # Decode using the specified number of frames
            print(f"  Decoding sample {i+1}...")
            start_decode_time = time.time()
            audio_np = decode_latent_vector(rave_model, z_sample, device, num_frames=args.num_frames)
            print(f"  Decoding done ({time.time() - start_decode_time:.2f}s). Audio length: {len(audio_np)} samples ({len(audio_np)/args.sr:.2f}s)")

            # Extract features
            print(f"  Extracting features for sample {i+1}...")
            start_feature_time = time.time()
            current_features, current_feature_names = extract_audio_features(audio_np, args.sr)
            print(f"  Feature extraction done ({time.time() - start_feature_time:.2f}s).")

            if num_features is None and current_features is not None and len(current_features) > 0:
                num_features = len(current_features)
                feature_names = current_feature_names
                print(f"\nDetermined number of features: {num_features}")
                # print(f"Feature names: {feature_names}") # Optional: print feature names

            # Ensure consistent feature vector length
            if current_features is not None and num_features is not None and len(current_features) != num_features:
                 print(f"Warning: Sample {i} yielded {len(current_features)} features, expected {num_features}. Padding with zeros.")
                 padded_features = np.zeros(num_features)
                 copy_len = min(len(current_features), num_features)
                 padded_features[:copy_len] = current_features[:copy_len]
                 current_features = padded_features.tolist()
            elif current_features is None and num_features is not None:
                 print(f"Warning: Feature extraction failed for sample {i}. Filling with zeros.")
                 current_features = np.zeros(num_features).tolist()
            elif current_features is None and num_features is None:
                 # Should not happen if the first sample worked, but handle defensively
                 print(f"Warning: Feature extraction failed for sample {i} and num_features unknown. Skipping sample.")
                 continue # Skip this sample

            sample_id = f"sample_{i}"
            # Add print before storing
            print(f"  Storing results for sample {i+1} (ID: {sample_id}).")
            results_data[sample_id] = {
                "latent_vector": z_sample.tolist(),
                "features": current_features
            }

        # --- Post-processing: Normalization and UMAP ---
        sample_ids = list(results_data.keys()) # Get IDs in consistent order
        if not sample_ids:
            print("Error: No samples were successfully processed.")
            return

        # Extract original latents and features
        original_latent_matrix = np.array([results_data[sid]["latent_vector"] for sid in sample_ids])
        feature_matrix = np.array([results_data[sid]["features"] for sid in sample_ids])

        if feature_matrix.size == 0 or num_features is None or num_features == 0:
             print("Warning: No features extracted or feature matrix is empty. Skipping normalization and UMAP.")
             # Still save latents if they exist
             if original_latent_matrix.size > 0:
                 latent_output_data = { "cols": n_dims, "data": {sid: results_data[sid]["latent_vector"] for sid in sample_ids} }
                 print(f"Saving original latent vectors to {output_latents_json}...")
                 with open(output_latents_json, "w") as f: json.dump(latent_output_data, f, indent=2)
             else:
                 print("No data to save.")
             return # Exit if no features
        else:
            print(f"\nOriginal feature range: min={np.min(feature_matrix):.4f}, max={np.max(feature_matrix):.4f}")

            # 5. Normalize features to [0, 1]
            print("Normalizing features to range [0, 1]...")
            feature_scaler = MinMaxScaler()
            features_normalized = feature_scaler.fit_transform(feature_matrix)
            print(f"Normalized feature range: min={np.min(features_normalized):.4f}, max={np.max(features_normalized):.4f}")

            # --- UMAP Processing ---
            print("\nApplying UMAP to reduce features to 2 dimensions...")
            try:
                umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                umap_results = umap_model.fit_transform(features_normalized)

                # Scale UMAP results to [0, 1] range
                umap_scaler = MinMaxScaler()
                umap_results_scaled = umap_scaler.fit_transform(umap_results)
                print(f"UMAP results scaled to range: min={np.min(umap_results_scaled):.4f}, max={np.max(umap_results_scaled):.4f}")

                # Prepare UMAP output data (fluid.dataset~ format)
                umap_output_data = {
                    "cols": 2,
                    "data": { sample_ids[i]: umap_results_scaled[i].tolist() for i in range(len(sample_ids)) }
                }
                # Save UMAP results
                print(f"Saving 2D UMAP results to {output_umap_json}...")
                with open(output_umap_json, "w") as f: json.dump(umap_output_data, f, indent=2)

                # Prepare and save UMAP metadata
                umap_metadata = {
                    "original_output_file": output_features_json,
                    "num_samples": len(sample_ids),
                    "original_dimensions": num_features,
                    "reduced_dimensions": 2,
                    "feature_normalization": "MinMax [0, 1]",
                    "umap_normalization": "MinMax [0, 1]",
                    "umap_n_neighbors": umap_model.n_neighbors,
                    "umap_min_dist": umap_model.min_dist,
                    "umap_metric": umap_model.metric,
                    "latent_dimensions": n_dims,
                    "model_path": args.model_path,
                    "decoded_frames_per_sample": args.num_frames
                }
                print(f"Saving UMAP metadata to {output_umap_metadata_json}...")
                with open(output_umap_metadata_json, "w") as f: json.dump(umap_metadata, f, indent=2)

            except Exception as e:
                 print(f"\nError during UMAP processing: {str(e)}")
                 print("Skipping UMAP output files.")
                 # import traceback # Optionally uncomment for full traceback
                 # traceback.print_exc()


            # --- Final Output Saving ---

            # 1. Save Main Normalized Features JSON (fluid.dataset~ format)
            main_features_output_data = {
                "cols": num_features,
                "data": { sample_ids[i]: features_normalized[i].tolist() for i in range(len(sample_ids)) }
            }
            print(f"\nSaving main dataset (normalized features) to {output_features_json}...")
            with open(output_features_json, "w") as f: json.dump(main_features_output_data, f, indent=2)

            # 2. Save Latent Vectors JSON (fluid.dataset~ format)
            # Latent vectors were extracted earlier, use original_latent_matrix
            latent_output_data = {
                "cols": n_dims,
                "data": { sample_ids[i]: original_latent_matrix[i].tolist() for i in range(len(sample_ids)) }
            }
            print(f"Saving original latent vectors to {output_latents_json}...")
            with open(output_latents_json, "w") as f: json.dump(latent_output_data, f, indent=2)

            # 3. Save Main Metadata JSON (without latent vectors)
            metadata = {
                "model_path": args.model_path,
                "num_samples": len(sample_ids),
                "latent_dimensions": n_dims,
                "latent_range": [args.min_val, args.max_val],
                "sample_rate": args.sr,
                "feature_names": feature_names if feature_names else [],
                "feature_normalization": "MinMax [0, 1]",
                "decoded_frames_per_sample": args.num_frames
            }
            print(f"Saving main metadata to {output_main_metadata_json}...")
            with open(output_main_metadata_json, "w") as f: json.dump(metadata, f, indent=2)

        print("\nDone.")

    except Exception as e:
        print(f"\nAn critical error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
