"""
SPHEREx Generative Art Framework (Clean & Optimized)
===================================================

A comprehensive Python framework for creating generative art using SPHEREx data
with proper error handling, type hints, and clean code structure.

Author: Digital Media Artist specializing in Scientific Data Art
Date: July 2025
"""

import os
import gc
import io
import warnings
import weakref
import atexit
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

# Set matplotlib backend before importing
os.environ['MPLBACKEND'] = 'TkAgg'

import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import colorsys
from scipy.ndimage import gaussian_filter, zoom
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from astropy.io import fits

# Configure warnings
warnings.filterwarnings('ignore')
plt.ion()  # Interactive mode

# Register cleanup on exit
def cleanup_on_exit():
    """Clean up resources on script exit"""
    plt.close('all')
    gc.collect()

atexit.register(cleanup_on_exit)

# Configuration
SPHEREX_BUCKET = 'nasa-irsa-spherex'
AWS_REGION = 'us-east-1'
DEFAULT_IMAGE_SIZE = (256, 256)
MAX_PARTICLES = 30
MAX_ITERATIONS = 10
CHUNK_SIZE = 1024 * 1024  # 1MB


class SpectralBand(Enum):
    """SPHEREx spectral bands with wavelength ranges and resolutions"""
    BAND1 = (0.75, 1.1, 39)
    BAND2 = (1.1, 1.6, 41)
    BAND3 = (1.6, 2.4, 41)
    BAND4 = (2.4, 3.8, 35)
    BAND5 = (3.8, 4.4, 112)
    BAND6 = (4.4, 5.1, 128)


@dataclass
class ArtisticPalette:
    """Color palettes inspired by cosmic phenomena"""
    name: str
    colors: List[Tuple[float, float, float]]
    wavelength_map: Dict[float, Tuple[float, float, float]]


class SPHERExDataAccess:
    """Handle data access from SPHEREx AWS S3 buckets"""

    def __init__(self, cache_dir: str = "./spherex_cache"):
        self.s3 = boto3.client(
            's3',
            region_name=AWS_REGION,
            config=Config(signature_version=UNSIGNED)
        )
        self.bucket = SPHEREX_BUCKET
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._open_files = weakref.WeakSet()

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup_resources()

    def cleanup_resources(self):
        """Clean up any open resources"""
        for file_handle in self._open_files:
            try:
                file_handle.close()
            except Exception:
                pass
        self._open_files.clear()

    def list_available_data(self, prefix: str = 'qr/level2/', max_keys: int = 100) -> List[str]:
        """List available SPHEREx data files"""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            print(f"Warning: Could not access S3 bucket: {str(e)}")
            return []

    def get_cache_path(self, s3_key: str) -> Path:
        """Generate cache file path for an S3 key"""
        safe_name = s3_key.replace('/', '_')
        return self.cache_dir / safe_name

    def download_spectral_cube(self, s3_key: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Download a spectral cube from S3 with caching"""
        cache_path = self.get_cache_path(s3_key)

        try:
            # Check cache first
            if cache_path.exists():
                print(f"Using cached file: {cache_path.name}")
                with fits.open(cache_path, memmap=False) as hdul:
                    return self._extract_data_from_fits(hdul)

            # Download from S3
            print(f"Downloading from AWS S3: {s3_key}...")
            print("This may take a few minutes depending on file size...")

            obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)

            # Stream download
            data_chunks = []
            for chunk in obj['Body'].iter_chunks(chunk_size=CHUNK_SIZE):
                data_chunks.append(chunk)

            data = b''.join(data_chunks)
            del data_chunks

            # Save to cache
            print("Download complete! Saving to cache...")
            cache_path.write_bytes(data)

            # Load FITS data
            with io.BytesIO(data) as f:
                with fits.open(f, memmap=False) as hdul:
                    result = self._extract_data_from_fits(hdul)

            del data
            gc.collect()
            return result

        except Exception as e:
            print(f"\nError downloading data: {str(e)}")
            return None, None

    @staticmethod
    def _extract_data_from_fits(hdul) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Extract data from FITS HDU list"""
        print(f"Analyzing FITS structure ({len(hdul)} HDUs found):")
        hdul.info()

        cube_data = None
        header = None

        # Check for PSF data (3D) first
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'PSF' and hasattr(hdu, 'data') and hdu.data is not None:
                if hdu.data.ndim == 3:
                    print(f"Found PSF (Point Spread Function) data in HDU {idx}")
                    print("Using 3D PSF data as spectral cube")

                    data_shape = hdu.data.shape
                    if data_shape[1] > 512 or data_shape[2] > 512:
                        print(f"PSF data large ({data_shape}), downsampling...")
                        cube_data = hdu.data[:, ::4, ::4].astype(np.float32)
                    else:
                        cube_data = hdu.data.astype(np.float32)

                    header = dict(hdu.header)

                    # Normalize PSF data
                    for wavelength_idx in range(cube_data.shape[0]):
                        max_val = np.max(cube_data[wavelength_idx])
                        if max_val > 0:
                            cube_data[wavelength_idx] /= max_val

                    break

        # If no 3D data, use IMAGE HDU
        if cube_data is None:
            for idx, hdu in enumerate(hdul):
                if hdu.name == 'IMAGE' and hasattr(hdu, 'data') and hdu.data is not None:
                    print(f"Found 2D image data in HDU {idx}")
                    print("Converting to spectral cube format...")

                    image_data = hdu.data
                    header = dict(hdu.header)

                    # Downsample if too large
                    if image_data.shape[0] > 1024 or image_data.shape[1] > 1024:
                        print(f"Image too large ({image_data.shape}), downsampling...")
                        image_data = image_data[::4, ::4]

                    # Create synthetic spectral cube
                    n_wavelengths = 10
                    h, w = image_data.shape
                    cube_data = np.zeros((n_wavelengths, h, w), dtype=np.float32)

                    for wave_idx in range(n_wavelengths):
                        sigma = 0.5 + wave_idx * 0.5
                        filtered = gaussian_filter(image_data, sigma=sigma)
                        spectral_weight = np.sin(wave_idx * np.pi / n_wavelengths) * 0.5 + 0.5
                        cube_data[wave_idx] = filtered * spectral_weight

                    print(f"Created synthetic cube with shape: {cube_data.shape}")
                    break

        if cube_data is None:
            print("No suitable data found in FITS file")
            return None, None

        # Final size check
        if cube_data.shape[1] > DEFAULT_IMAGE_SIZE[0] or cube_data.shape[2] > DEFAULT_IMAGE_SIZE[1]:
            print(f"Final downsampling to {DEFAULT_IMAGE_SIZE}")
            zoom_factors = (
                1,
                DEFAULT_IMAGE_SIZE[0] / cube_data.shape[1],
                DEFAULT_IMAGE_SIZE[1] / cube_data.shape[2]
            )
            cube_data = zoom(cube_data, zoom_factors, order=1)

        gc.collect()
        return cube_data, header


class SpectralArtEngine:
    """Core engine for transforming spectral data into art"""

    def __init__(self):
        self.palettes = self._create_cosmic_palettes()

    @staticmethod
    def _create_cosmic_palettes() -> Dict[str, ArtisticPalette]:
        """Create artistic color palettes"""
        palettes = {}

        # Stellar Evolution Palette
        stellar_colors = [
            (0.6, 0.7, 1.0),
            (0.8, 0.85, 1.0),
            (1.0, 1.0, 0.9),
            (1.0, 0.95, 0.7),
            (1.0, 0.8, 0.5),
            (1.0, 0.6, 0.4),
            (0.8, 0.3, 0.3)
        ]

        wavelengths = np.linspace(0.75, 5.1, len(stellar_colors))
        stellar_map = {float(w): c for w, c in zip(wavelengths, stellar_colors)}

        palettes['stellar'] = ArtisticPalette('Stellar Evolution', stellar_colors, stellar_map)
        palettes['nebula'] = ArtisticPalette(
            'Emission Nebula',
            [(1.0, 0.0, 0.3), (0.0, 1.0, 0.5), (0.0, 0.5, 1.0), (0.8, 0.0, 0.8)],
            {}
        )
        palettes['ice'] = ArtisticPalette(
            'Planetary Ices',
            [(0.85, 0.95, 1.0), (0.7, 0.85, 0.95), (0.95, 0.98, 0.85)],
            {}
        )

        return palettes

    def spectrum_to_rgb(self, spectrum: np.ndarray, wavelengths: np.ndarray,
                        palette: str = 'stellar') -> Tuple[float, float, float]:
        """Convert spectrum to RGB"""
        if np.sum(spectrum) == 0 or np.max(spectrum) == 0:
            return (0.2, 0.2, 0.2)

        spectrum_norm = spectrum / np.max(spectrum)
        pal = self.palettes[palette]

        if pal.wavelength_map:
            rgb = np.zeros(3)
            wl_keys = np.array(list(pal.wavelength_map.keys()))

            total_weight = 0.0
            for wl, intensity in zip(wavelengths, spectrum_norm):
                if intensity > 0.01:
                    idx = np.argmin(np.abs(wl_keys - wl))
                    color = pal.wavelength_map[wl_keys[idx]]
                    rgb += np.array(color) * intensity
                    total_weight += intensity

            if total_weight > 0:
                return tuple(float(x) for x in np.clip(rgb / total_weight, 0, 1))
            return (0.3, 0.3, 0.3)
        else:
            idx = int(np.argmax(spectrum_norm) * len(pal.colors) / len(spectrum_norm))
            return pal.colors[min(idx, len(pal.colors) - 1)]


class GenerativeArtworks:
    """Collection of generative art algorithms"""

    def __init__(self, art_engine: SpectralArtEngine):
        self.engine = art_engine

    def cosmic_tapestry(self, cube: np.ndarray, wavelengths: np.ndarray,
                        palette: str = 'stellar') -> np.ndarray:
        """Create tapestry visualization"""
        print("Creating cosmic tapestry...")

        height, width = cube.shape[1], cube.shape[2]
        artwork = np.zeros((height, width, 3), dtype=np.float32)

        # Add base pattern
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        base_pattern = (np.sin(x * 10) * np.cos(y * 10) * 0.1 + 0.2)
        for c in range(3):
            artwork[:, :, c] = base_pattern

        # Create thread pattern
        n_threads = 50
        for thread_idx in range(n_threads):
            x_base = int(thread_idx * width / n_threads)
            wave_amplitude = 10

            for y_pos in range(height):
                x_pos = int(x_base + wave_amplitude * np.sin(y_pos * 0.05 + thread_idx * 0.3))
                x_pos = max(0, min(width - 1, x_pos))

                spectrum = cube[:, y_pos, x_pos]
                rgb = self.engine.spectrum_to_rgb(spectrum, wavelengths, palette)

                blend_factor = 0.6
                artwork[y_pos, x_pos] = artwork[y_pos, x_pos] * (1 - blend_factor) + np.array(rgb) * blend_factor

                # Add horizontal influence
                for dx in [-1, 0, 1]:
                    nx = x_pos + dx
                    if 0 <= nx < width:
                        artwork[y_pos, nx] = artwork[y_pos, nx] * 0.8 + np.array(rgb) * 0.2

        return artwork

    def spectral_aurora(self, cube: np.ndarray, wavelengths: np.ndarray,
                        time_steps: int = 5) -> List[np.ndarray]:
        """Generate aurora animation frames"""
        print("Generating spectral aurora...")

        height, width = cube.shape[1], cube.shape[2]
        frames = []

        # Normalize cube data
        cube_norm = cube.copy()
        for idx in range(cube.shape[0]):
            max_val = np.max(cube[idx])
            if max_val > 0:
                cube_norm[idx] = cube[idx] / max_val

        for t in range(min(time_steps, 3)):
            frame = np.zeros((height, width, 3), dtype=np.float32)
            phase = 2 * np.pi * t / time_steps

            for wl_idx in range(0, min(len(wavelengths), 20), 5):
                if wl_idx < cube.shape[0]:
                    slice_data = gaussian_filter(cube_norm[wl_idx], sigma=1)

                    x_wave = np.sin(np.linspace(0, 2*np.pi, width) + phase)
                    y_wave = np.cos(np.linspace(0, 2*np.pi, height) + phase * 0.5)
                    wave = np.outer(y_wave, x_wave) * 0.3 + 0.7

                    intensity = slice_data * wave

                    wl = wavelengths[wl_idx]
                    hue = (wl - 0.75) / (5.1 - 0.75)
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)

                    for c in range(3):
                        frame[:, :, c] += intensity * rgb[c] * 0.3

            frame = frame + 0.1  # Add baseline brightness
            frames.append(np.clip(frame, 0, 1))

        return frames

    def ice_crystal_growth(self, cube: np.ndarray, ice_features: np.ndarray,
                           iterations: int = 10) -> np.ndarray:
        """Simulate crystal growth"""
        print("Growing ice crystals...")

        height, width = ice_features.shape[:2]
        crystal = np.zeros((height, width, 3), dtype=np.float32)

        # Normalize ice features
        if np.max(ice_features) > 0:
            ice_features = ice_features / np.max(ice_features)

        # Create seed points
        seeds = np.random.rand(height, width) < 0.01
        growth_map = seeds.astype(np.float32) * 0.8

        # Add initial structure
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        growth_map += np.exp(-dist**2 / (width * height * 0.1)) * 0.3

        for iteration in range(min(iterations, MAX_ITERATIONS)):
            new_growth = gaussian_filter(growth_map, sigma=2.0)
            growth_factor = 1.0 + ice_features * 0.5
            growth_map = np.clip(new_growth * growth_factor, 0, 1)

            progress = iteration / max(iterations - 1, 1)
            color = plt.cm.winter(progress)[:3]

            angle = iteration * np.pi / 6
            pattern = np.sin(x * np.cos(angle) + y * np.sin(angle)) * 0.5 + 0.5

            crystal += growth_map[:, :, np.newaxis] * pattern[:, :, np.newaxis] * np.array(color) * 0.15

        crystal = crystal + 0.2  # Add base brightness
        return np.clip(crystal, 0, 1)

    def gravitational_lensing_art(self, cube: np.ndarray,
                                  mass_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Create gravitational lensing effect with multiple visual phenomena"""
        print("Creating gravitational lens art...")

        height, width = cube.shape[1], cube.shape[2]

        # Create sophisticated mass distribution
        if mass_map is None:
            # Use multiple wavelengths to create complex mass distribution
            mass_layers = []
            for i in range(0, min(10, cube.shape[0]), 2):
                layer = gaussian_filter(cube[i], sigma=3)
                mass_layers.append(layer)

            mass_map = np.mean(mass_layers, axis=0) if mass_layers else np.ones((height, width))

            # Add additional mass concentrations
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2

            # Primary mass concentration
            r1 = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mass_map += np.exp(-r1**2 / (width * height * 0.02))

            # Secondary mass clump for more complex lensing
            offset_x, offset_y = width // 4, height // 4
            r2 = np.sqrt((x - (center_x + offset_x))**2 + (y - (center_y + offset_y))**2)
            mass_map += 0.5 * np.exp(-r2**2 / (width * height * 0.03))

            # Smooth and normalize
            mass_map = gaussian_filter(mass_map, sigma=5)
            if np.max(mass_map) > 0:
                mass_map = mass_map / np.max(mass_map)

        # Create base image from spectral data
        artwork = np.zeros((height, width, 3), dtype=np.float32)

        # Use more wavelengths for richer color
        n_slices = min(12, cube.shape[0])
        for i in range(n_slices):
            wl_idx = int(i * cube.shape[0] / n_slices)
            if wl_idx < cube.shape[0]:
                slice_data = cube[wl_idx].copy()
                if np.max(slice_data) > 0:
                    slice_data = slice_data / np.max(slice_data)

                # Map to RGB channels with spectral progression
                channel = i % 3
                weight = 1.0 - (i / n_slices) * 0.5
                artwork[:, :, channel] += slice_data * weight

        # Normalize base image
        for c in range(3):
            if np.max(artwork[:, :, c]) > 0:
                artwork[:, :, c] = artwork[:, :, c] / np.max(artwork[:, :, c])

        # Apply gravitational lensing distortion
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2

        # Calculate deflection angles based on mass distribution
        # Using simplified lensing equation
        deflection_x = np.zeros((height, width))
        deflection_y = np.zeros((height, width))

        # Calculate gradients of gravitational potential
        grad_y, grad_x = np.gradient(mass_map)

        # Apply lensing equation with Einstein radius scaling
        einstein_radius = min(height, width) * 0.15
        for iy in range(height):
            for ix in range(width):
                dx = ix - center_x
                dy = iy - center_y
                r = np.sqrt(dx**2 + dy**2) + 1e-6  # Avoid division by zero

                # Deflection proportional to mass gradient and inversely to distance
                deflection_scale = einstein_radius / r
                deflection_x[iy, ix] = grad_x[iy, ix] * deflection_scale * 10
                deflection_y[iy, ix] = grad_y[iy, ix] * deflection_scale * 10

        # Apply deflection to create lensed image
        lensed_artwork = np.zeros_like(artwork)
        for iy in range(height):
            for ix in range(width):
                # Source position
                source_x = int(ix - deflection_x[iy, ix])
                source_y = int(iy - deflection_y[iy, ix])

                # Ensure within bounds
                if 0 <= source_x < width and 0 <= source_y < height:
                    lensed_artwork[iy, ix] = artwork[source_y, source_x]

        # Create multiple Einstein rings with different radii
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Primary Einstein ring
        ring_radius1 = einstein_radius
        ring_width = 3
        ring_mask1 = np.abs(r - ring_radius1) < ring_width
        ring_intensity1 = np.exp(-np.abs(r - ring_radius1)**2 / (2 * ring_width**2))

        # Secondary Einstein ring (from secondary mass)
        ring_radius2 = einstein_radius * 1.5
        ring_mask2 = np.abs(r - ring_radius2) < ring_width
        ring_intensity2 = np.exp(-np.abs(r - ring_radius2)**2 / (2 * ring_width**2))

        # Create arc segments (partial Einstein rings)
        theta = np.arctan2(y - center_y, x - center_x)
        arc_mask1 = ring_mask1 & (np.abs(theta) < np.pi/3)
        arc_mask2 = ring_mask2 & (np.abs(theta - np.pi) < np.pi/4)

        # Apply rings and arcs with color variation
        ring_color1 = np.array([0.9, 0.95, 1.0])  # Blue-white
        ring_color2 = np.array([1.0, 0.9, 0.8])   # Yellow-white
        arc_color = np.array([0.8, 0.9, 1.0])     # Cyan-white

        for c in range(3):
            lensed_artwork[:, :, c] += ring_intensity1 * ring_color1[c] * 0.5
            lensed_artwork[:, :, c] += ring_intensity2 * ring_color2[c] * 0.3
            lensed_artwork[:, :, c] = np.where(arc_mask1,
                                               lensed_artwork[:, :, c] * 0.5 + arc_color[c] * 0.5,
                                               lensed_artwork[:, :, c])
            lensed_artwork[:, :, c] = np.where(arc_mask2,
                                               lensed_artwork[:, :, c] * 0.5 + arc_color[c] * 0.5,
                                               lensed_artwork[:, :, c])

        # Add caustics with more detail
        # Calculate magnification from jacobian of lens mapping
        mag_xx = 1 - np.gradient(deflection_x, axis=1)
        mag_yy = 1 - np.gradient(deflection_y, axis=0)
        mag_xy = -np.gradient(deflection_x, axis=0)
        mag_yx = -np.gradient(deflection_y, axis=1)

        # Determinant gives magnification
        determinant = mag_xx * mag_yy - mag_xy * mag_yx
        magnification = np.abs(1.0 / (determinant + 1e-6))
        magnification = np.clip(magnification, 0, 10)

        # Caustics occur at high magnification
        caustic_mask = magnification > 3
        caustic_intensity = gaussian_filter(caustic_mask.astype(float), sigma=1)

        # Apply caustics with golden color
        caustic_color = np.array([1.0, 0.85, 0.4])
        for c in range(3):
            lensed_artwork[:, :, c] += caustic_intensity * caustic_color[c] * 0.4

        # Add gravitational microlensing sparkles
        n_sparkles = 50
        np.random.seed(42)  # For reproducibility
        for _ in range(n_sparkles):
            sx = np.random.randint(0, width)
            sy = np.random.randint(0, height)

            # Check if in high magnification region
            if magnification[sy, sx] > 2:
                # Create small bright spot
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if 0 <= sy+dy < height and 0 <= sx+dx < width:
                            dist = np.sqrt(dy**2 + dx**2)
                            if dist <= 2:
                                brightness = np.exp(-dist**2 / 2) * magnification[sy, sx] / 10
                                lensed_artwork[sy+dy, sx+dx] += brightness

        # Add subtle radial gradient for depth
        radial_gradient = 1 - (r / np.max(r)) * 0.3
        for c in range(3):
            lensed_artwork[:, :, c] *= radial_gradient

        # Final adjustments
        lensed_artwork = lensed_artwork * 0.8 + 0.1  # Ensure minimum brightness
        lensed_artwork = np.clip(lensed_artwork, 0, 1)

        # Apply subtle color grading for more dramatic effect
        # Boost blues in outer regions, warm colors in center
        for c in range(3):
            if c == 2:  # Blue channel
                lensed_artwork[:, :, c] *= (1 + (r / np.max(r)) * 0.2)
            else:  # Red and green channels
                lensed_artwork[:, :, c] *= (1 - (r / np.max(r)) * 0.1)

        return np.clip(lensed_artwork, 0, 1)

    def cosmic_memory_hallucination(self, cube: np.ndarray, wavelengths: np.ndarray,
                                    memory_depth: int = 30, flow_particles: int = 500) -> np.ndarray:
        """
        Create a 'Cosmic Memory' visualization inspired by Refik Anadol's machine hallucinations,
        Ryoji Ikeda's data aesthetics, and Casey Reas' generative systems.

        Treats spectral data as living memory - particles flow through spectral fields,
        leaving traces that accumulate into a breathing, evolving visualization.
        """
        print("Generating Cosmic Memory Hallucination...")

        height, width = cube.shape[1], cube.shape[2]

        # Initialize memory canvas with subtle noise
        memory_canvas = np.zeros((height, width, 3), dtype=np.float32)
        memory_buffer = np.zeros((memory_depth, height, width, 3), dtype=np.float32)

        # Create flow field from spectral gradients
        # Inspired by Reas' organic movement patterns
        flow_field_x = np.zeros((height, width), dtype=np.float32)
        flow_field_y = np.zeros((height, width), dtype=np.float32)

        # Use multiple wavelengths to create complex flow patterns
        for wl_idx in range(0, min(cube.shape[0], 10), 2):
            if wl_idx < cube.shape[0]:
                slice_data = cube[wl_idx]
                if np.max(slice_data) > 0:
                    slice_norm = slice_data / np.max(slice_data)
                else:
                    slice_norm = slice_data

                # Calculate gradients for flow field
                gy, gx = np.gradient(gaussian_filter(slice_norm, sigma=3))

                # Add rotational component for more organic flow
                angle = wl_idx * np.pi / 5
                flow_field_x += gx * np.cos(angle) - gy * np.sin(angle)
                flow_field_y += gx * np.sin(angle) + gy * np.cos(angle)

                del gy, gx

        # Normalize flow field
        flow_magnitude = np.sqrt(flow_field_x**2 + flow_field_y**2)
        max_flow = np.max(flow_magnitude)
        if max_flow > 0:
            flow_field_x /= max_flow
            flow_field_y /= max_flow

        # Initialize particles with spectral memory
        particles = {
            'x': np.random.rand(flow_particles) * width,
            'y': np.random.rand(flow_particles) * height,
            'age': np.zeros(flow_particles),
            'wavelength_idx': np.random.randint(0, min(cube.shape[0], 20), flow_particles),
            'intensity': np.ones(flow_particles),
            'memory': np.zeros((flow_particles, 3))  # RGB memory
        }

        # Create Ikeda-inspired data pulse patterns
        pulse_grid = np.zeros((height, width), dtype=np.float32)
        grid_size = 32
        for gy in range(0, height, grid_size):
            for gx in range(0, width, grid_size):
                if gy < height and gx < width:
                    # Sample spectral data at grid points
                    y_end = min(height, gy+grid_size)
                    x_end = min(width, gx+grid_size)
                    grid_region = cube[:, gy:y_end, gx:x_end]
                    if grid_region.size > 0:
                        avg_spectrum = np.mean(grid_region, axis=(1, 2))
                        pulse_intensity = np.max(avg_spectrum) if len(avg_spectrum) > 0 else 0
                        pulse_grid[gy:y_end, gx:x_end] = pulse_intensity
                        del avg_spectrum
                    del grid_region

        # Memory accumulation loop - inspired by Anadol's layered time
        for iteration in range(memory_depth):
            frame = np.zeros((height, width, 3), dtype=np.float32)

            # Update and render particles
            for i in range(flow_particles):
                # Get particle position
                px, py = int(particles['x'][i]), int(particles['y'][i])

                if 0 <= px < width and 0 <= py < height:
                    # Sample flow field
                    fx = flow_field_x[py, px]
                    fy = flow_field_y[py, px]

                    # Add spectral influence
                    wl_idx = particles['wavelength_idx'][i]
                    if wl_idx < cube.shape[0]:
                        spectral_influence = cube[wl_idx, py, px]

                        # Update particle position with organic movement
                        noise_scale = 0.1
                        particles['x'][i] += fx * 2 + np.random.randn() * noise_scale
                        particles['y'][i] += fy * 2 + np.random.randn() * noise_scale

                        # Wrap around edges
                        particles['x'][i] = particles['x'][i] % width
                        particles['y'][i] = particles['y'][i] % height

                        # Calculate color from wavelength and spectral intensity
                        wl = wavelengths[wl_idx] if wl_idx < len(wavelengths) else 2.5

                        # Create color inspired by spectral data
                        hue = (wl - 0.75) / (5.1 - 0.75)

                        # Modulate saturation based on spectral intensity
                        saturation = 0.3 + spectral_influence * 0.7
                        value = 0.5 + spectral_influence * 0.5

                        # Convert to RGB
                        rgb = colorsys.hsv_to_rgb(hue, saturation, value)

                        # Update particle memory (accumulate color history)
                        particles['memory'][i] = particles['memory'][i] * 0.95 + np.array(rgb) * 0.05

                        # Render particle with gaussian splat
                        render_size = 3
                        y_start, y_end = max(0, py-render_size), min(height, py+render_size+1)
                        x_start, x_end = max(0, px-render_size), min(width, px+render_size+1)

                        for dy in range(y_start, y_end):
                            for dx in range(x_start, x_end):
                                dist = np.sqrt((dy-py)**2 + (dx-px)**2)
                                if dist <= render_size:
                                    weight = np.exp(-dist**2 / 2)
                                    frame[dy, dx] += particles['memory'][i] * weight * 0.02

                    # Age particles
                    particles['age'][i] += 1

                    # Respawn old particles
                    if particles['age'][i] > 200:
                        particles['x'][i] = np.random.rand() * width
                        particles['y'][i] = np.random.rand() * height
                        particles['age'][i] = 0
                        particles['wavelength_idx'][i] = np.random.randint(0, min(cube.shape[0], 20))

            # Add Ikeda-inspired data grid overlay
            time_phase = iteration / memory_depth
            grid_alpha = 0.05 + 0.05 * np.sin(time_phase * np.pi * 4)

            # Create grid pattern
            grid_pattern = np.zeros((height, width), dtype=np.float32)
            grid_lines = 16
            for i in range(0, width, width // grid_lines):
                grid_pattern[:, max(0, min(i, width-1))] = 1
            for i in range(0, height, height // grid_lines):
                grid_pattern[max(0, min(i, height-1)), :] = 1

            # Modulate grid by spectral data
            grid_modulated = grid_pattern * pulse_grid * grid_alpha

            # Add grid to frame
            for c in range(3):
                frame[:, :, c] += grid_modulated

            # Store in memory buffer
            memory_buffer[iteration % memory_depth] = frame

            # Accumulate with temporal decay - Anadol's time-based layering
            decay_factor = np.exp(-iteration * 0.02)
            memory_canvas += frame * decay_factor

        # Final composition combining all memory layers
        # Average recent memory frames
        recent_memory = np.mean(memory_buffer[-10:], axis=0)

        # Clear memory buffer after use
        del memory_buffer
        gc.collect()

        # Create final hallucination effect
        # Combine accumulated memory with recent activity
        final_image = memory_canvas * 0.7 + recent_memory * 0.3

        # Clear intermediate arrays
        del memory_canvas
        del recent_memory
        gc.collect()

        # Add spectral resonance highlights
        # Find peaks in spectral data across all wavelengths
        spectral_max = np.max(cube, axis=0)
        if np.max(spectral_max) > 0:
            spectral_max = spectral_max / np.max(spectral_max)

        # Create resonance effect
        resonance = gaussian_filter(spectral_max, sigma=5)
        resonance_color = np.array([0.9, 0.95, 1.0])  # Slight blue-white

        for c in range(3):
            final_image[:, :, c] += resonance * resonance_color[c] * 0.2

        # Clean up
        del spectral_max
        del resonance
        del flow_field_x
        del flow_field_y
        del pulse_grid
        del particles
        gc.collect()

        # Add subtle noise texture inspired by Ikeda's aesthetic
        noise = np.random.randn(height, width) * 0.02
        for c in range(3):
            final_image[:, :, c] += noise

        del noise

        # Final normalization and contrast adjustment
        # Enhance contrast while preserving delicate details
        final_image = np.clip(final_image, 0, 1)

        # Apply subtle contrast curve
        final_image = np.power(final_image, 0.9)  # Slight brightening

        # Add minimal baseline to ensure visibility
        final_image = final_image * 0.9 + 0.05

        return np.clip(final_image, 0, 1)


class SPHERExSonification:
    """Convert spectral data to sound"""

    def __init__(self, sample_rate: int = 11025):
        self.sample_rate = sample_rate

    def spectrum_to_audio(self, spectrum: np.ndarray, wavelengths: np.ndarray,
                          duration: float = 1.0) -> np.ndarray:
        """Convert spectrum to audio"""
        n_samples = int(duration * self.sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)

        significant = spectrum > np.max(spectrum) * 0.1
        indices = np.where(significant)[0]

        if len(indices) > 0:
            indices = indices[::max(1, len(indices)//10)]

            t = np.linspace(0, duration, n_samples)
            for idx in indices:
                wl = wavelengths[idx]
                amplitude = spectrum[idx]
                freq = 100 + 1900 * (wl - 0.75) / (5.1 - 0.75)

                wave = amplitude * np.sin(2 * np.pi * freq * t) * np.exp(-t * 2)
                audio += wave

            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

        return audio


class ArtworkRenderer:
    """Render and save artworks"""

    def __init__(self, dpi: int = 100):
        self.dpi = dpi
        self.figures = weakref.WeakSet()

    def __del__(self):
        """Cleanup figures on deletion"""
        self.cleanup_figures()

    def cleanup_figures(self):
        """Close all tracked figures"""
        for fig in list(self.figures):
            try:
                plt.close(fig)
            except Exception:
                pass
        self.figures.clear()

    def render_cosmic_tapestry(self, artwork: np.ndarray, title: str = "Cosmic Tapestry",
                               save_path: Optional[str] = None):
        """Render artwork"""
        print(f"Rendering: {title}")

        fig = plt.figure(figsize=(8, 6), dpi=self.dpi)
        self.figures.add(fig)

        try:
            ax = fig.add_subplot(111)

            # Ensure visibility
            display_art = artwork.copy()
            if np.max(display_art) < 0.1:
                display_art = display_art * 5 + 0.1
            display_art = np.clip(display_art, 0, 1)

            ax.imshow(display_art, interpolation='bilinear', aspect='auto')
            ax.axis('off')
            ax.set_title(title, color='white', fontsize=14, pad=10)

            fig.patch.set_facecolor('black')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                            facecolor='black', edgecolor='none')
                print(f"Saved to {save_path}")

            plt.draw()
            plt.pause(0.001)
            plt.show(block=True)

        finally:
            plt.close(fig)
            self.figures.discard(fig)
            gc.collect()


def create_spherex_artwork():
    """Main function to create SPHEREx generative art"""
    print("=== Starting SPHEREx Artwork Generation ===")
    print(f"Cache directory: ./spherex_cache")

    # Track resources
    data_access = None
    renderer = None

    try:
        # Initialize components
        data_access = SPHERExDataAccess()
        art_engine = SpectralArtEngine()
        artworks = GenerativeArtworks(art_engine)
        renderer = ArtworkRenderer()
        sonifier = SPHERExSonification()

        # Initialize variables
        cube_data = None
        wavelengths = None

        # List available data
        print("Checking available SPHEREx data...")
        available_files = data_access.list_available_data()

        if available_files and len(available_files) > 0:
            print(f"Found {len(available_files)} data files")

            # Check if first file is cached
            sample_file = available_files[0]
            cache_path = data_access.get_cache_path(sample_file)

            if cache_path.exists():
                print(f"Selected file: {sample_file}")
                print("This file is already cached locally.")
                print("Press Enter to use cached data, 'download' to re-download, or 'skip' for synthetic data: ", end='')
            else:
                print(f"Selected file: {sample_file}")
                print("This file needs to be downloaded from AWS (may take a few minutes).")
                print("Press Enter to download, or 'skip' to use synthetic data instead: ", end='')

            user_input = input().strip().lower()

            if user_input == 'skip':
                print("\nUsing synthetic data as requested...")
                available_files = []
            elif user_input == 'download' and cache_path.exists():
                print("\nRemoving cached file and re-downloading...")
                cache_path.unlink()
                cube_data, header = data_access.download_spectral_cube(sample_file)
            else:
                cube_data, header = data_access.download_spectral_cube(sample_file)

            if cube_data is not None:
                n_wavelengths, height, width = cube_data.shape
                wavelengths = np.linspace(0.75, 5.1, n_wavelengths)
                print(f"Successfully loaded cube: {n_wavelengths} wavelengths × {height}×{width} pixels")
                print(f"Data range: [{np.min(cube_data):.3f}, {np.max(cube_data):.3f}]")
            else:
                print("Failed to load data from file")
                available_files = []
        else:
            print("No SPHEREx data files found on AWS")
            print("This might be due to network issues or changes in the data repository")

        # Generate synthetic data if needed
        if not available_files or cube_data is None:
            if cube_data is None and available_files:
                print("\nFailed to load data, falling back to synthetic data...")
            else:
                print("\nUsing synthetic data for demonstration...")

            # Create synthetic spectral cube
            n_wavelengths = 20
            height, width = DEFAULT_IMAGE_SIZE

            print("Generating synthetic spectral cube...")
            wavelengths = np.linspace(0.75, 5.1, n_wavelengths)
            cube_data = np.zeros((n_wavelengths, height, width), dtype=np.float32)

            # Generate patterns
            x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))

            for i in range(n_wavelengths):
                theta = np.arctan2(y, x)
                r = np.sqrt(x**2 + y**2)

                spiral = np.sin(5 * theta - 10 * r + i * 0.2) * 0.3
                blob1 = np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / 0.2)
                blob2 = np.exp(-((x + 0.3)**2 + (y + 0.3)**2) / 0.2)

                weight = np.sin(i * np.pi / n_wavelengths)
                cube_data[i] = spiral + (blob1 + blob2) * weight * 0.5 + 0.3
                cube_data[i] += np.random.randn(height, width) * 0.05

            cube_data = np.clip(cube_data, 0, None)

            # Normalize synthetic data
            for i in range(n_wavelengths):
                if np.max(cube_data[i]) > 0:
                    cube_data[i] = cube_data[i] / np.max(cube_data[i])

            print("Synthetic data generated successfully")

        # Ensure we have data
        if cube_data is None or wavelengths is None:
            print("\nERROR: No data available for visualization")
            print("Please check your internet connection or try again later")
            return

        # Generate artworks
        print("\n=== Creating Artworks ===")

        # Extract data for sonification before processing (small memory footprint)
        if cube_data is not None:
            center_y, center_x = cube_data.shape[1]//2, cube_data.shape[2]//2
            center_spectrum_for_audio = cube_data[:, center_y, center_x].copy()
            wavelengths_for_audio = wavelengths.copy()
        else:
            center_spectrum_for_audio = None
            wavelengths_for_audio = None

        # 1. Cosmic Tapestry
        print("\n1. Creating Cosmic Tapestry...")
        tapestry = artworks.cosmic_tapestry(cube_data, wavelengths, 'stellar')
        renderer.render_cosmic_tapestry(tapestry, "Stellar Tapestry", "spherex_stellar_tapestry.png")
        del tapestry
        gc.collect()

        # 2. Spectral Aurora (save only one frame)
        print("\n2. Generating Spectral Aurora...")
        aurora_frames = artworks.spectral_aurora(cube_data, wavelengths, 3)
        if aurora_frames and len(aurora_frames) > 0:
            # Save the middle frame
            renderer.render_cosmic_tapestry(aurora_frames[1], "Spectral Aurora", "spherex_spectral_aurora.png")
            for frame in aurora_frames:
                del frame
            del aurora_frames
            gc.collect()

        # 3. Ice Crystals
        print("\n3. Growing Ice Crystals...")
        current_height, current_width = cube_data.shape[1], cube_data.shape[2]

        if cube_data.shape[0] > 10:
            ice_slice_start = cube_data.shape[0] // 3
            ice_slice_end = 2 * cube_data.shape[0] // 3
            ice_features = np.mean(cube_data[ice_slice_start:ice_slice_end], axis=0)
        else:
            ice_features = np.mean(cube_data, axis=0)

        ice_features = gaussian_filter(ice_features, sigma=5)

        if np.max(ice_features) > 0:
            ice_features = ice_features / np.max(ice_features)
        else:
            y, x = np.ogrid[:current_height, :current_width]
            ice_features = np.sin(x * 0.1) * np.cos(y * 0.1) * 0.5 + 0.5

        crystals = artworks.ice_crystal_growth(cube_data, ice_features)
        renderer.render_cosmic_tapestry(crystals, "Ice Crystals", "spherex_ice_crystals.png")
        del ice_features, crystals
        gc.collect()

        # 4. Gravitational Lens
        print("\n4. Creating Gravitational Lens...")
        lensed = artworks.gravitational_lensing_art(cube_data)
        renderer.render_cosmic_tapestry(lensed, "Gravitational Lens", "spherex_gravitational_lens.png")
        del lensed
        gc.collect()

        # 5. Cosmic Memory Hallucination
        print("\n5. Creating Cosmic Memory Hallucination...")
        cosmic_memory = artworks.cosmic_memory_hallucination(cube_data, wavelengths)
        renderer.render_cosmic_tapestry(cosmic_memory, "Cosmic Memory - Data as Living Pigment",
                                       "spherex_cosmic_memory.png")
        del cosmic_memory
        gc.collect()

        # Free the large cube_data array now that visualizations are complete
        del cube_data
        gc.collect()

        # Create sonification using the saved spectrum
        print("\nCreating cosmic sonification...")
        if center_spectrum_for_audio is not None:
            audio = sonifier.spectrum_to_audio(center_spectrum_for_audio, wavelengths_for_audio, 3.0)
            print(f"Generated {len(audio)} audio samples")
            print(f"(Audio: {sonifier.sample_rate} Hz sample rate × 3 seconds = {len(audio)} samples)")
            print("Note: Audio is generated but not saved to file in this version")
            del audio

        if center_spectrum_for_audio is not None:
            del center_spectrum_for_audio
        if wavelengths_for_audio is not None:
            del wavelengths_for_audio
        gc.collect()

        # Summary of saved files
        print("\n=== SPHEREx Generative Art Complete! ===")
        print("\nSaved visualizations:")
        print("- spherex_stellar_tapestry.png")
        print("- spherex_spectral_aurora.png")
        print("- spherex_ice_crystals.png")
        print("- spherex_gravitational_lens.png")
        print("- spherex_cosmic_memory.png")

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        if os.environ.get('DEBUG'):
            import traceback
            traceback.print_exc()

    finally:
        print("\nCleaning up resources...")

        if renderer is not None:
            renderer.cleanup_figures()
            del renderer

        if data_access is not None:
            data_access.cleanup_resources()
            del data_access

        plt.close('all')
        gc.collect()

        print("Resources cleaned up")


if __name__ == "__main__":
    create_spherex_artwork()
    print("\nPress Enter to exit...")
    input()

    plt.close('all')
    gc.collect()