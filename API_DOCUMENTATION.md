# SPHEREx Generative Art Framework - API Documentation

## Classes Overview

```python
SPHERExDataAccess      # S3 data retrieval and caching
SpectralArtEngine      # Spectral-to-color conversion
GenerativeArtworks     # Visualization algorithms
ArtworkRenderer        # Display and file I/O
SPHERExSonification    # Audio generation
```

## SPHERExDataAccess

Manages data retrieval from AWS S3 and local caching.

### Methods

#### `__init__(cache_dir: str = "./spherex_cache")`
Initialize data access with local cache directory.

#### `list_available_data(prefix: str = 'qr/level2/', max_keys: int = 100) -> List[str]`
List available SPHEREx files in S3 bucket.

**Parameters:**
- `prefix`: S3 prefix to search
- `max_keys`: Maximum number of results

**Returns:** List of S3 object keys

#### `download_spectral_cube(s3_key: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]`
Download and extract spectral cube from FITS file.

**Parameters:**
- `s3_key`: S3 object key

**Returns:** Tuple of (spectral_cube, header_dict) or (None, None) on error

#### `cleanup_resources()`
Clean up any open file handles and resources.

### Private Methods

#### `_extract_data_from_fits(hdul) -> Tuple[Optional[np.ndarray], Optional[Dict]]`
Extract 3D spectral cube from FITS HDU list.

## SpectralArtEngine

Converts spectral data to RGB colors using various palettes.

### Methods

#### `__init__()`
Initialize with cosmic color palettes.

#### `spectrum_to_rgb(spectrum: np.ndarray, wavelengths: np.ndarray, palette: str = 'stellar') -> Tuple[float, float, float]`
Convert a spectrum to RGB color.

**Parameters:**
- `spectrum`: 1D array of spectral intensities
- `wavelengths`: 1D array of wavelengths in micrometers
- `palette`: Color palette name ('stellar', 'nebula', 'ice')

**Returns:** RGB tuple with values 0-1

### Available Palettes

- **stellar**: Temperature-based stellar evolution colors
- **nebula**: Emission line inspired colors
- **ice**: Cool blues and whites for icy regions

## GenerativeArtworks

Core visualization algorithms.

### Methods

#### `__init__(art_engine: SpectralArtEngine)`
Initialize with a spectral art engine.

#### `cosmic_tapestry(cube: np.ndarray, wavelengths: np.ndarray, palette: str = 'stellar') -> np.ndarray`
Create woven tapestry visualization.

**Parameters:**
- `cube`: 3D spectral cube [wavelength, y, x]
- `wavelengths`: 1D array of wavelengths
- `palette`: Color palette name

**Returns:** RGB image array [height, width, 3]

#### `spectral_aurora(cube: np.ndarray, wavelengths: np.ndarray, time_steps: int = 5) -> List[np.ndarray]`
Generate aurora animation frames.

**Parameters:**
- `cube`: 3D spectral cube
- `wavelengths`: 1D array of wavelengths
- `time_steps`: Number of animation frames

**Returns:** List of RGB image arrays

#### `ice_crystal_growth(cube: np.ndarray, ice_features: np.ndarray, iterations: int = 10) -> np.ndarray`
Simulate ice crystal formation.

**Parameters:**
- `cube`: 3D spectral cube
- `ice_features`: 2D array of ice-related features
- `iterations`: Growth simulation steps

**Returns:** RGB image array

#### `gravitational_lensing_art(cube: np.ndarray, mass_map: Optional[np.ndarray] = None) -> np.ndarray`
Create gravitational lensing effects.

**Parameters:**
- `cube`: 3D spectral cube
- `mass_map`: Optional 2D mass distribution

**Returns:** RGB image array with lensing effects

#### `cosmic_memory_hallucination(cube: np.ndarray, wavelengths: np.ndarray, memory_depth: int = 30, flow_particles: int = 500) -> np.ndarray`
Data-as-memory visualization with particle flows.

**Parameters:**
- `cube`: 3D spectral cube
- `wavelengths`: 1D array of wavelengths
- `memory_depth`: Temporal accumulation layers
- `flow_particles`: Number of flow particles

**Returns:** RGB image array

## ArtworkRenderer

Handles visualization display and file saving.

### Methods

#### `__init__(dpi: int = 100)`
Initialize renderer with DPI setting.

#### `render_cosmic_tapestry(artwork: np.ndarray, title: str = "Cosmic Tapestry", save_path: Optional[str] = None)`
Display and optionally save artwork.

**Parameters:**
- `artwork`: RGB image array
- `title`: Display title
- `save_path`: Optional file path for saving

#### `cleanup_figures()`
Close all matplotlib figures and free memory.

## SPHERExSonification

Converts spectral data to audio.

### Methods

#### `__init__(sample_rate: int = 11025)`
Initialize with audio sample rate.

#### `spectrum_to_audio(spectrum: np.ndarray, wavelengths: np.ndarray, duration: float = 1.0) -> np.ndarray`
Convert spectrum to audio signal.

**Parameters:**
- `spectrum`: 1D spectral intensities
- `wavelengths`: 1D wavelengths in micrometers
- `duration`: Audio duration in seconds

**Returns:** 1D audio signal array

## Data Structures

### SpectralBand Enum
```python
class SpectralBand(Enum):
    BAND1 = (0.75, 1.1, 39)   # wavelength_min, wavelength_max, resolution
    BAND2 = (1.1, 1.6, 41)
    BAND3 = (1.6, 2.4, 41)
    BAND4 = (2.4, 3.8, 35)
    BAND5 = (3.8, 4.4, 112)
    BAND6 = (4.4, 5.1, 128)
```

### ArtisticPalette Dataclass
```python
@dataclass
class ArtisticPalette:
    name: str
    colors: List[Tuple[float, float, float]]
    wavelength_map: Dict[float, Tuple[float, float, float]]
```

## Usage Examples

### Basic Visualization Pipeline

```python
# Initialize components
data_access = SPHERExDataAccess()
art_engine = SpectralArtEngine()
artworks = GenerativeArtworks(art_engine)
renderer = ArtworkRenderer()

# Get data
files = data_access.list_available_data()
cube, header = data_access.download_spectral_cube(files[0])

# Create visualization
wavelengths = np.linspace(0.75, 5.1, cube.shape[0])
artwork = artworks.cosmic_tapestry(cube, wavelengths)

# Display and save
renderer.render_cosmic_tapestry(artwork, "My Tapestry", "output.png")
```

### Custom Color Palette

```python
# Create custom palette
from dataclasses import dataclass

custom_colors = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
]

custom_palette = ArtisticPalette(
    name='custom',
    colors=custom_colors,
    wavelength_map={}
)

# Add to engine
art_engine.palettes['custom'] = custom_palette

# Use in visualization
artwork = artworks.cosmic_tapestry(cube, wavelengths, 'custom')
```

### Memory-Efficient Processing

```python
# Process large cube in chunks
chunk_size = 10  # wavelengths per chunk
height, width = cube.shape[1], cube.shape[2]
result = np.zeros((height, width, 3))

for i in range(0, cube.shape[0], chunk_size):
    chunk = cube[i:i+chunk_size]
    partial = process_chunk(chunk)  # Your processing
    result += partial
    del chunk
    gc.collect()
```

## Error Handling

The framework uses defensive programming with fallbacks:

```python
try:
    cube, header = data_access.download_spectral_cube(s3_key)
except Exception as e:
    print(f"Download failed: {e}")
    # Generate synthetic data as fallback
    cube = generate_synthetic_cube()
```

## Performance Considerations

1. **Memory Usage**: Spectral cubes can be large (100+ MB)
   - Use `del` and `gc.collect()` after processing
   - Extract only needed data early

2. **Processing Time**: Visualizations are CPU-intensive
   - Cosmic Memory: ~10-30 seconds
   - Others: ~2-5 seconds each

3. **Disk Space**: Cache can grow large
   - Monitor `./spherex_cache/` size
   - Clear old files periodically

## Extension Points

### Adding New Visualizations

```python
def my_visualization(self, cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """My custom visualization"""
    height, width = cube.shape[1], cube.shape[2]
    result = np.zeros((height, width, 3))
    
    # Your algorithm here
    
    return np.clip(result, 0, 1)

# Add to GenerativeArtworks class
GenerativeArtworks.my_visualization = my_visualization
```

### Custom Data Sources

```python
class CustomDataAccess(SPHERExDataAccess):
    def download_spectral_cube(self, identifier: str):
        # Your custom data loading
        data = load_my_data(identifier)
        return process_to_cube(data), {}
```

## Best Practices

1. **Always clip final images**: `np.clip(image, 0, 1)`
2. **Normalize spectral data**: Account for varying intensities
3. **Handle edge cases**: Check for zero/NaN values
4. **Document parameters**: Explain artistic choices
5. **Test with synthetic data**: Ensure robustness

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `DEFAULT_IMAGE_SIZE`
   - Process smaller wavelength ranges
   - Increase garbage collection

2. **Slow Performance**
   - Reduce `flow_particles` in cosmic memory
   - Lower iteration counts
   - Use smaller image sizes

3. **No Data Available**
   - Check internet connection
   - Verify S3 bucket accessibility
   - Use synthetic data mode

### Debug Mode

Enable detailed logging:
```bash
export DEBUG=1
python main.py
```