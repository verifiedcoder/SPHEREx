# Claude Technical Prompt - SPHEREx Framework Developer Context

## Codebase Overview
Python framework transforming SPHEREx spectral cube data (FITS format) into artistic visualizations. Clean architecture with 5 main classes, memory-efficient processing, AWS S3 integration.

## Key Technical Details

### Data Flow
```
S3 FITS → SPHERExDataAccess → numpy cube[wavelength,y,x] → GenerativeArtworks → RGB array → PNG
                                          ↓
                                   SPHERExSonification → audio array
```

### Critical Functions

```python
# Data extraction (lines 165-199)
def _extract_data_from_fits(hdul) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    # Looks for PSF (3D) or IMAGE (2D) HDUs
    # Normalizes and downsamples if needed
    # Returns cube[wavelengths, height, width]

# Memory management pattern (lines 1000-1050)
# Extract audio data BEFORE visualizations
center_spectrum_for_audio = cube_data[:, center_y, center_x].copy()
# Process all visualizations
# Delete cube_data IMMEDIATELY after
del cube_data
gc.collect()

# Cosmic Memory algorithm (lines 550-720)
def cosmic_memory_hallucination():
    # Flow fields from spectral gradients
    # Particle system with memory
    # Temporal accumulation with decay
    # Aggressive memory cleanup throughout
```

### Current Architecture Constraints

1. **Memory**: ~500MB peak for 121×101×101 cube
2. **No GPU**: Pure NumPy/SciPy operations
3. **Single-threaded**: Sequential processing
4. **Display**: Matplotlib with TkAgg backend
5. **Storage**: Local cache, no database

### Performance Bottlenecks

1. **Gravitational Lens** (lines 500-548): Pixel-by-pixel deflection calculation
2. **Cosmic Memory** (lines 550-720): Particle rendering loops
3. **File I/O**: FITS parsing and S3 downloads
4. **Memory copies**: Some unnecessary .copy() operations remain

## Ready-to-Implement Enhancements

### 1. Save Audio (Easiest)
```python
# Add to SPHERExSonification class
def save_audio(self, audio: np.ndarray, filename: str):
    import wave
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(self.sample_rate)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())
```

### 2. Progress Bars
```python
from tqdm import tqdm
# Wrap loops: for i in tqdm(range(n), desc="Processing"):
```

### 3. GPU Acceleration Points
- Gravitational deflection calculation (vectorize with CuPy)
- Particle rendering in cosmic memory
- Gaussian filtering operations
- Color space conversions

### 4. New Visualization Slot
```python
def quantum_interference_pattern(self, cube, wavelengths):
    # Your new algorithm here
    # Follow pattern: print status, process data, cleanup memory
    pass
```

## Common Modification Patterns

### Add Parameter to Existing Viz
```python
def cosmic_tapestry(self, cube, wavelengths, palette='stellar', 
                   thread_density=50):  # New parameter
    n_threads = thread_density  # Use it
```

### Add New Color Palette
```python
palettes['quantum'] = ArtisticPalette(
    'Quantum', 
    [(0.1, 0.0, 0.3), (0.5, 0.0, 0.8), (0.9, 0.3, 1.0)],
    {}
)
```

### Optimize Memory Usage
```python
# Replace: slice_data = cube[i].copy()
# With: slice_data = cube[i]  # If not modifying
```

## Testing Snippets

### Test with Synthetic Data
```bash
# When prompted, type 'skip' to use synthetic data
python main.py
```

### Debug Mode
```bash
export DEBUG=1
python main.py
```

### Memory Profiling
```python
import tracemalloc
tracemalloc.start()
# ... code ...
print(tracemalloc.get_traced_memory())
```

## File Structure
```
main.py                 # 1198 lines total
├── Imports            # Lines 1-35
├── Config             # Lines 36-55
├── Classes            # Lines 56-825
│   ├── SPHERExDataAccess      # 75-199
│   ├── SpectralArtEngine      # 202-247  
│   ├── GenerativeArtworks     # 250-720
│   ├── SPHERExSonification    # 723-773
│   └── ArtworkRenderer        # 776-825
├── Main Function      # Lines 828-1194
└── Entry Point        # Lines 1196-1198
```

## Branch Points for Major Features

1. **Web Interface**: Add Flask routes calling existing methods
2. **Batch Mode**: Wrap main() in file loop
3. **Video Export**: Collect frames, use OpenCV/ffmpeg
4. **3D Mode**: Replace matplotlib with vispy/moderngl
5. **Real-time**: Add threading, queue system

---
*For developers: This framework is production-ready but designed for extensibility. Memory management is critical due to large spectral cubes.*