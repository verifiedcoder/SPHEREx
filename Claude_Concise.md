# Claude Quick Start - SPHEREx Generative Art

## What This Is
A complete Python framework that creates artistic visualizations from NASA SPHEREx telescope spectral data. Currently functional with 5 unique visualizations.

## Current State
- **Status**: ✅ Fully working, documented, memory-efficient
- **Latest Fix**: Enhanced gravitational lens visualization with realistic physics
- **Output**: 5 PNG files + audio sonification (not saved)
- **Data Source**: AWS S3 public bucket `nasa-irsa-spherex`

## The 5 Visualizations
1. **Cosmic Tapestry** - Woven spectral threads
2. **Spectral Aurora** - Flowing wavelength animations  
3. **Ice Crystal Growth** - Simulated crystal formation
4. **Gravitational Lens** - Einstein rings and distortions
5. **Cosmic Memory** - Particle flows (Anadol/Ikeda/Reas inspired)

## Quick Context for Changes
```python
# Main components:
SPHERExDataAccess     # Handles S3 downloads
SpectralArtEngine     # Colors from spectra
GenerativeArtworks    # 5 visualization methods
ArtworkRenderer       # Display and save
SPHERExSonification   # Audio generation

# Key parameters:
DEFAULT_IMAGE_SIZE = (256, 256)
memory_depth = 30      # For cosmic memory
flow_particles = 500   # For cosmic memory
```

## Most Likely Next Steps
1. Save audio output as WAV file
2. Add new visualization algorithm
3. Create web interface
4. Process multiple FITS files
5. Higher resolution output
6. GPU acceleration
7. Video/animation export

## Ask Me To:
- "Add a 6th visualization based on [concept]"
- "Save the audio sonification as a WAV file"  
- "Create 4K resolution outputs"
- "Add progress bars during processing"
- "Build a FastAPI web interface"
- "Explain how [specific visualization] works"
- "Optimize memory usage further"

## Project Files
- `main.py` - The complete framework (~1200 lines)
- `README.md` - Quick start
- `API_DOCUMENTATION.md` - Technical reference

---
*Paste this into a new Claude conversation to continue development of the SPHEREx Generative Art Framework*