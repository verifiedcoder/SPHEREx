# SPHEREx Generative Art Framework

## Project Context

This repository is a Python framework that transforms NASA's SPHEREx telescope data into artistic visualizations. The project is complete and functional, creating distinct visualizations from spectral cube data.

### Technical Stack
- Python 3.x with numpy, scipy, matplotlib, astropy, boto3
- AWS S3 access to public SPHEREx data (bucket: `nasa-irsa-spherex`)
- FITS file processing for astronomical data
- Memory-efficient processing of large spectral cubes

### Current Features
1. **Cosmic Tapestry** - Weaving patterns from spectral threads
2. **Spectral Aurora** - Animated aurora-like flows
3. **Ice Crystal Growth** - Simulated crystal formation
4. **Gravitational Lens** - Einstein rings and light distortion (recently enhanced)
5. **Cosmic Memory Hallucination** - Particle flow inspired by Refik Anadol, Ryoji Ikeda, Casey Reas

### Key Implementation Details
- Main script: `main.py` (complete, ~1200 lines)
- Automatic caching in `./spherex_cache/`
- Outputs 5 PNG files (256x256 default)
- Audio sonification (11025 Hz, 3 seconds)
- Memory management with aggressive garbage collection
- No triptych displays (redundant)

## Working State

The framework is fully functional with:
- ✅ All 5 visualizations implemented and tested
- ✅ Memory leaks fixed through early data extraction
- ✅ Enhanced gravitational lens visualization with realistic physics
- ✅ Complete documentation (README.md, API docs)
- ✅ Clean code with type hints and error handling

You must keep all markdown documents up to date at all times.

## Known Considerations

1. **Memory Usage**: Large spectral cubes require careful management
2. **Download Time**: First run downloads ~50-100MB FITS files
3. **Processing Time**: Each visualization takes 2-30 seconds
4. **Display**: Uses matplotlib with TkAgg backend for compatibility

## Potential Next Steps

### Immediate Enhancements
1. **Save Audio**: Currently generates but doesn't save WAV files
2. **Progress Indicators**: Add progress bars for long operations  
3. **Configuration File**: Move hardcoded parameters to config.yaml
4. **Batch Processing**: Process multiple FITS files automatically

### Advanced Features
1. **Real-time Visualization**: Stream data as it becomes available
2. **Interactive Mode**: Parameter adjustment during rendering
3. **Video Export**: Animate transitions between wavelengths
4. **3D Visualizations**: Use vispy or OpenGL for 3D renders
5. **Web Interface**: Flask/FastAPI for browser-based access

### Artistic Extensions
1. **Music Composition**: Full musical pieces from spectral data
2. **VR Experience**: Immersive data exploration
3. **Style Transfer**: Apply artistic styles to visualizations
4. **Collaborative Mode**: Multiple data sources combined
5. **Print Quality**: High-resolution outputs for physical media

## Code Quality Improvements

1. **Testing**: Add unit tests for each component
2. **Logging**: Replace print statements with proper logging
3. **CLI**: Add argparse for command-line options
4. **Docker**: Containerize for easy deployment
5. **GPU Acceleration**: Use CuPy for faster processing

## Integration Possibilities

1. **Other Telescopes**: JWST, Hubble, Spitzer data
2. **Machine Learning**: Classify spectral features automatically
3. **Scientific Analysis**: Extract metrics while creating art
4. **Educational Tools**: Interactive explanations of the science
5. **Museum Installations**: Large-scale displays with real-time data

## Important Context

- SPHEREx mission launches in 2025 for all-sky infrared survey
- Data covers 0.75-5.0 μm in 102 spectral bands
- Current data is pre-launch test/simulation data
- Framework designed to handle real mission data when available

## Request Format for Modifications

When asking for changes, please specify:
1. Which visualization(s) to modify
2. Desired artistic/technical outcome
3. Any performance constraints
4. Target output format/resolution

## Example Prompts for Extension

- "Add a new visualization inspired by [artistic concept]"
- "Modify the cosmic memory visualization to include [feature]"
- "Create a web interface using FastAPI"
- "Add GPU acceleration using CuPy"
- "Export visualizations as video with smooth transitions"

## Files in Project

```
main.py                    # Complete framework code
README.md                 # Quick start guide
API_DOCUMENTATION.md      # Technical reference
Claude.md                 # This file
spherex_cache/           # Downloaded FITS files (gitignored)
*.png                    # Output visualizations
```

## Key Code Sections

- **Lines 75-199**: SPHERExDataAccess class (S3 operations)
- **Lines 202-247**: SpectralArtEngine (color mapping)
- **Lines 250-720**: GenerativeArtworks (all visualizations)
- **Lines 723-773**: SPHERExSonification (audio generation)
- **Lines 776-825**: ArtworkRenderer (display/save)
- **Lines 828-1198**: Main function with memory management

---

**Note**: This is a working artistic/scientific visualization project that successfully bridges astronomy and digital art. The code is production-ready but has room for enhancements based on specific use cases or artistic visions.