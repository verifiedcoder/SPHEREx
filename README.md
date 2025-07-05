# SPHEREx Generative Art Framework

Transform NASA's SPHEREx infrared telescope data into stunning artistic visualizations.

## 🌌 About

This Python framework creates artistic visualizations from SPHEREx (Spectro-Photometer for the History of the Universe) spectral data. It combines scientific data processing with creative algorithms inspired by natural phenomena and data artists like Refik Anadol, Ryoji Ikeda, and Casey Reas.

## 🎨 Visualizations

1. **Cosmic Tapestry** - Weaves spectral threads into textile-like patterns
2. **Spectral Aurora** - Animated flows mimicking cosmic auroras
3. **Ice Crystal Growth** - Simulates formation of cosmic ices
4. **Gravitational Lens** - Einstein rings and light-bending effects
5. **Cosmic Memory** - Data as living, flowing pigment

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy scipy matplotlib astropy boto3
```

### Installation

```bash
git clone https://github.com/yourusername/spherex-generative-art.git
cd spherex-generative-art
```

### Usage

```bash
python main.py
```

The script will:
- Download/cache SPHEREx data from AWS S3 (first run only)
- Generate all five visualizations
- Save high-quality PNG files
- Create audio sonification of spectral data

### Output

- `spherex_stellar_tapestry.png`
- `spherex_spectral_aurora.png`
- `spherex_ice_crystals.png`
- `spherex_gravitational_lens.png`
- `spherex_cosmic_memory.png`

## 📊 Data Source

Accesses public SPHEREx data from NASA/IRSA:
- **S3 Bucket**: `nasa-irsa-spherex`
- **Format**: FITS files with spectral cubes
- **Wavelength Range**: 0.75-5.0 μm (102 bands)

## 🔧 Configuration

Key parameters in `main.py`:
- `DEFAULT_IMAGE_SIZE`: Output resolution (default: 256×256)
- `MAX_PARTICLES`: Particle count for cosmic memory viz
- `CHUNK_SIZE`: Download chunk size
- Cache directory: `./spherex_cache/`

## 💡 Features

- **Automatic Caching**: Downloads are cached locally
- **Memory Efficient**: Handles large spectral cubes
- **Flexible Input**: Works with real or synthetic data
- **Audio Output**: Sonification of spectral signatures

## 🎯 Use Cases

- Scientific outreach and education
- Museum installations
- Data art exhibitions
- Astronomy visualization
- Creative coding workshops

## 🔗 Links

- [SPHEREx Mission](https://en.wikipedia.org/wiki/SPHEREx)
- [NASA JPL SPHEREx](https://www.jpl.nasa.gov/missions/spherex)
- [IRSA Archive](https://irsa.ipac.caltech.edu/Missions/spherex.html)
- [Data User Guide](https://caltech-ipac.github.io/spherex-archive-documentation/)

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- NASA/JPL-Caltech for SPHEREx mission data
- NASA Infrared Science Archive (IRSA)
- Inspired by Refik Anadol, Ryoji Ikeda, and Casey Reas

---

*"Transforming the invisible universe into visible art"*