# TinyNeRF

A lightweight implementation of Neural Radiance Fields (NeRF) for novel view synthesis.

## Overview

TinyNeRF is a simplified version of NeRF that enables the generation of novel views from trained neural radiance fields. This implementation uses a 2-bit quantized model to reduce size while maintaining quality.

## Project Structure

```
TinyNeRF/
├── tiny_nerf.py      # Main implementation file
├── models/
│   └── 2bit_nerf.pth # Pre-trained 2-bit quantized NeRF model
|   └── 4bit_nerf.pth # Pre-trained 4-bit quantized NeRF model
|   └── 8bit_nerf.pth # Pre-trained 8-bit quantized NeRF model (best in inference and accuracy)
└── novel_views/      # Directory for rendered output images (create before running)
```

## Prerequisites

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/smehta9711/TinyNeRF.git
   cd TinyNeRF
   ```

2. Create the output directory:
   ```
   mkdir novel_views
   ```

3. Install required dependencies:
   ```
   pip install torch numpy matplotlib
   ```

## Usage

1. Make sure to follow the folder structure exactly as shown above, or modify the model path in the code.

2. Run the main script:
   ```
   python tiny_nerf.py
   ```

3. When the Matplotlib window opens, use your keyboard to adjust the camera position:
   - Arrow keys: Move camera position
   - WASD: Alternative camera movement
   - Q/E: Move up/down

4. Wait for the rendering to complete. The generated novel view will be saved in the `novel_views` directory.

## Customization

To use a different model, modify the `nerf_path` variable at the end of `tiny_nerf.py`:

```python
nerf_path = "path/to/your/model.pth"
```

## Demo

Watch our [live demo video](https://drive.google.com/file/d/18zcGvH3jPVCReTmAej3ccYiRCr75xZ7R/view?usp=sharing) to see TinyNeRF in action.

## Citation

If you use this implementation in your research, please cite:

```
@article{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P. and Tancik, Matthew and Barron, Jonathan T. and Ramamoorthi, Ravi and Ng, Ren},
  journal={ECCV},
  year={2020}
}
```