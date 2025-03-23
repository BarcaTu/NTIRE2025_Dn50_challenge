```markdown
# [NTIRE 2025 Challenge on Image Denoising](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

This repository provides code for denoising noisy images by Team 28 Sky-D.

## Setup Instructions

### 1. Create Model Directory
First, you need to create a model_zoo directory to store the pretrained model weights:
```bash
mkdir -p model_zoo
```

### 2. Download Pretrained Weights
Download our pretrained model weights from this link: [Sky-D Model Weights](https://drive.google.com/file/d/12KQwBiW_mpB0wxnz_wUEIsXgHWG2IgTW/view)

After downloading, place the file in the model_zoo directory with the correct name:
```bash
# Place the downloaded file in the model_zoo directory
mv path/to/downloaded/file.pth ./model_zoo/team28_C2S.pth
```

## How to add noise to images?
```
python add_noise.py
```

## How to test our model

Our model (Team 28 - Sky-D) uses a time-conditioned diffusion model for high-quality image denoising with geometric ensemble. We support adaptive patch sizes for optimal processing of images of various resolutions.

### Basic usage:
```bash
python test_demo.py --data_dir ./path/to/noisy/images --save_dir ./path/to/results --model_id 28 --stride 384 --ensemble --adaptive_patches --hybrid_test
```

### Command-line arguments:
- `--data_dir`: Directory containing noisy input images
- `--save_dir`: Directory to save the denoised results
- `--model_id`: Use 28 for our C2S model
- `--stride`: Stride for overlapping patches (default: 384)
- `--ensemble`: Enable geometric self-ensemble for better results
- `--adaptive_patches`: Use adaptive patch sizes based on image dimensions
- `--hybrid_test`: Process test images without ground truth

### Advanced features:

1. **Adaptive Patch Sizes**: Our model automatically selects optimal patch sizes based on image dimensions:
   - Large images (≥896×896): 896×896 patches
   - Medium images (≥768×768): 768×768 patches
   - Small images (≥512×512): 512×512 patches
   - Tiny images: Processed directly

2. **Geometric Self-Ensemble**: Applies 8 different transformations (rotations, flips) and averages the results for improved quality.

3. **Noise Estimation**: Automatically estimates noise level for each image and adjusts denoising strength accordingly.

## Example Results

Our TimeDiffiT model achieves excellent denoising quality with efficient processing.

## Notes:
1. When using `--hybrid_test` mode, the metrics (PSNR/SSIM) should be disregarded as they are calculated against the noisy input, not ground truth.
2. For optimal performance, make sure you have enough GPU memory (at least 8GB recommended).
3. Processing large images with ensemble mode is more memory-intensive but produces better results.

## How to add your model to this baseline?
1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1XVa8LIaAURYpPvMf7i-_Yqlzh-JsboG0hvcnp-oI9rs/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
```