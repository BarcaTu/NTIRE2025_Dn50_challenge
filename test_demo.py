import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from utils.model_summary import get_model_activation, get_model_flops
from utils import utils_logger
from utils import utils_image as util
from skimage.restoration import estimate_sigma
import numpy as np
from collections import OrderedDict


# Define helper functions for normalization
def normalize_to_neg_one_to_one(x):
    """Normalize tensor values from [0,1] to [-1,1]"""
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    """Convert tensor values from [-1,1] back to [0,1]"""
    return (x + 1) * 0.5

def identity(x):
    """Identity function, returns input unchanged"""
    return x


# Custom patch-based processing functions
def overlapping_grid_indices(inp, output_size, r=None):
    """
    Generate overlapping grid indices for patch-based processing.
    """
    _, c, h, w = inp.shape
    r = 384 if r is None else r  # Using stride of 384 as requested
    h_list = [i for i in range(0, h - output_size + 1, r)]
    w_list = [i for i in range(0, w - output_size + 1, r)]
    
    # Make sure we cover the entire image by adding the final index if needed
    if h - output_size not in h_list:
        h_list.append(h - output_size)
    if w - output_size not in w_list:
        w_list.append(w - output_size)
        
    return h_list, w_list


def unpatchify_restore_overlapping(x, model=None, corners=None, p_size=64, time=None, model_time_conditioning=False):
    """
    Restore image by processing overlapping patches with time conditioning support.
    
    Args:
        x: Input tensor
        model: Neural network model
        corners: List of (h, w) coordinates for patches
        p_size: Patch size
        time: Time parameter for conditioning (noise level)
        model_time_conditioning: Whether model uses time conditioning
    """
    with torch.no_grad():
        x_grid_mask = torch.zeros_like(x, device=x.device)
        e_output = torch.zeros_like(x, device=x.device)
        
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
            
        for (hi, wi) in corners:
            x_patch = x[:, :, hi:hi+p_size, wi:wi+p_size]
            if model is None:
                e_output[:, :, hi:hi + p_size, wi:wi + p_size] += x_patch
            else:
                if model_time_conditioning and time is not None:
                    noise_amount = time  # Use provided time value
                    output_patch = model(x1=x_patch, time=noise_amount)
                else:
                    # Regular model forward pass
                    output_patch = model(x_patch)
                e_output[:, :, hi:hi + p_size, wi:wi + p_size] += output_patch
                
        # Average the overlapping regions
        e = torch.div(e_output, x_grid_mask)
    return e


def patch_based_restoration(x, model=None, r=None, patch_size=64, time=None, model_time_conditioning=False):
    """
    Perform patch-based restoration with time conditioning support.
    
    Args:
        x: Input tensor
        model: Neural network model
        r: Stride for overlapping patches
        patch_size: Size of each patch
        time: Time parameter for conditioning (noise level)
        model_time_conditioning: Whether model uses time conditioning
    """
    p_size = patch_size
    h_list, w_list = overlapping_grid_indices(x, output_size=p_size, r=r)
    corners = [(i, j) for i in h_list for j in w_list]
    x_output = unpatchify_restore_overlapping(
        x, 
        model=model, 
        corners=corners, 
        p_size=p_size, 
        time=time,
        model_time_conditioning=model_time_conditioning
    )
    return x_output


def geometric_ensemble_inference(image, model, patch_size=512, stride=384, device='cuda'):
    """
    Apply geometric self-ensemble technique to image denoising with adaptive patch size support.
    
    Args:
        image: Input noisy image tensor [B, C, H, W]
        model: Neural network model
        patch_size: Size of patch to use for patch-based processing
        stride: Stride to use for overlapping patches
        device: Device to run inference on
        
    Returns:
        Ensemble denoised output tensor [B, C, H, W]
    """
    # Ensure we're working with a batch of size 1
    assert image.size(0) == 1, "Geometric ensemble expects batch size 1"
    
    model_time_conditioning = hasattr(model, 'time_mlp')
    B, C, H, W = image.size()
    
    # Initialize storage for outputs
    ensemble_outputs = []
    
    # Loop through all 8 transformations
    for t in range(8):
        # Apply the geometric transformation to the input
        if t == 0:  # Identity
            transformed_img = image
        elif t == 1:  # Horizontal flip
            transformed_img = torch.flip(image, dims=[3])
        elif t == 2:  # Vertical flip
            transformed_img = torch.flip(image, dims=[2])
        elif t == 3:  # 90° rotation
            transformed_img = torch.rot90(image, k=1, dims=[2, 3])
        elif t == 4:  # 180° rotation
            transformed_img = torch.rot90(image, k=2, dims=[2, 3])
        elif t == 5:  # 270° rotation
            transformed_img = torch.rot90(image, k=3, dims=[2, 3])
        elif t == 6:  # Horizontal flip + 90° rotation
            transformed_img = torch.rot90(torch.flip(image, dims=[3]), k=1, dims=[2, 3])
        elif t == 7:  # Vertical flip + 90° rotation
            transformed_img = torch.rot90(torch.flip(image, dims=[2]), k=1, dims=[2, 3])
        
        # Get dimensions of transformed image
        t_H, t_W = transformed_img.shape[2], transformed_img.shape[3]
        
        # Determine if we need patch-based processing
        use_patch_based = (t_H > patch_size or t_W > patch_size)
        
        # Estimate noise level
        noise_amount = torch.zeros(1).to(device)
        if C == 3:  # RGB image
            channel_sigmas = []
            for c in range(C):
                img_channel = transformed_img[0, c].detach().cpu().numpy()
                channel_sigmas.append(estimate_sigma(img_channel))
            noise_amount[0] = torch.tensor(np.mean(channel_sigmas), device=device)
        else:
            img = transformed_img[0, 0].detach().cpu().numpy()
            noise_amount[0] = torch.tensor(estimate_sigma(img), device=device)
        
        # Process the image
        if use_patch_based:
            # Use patch-based processing for large images
            transformed_output = patch_based_restoration(
                transformed_img,
                model=model,
                r=stride,
                patch_size=patch_size,
                time=noise_amount,
                model_time_conditioning=model_time_conditioning
            )
        else:
            # Direct processing for small images
            with torch.no_grad():
                if model_time_conditioning:
                    transformed_output = model(x1=transformed_img, time=noise_amount)
                else:
                    transformed_output = model(transformed_img)
        
        # Apply the inverse transformation
        if t == 0:  # Identity
            aligned_output = transformed_output
        elif t == 1:  # Horizontal flip
            aligned_output = torch.flip(transformed_output, dims=[3])
        elif t == 2:  # Vertical flip
            aligned_output = torch.flip(transformed_output, dims=[2])
        elif t == 3:  # 90° rotation
            aligned_output = torch.rot90(transformed_output, k=3, dims=[2, 3])
        elif t == 4:  # 180° rotation
            aligned_output = torch.rot90(transformed_output, k=2, dims=[2, 3])
        elif t == 5:  # 270° rotation
            aligned_output = torch.rot90(transformed_output, k=1, dims=[2, 3])
        elif t == 6:  # Horizontal flip + 90° rotation
            aligned_output = torch.flip(torch.rot90(transformed_output, k=3, dims=[2, 3]), dims=[3])
        elif t == 7:  # Vertical flip + 90° rotation
            aligned_output = torch.flip(torch.rot90(transformed_output, k=3, dims=[2, 3]), dims=[2])
        
        # Add to ensemble collection
        ensemble_outputs.append(aligned_output)
    
    # Average all outputs
    ensemble_result = torch.mean(torch.stack(ensemble_outputs, dim=0), dim=0)
    
    return ensemble_result

    
def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        # SGN test
        from models.team00_SGN import SGNDN3
        name, data_range = f"{model_id:02}_RFDN_baseline", 1.0
        model_path = os.path.join('model_zoo', 'team00_sgn.ckpt')
        model = SGNDN3()

        state_dict = torch.load(model_path)["state_dict"]
        state_dict.pop("current_val_metric")
        state_dict.pop("best_val_metric")
        state_dict.pop("best_iter")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.find("model.") >= 0:
                new_state_dict[k.replace("model.", "")] = v
        model.load_state_dict(new_state_dict, strict=True)
    elif model_id == 28:
        from models.team28_C2S import TimeDiffiT_ResNet_color
        name, data_range = f"{model_id:02}_TimeDiffiT_ResNet_color", 1.0  # Using 1.0 for [0,1] input range
        model_path = os.path.join('model_zoo', 'team28_C2S.pth')
        
        # Initialize your model
        model = TimeDiffiT_ResNet_color(dim=64)
        
        # Load checkpoint
        state_dict = torch.load(model_path, map_location=device)
        # Clean up state dict if needed
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v
                
        model.load_state_dict(clean_state_dict, strict=True)
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # print(model)
    model.eval()
    tile = 512
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model, name, data_range, tile


def select_dataset(data_dir, mode):
    if mode == "test":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_test_noise50/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_test_HR/{i:04}.png")
            ) for i in range(901, 1001)
        ]
        # [f"DIV2K_test_LR/{i:04}.png" for i in range(901, 1001)]
    elif mode == "valid":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_valid_noise50/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_valid_HR/{i:04}.png")
            ) for i in range(801, 901)
        ]
    # elif mode == "hybrid_test":
    #     path = [
    #         (
    #             p.replace("_HR", "_LR").replace(".png", "noise50.png"),
    #             p
    #         ) for p in sorted(glob.glob(os.path.join(data_dir, "LSDIR_DIV2K_test_HR/*.png")))
    #     ]
    elif mode == "hybrid_test":
        path = [
            (
                p,
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")
    return path


def forward(img_lq, model, tile=None, tile_overlap=32, scale=1, adaptive_patch_sizes=None, ensemble=False, device='cuda'):
    """
    Enhanced forward function with support for adaptive patch sizes and geometric ensemble
    """
    is_timediffit = hasattr(model, 'time_mlp')
    
    # Apply normalization for TimeDiffiT model
    if is_timediffit:
        img_lq = normalize_to_neg_one_to_one(img_lq)
    
    # Use geometric ensemble if requested (only for TimeDiffiT model and batch_size=1)
    if ensemble and is_timediffit and img_lq.size(0) == 1:
        # Determine appropriate patch size based on image dimensions if using adaptive_patch_sizes
        if adaptive_patch_sizes is not None:
            img_height, img_width = img_lq.shape[2], img_lq.shape[3]
            if img_height >= adaptive_patch_sizes['large'] and img_width >= adaptive_patch_sizes['large']:
                patch_size = adaptive_patch_sizes['large']
            elif img_height >= adaptive_patch_sizes['medium'] and img_width >= adaptive_patch_sizes['medium']:
                patch_size = adaptive_patch_sizes['medium']
            elif img_height >= adaptive_patch_sizes['small'] and img_width >= adaptive_patch_sizes['small']:
                patch_size = adaptive_patch_sizes['small']
            else:
                patch_size = min(img_height, img_width)
        else:
            patch_size = tile
            
        output = geometric_ensemble_inference(img_lq, model, patch_size, tile_overlap, device)
    else:
        # Standard processing (either direct or patch-based)
        if tile is None or (img_lq.shape[2] <= tile and img_lq.shape[3] <= tile):
            # Direct processing for small images
            if is_timediffit:
                # For TimeDiffiT model with time conditioning
                b, c, h, w = img_lq.size()
                
                # Estimate noise level
                noise_amount = torch.zeros(b).to(img_lq.device)
                for i in range(b):
                    if c == 3:  # RGB image
                        channel_sigmas = []
                        for ch in range(c):
                            img_channel = img_lq[i, ch].detach().cpu().numpy()
                            channel_sigmas.append(estimate_sigma(img_channel))
                        noise_amount[i] = torch.tensor(np.mean(channel_sigmas), device=img_lq.device)
                    else:
                        img = img_lq[i, 0].detach().cpu().numpy()
                        noise_amount[i] = torch.tensor(estimate_sigma(img), device=img_lq.device)
                
                # Forward with time conditioning
                output = model(x1=img_lq, time=noise_amount)
            else:
                # Regular forward
                output = model(img_lq)
        else:
            # Patch-based processing for large images with adaptive patch sizes
            if adaptive_patch_sizes is not None and is_timediffit:
                img_height, img_width = img_lq.shape[2], img_lq.shape[3]
                if img_height >= adaptive_patch_sizes['large'] and img_width >= adaptive_patch_sizes['large']:
                    patch_size = adaptive_patch_sizes['large']
                elif img_height >= adaptive_patch_sizes['medium'] and img_width >= adaptive_patch_sizes['medium']:
                    patch_size = adaptive_patch_sizes['medium']
                elif img_height >= adaptive_patch_sizes['small'] and img_width >= adaptive_patch_sizes['small']:
                    patch_size = adaptive_patch_sizes['small']
                else:
                    patch_size = min(img_height, img_width)
            else:
                patch_size = tile
            
            # Estimate noise level for TimeDiffiT model
            noise_amount = None
            if is_timediffit:
                b, c, h, w = img_lq.size()
                noise_amount = torch.zeros(b).to(img_lq.device)
                for i in range(b):
                    if c == 3:  # RGB image
                        channel_sigmas = []
                        for ch in range(c):
                            img_channel = img_lq[i, ch].detach().cpu().numpy()
                            channel_sigmas.append(estimate_sigma(img_channel))
                        noise_amount[i] = torch.tensor(np.mean(channel_sigmas), device=img_lq.device)
                    else:
                        img = img_lq[i, 0].detach().cpu().numpy()
                        noise_amount[i] = torch.tensor(estimate_sigma(img), device=img_lq.device)
            
            # Use our custom patch-based processing
            output = patch_based_restoration(
                img_lq,
                model=model,
                r=tile_overlap,
                patch_size=patch_size,
                time=noise_amount,
                model_time_conditioning=is_timediffit
            )
    
    # Convert back to [0,1] range if it's TimeDiffiT model
    if is_timediffit:
        output = unnormalize_to_zero_to_one(output)
        
    return output


def run(model, model_name, data_range, tile, logger, device, args, mode="test"):
    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []
    
    # Setup adaptive patch sizes if enabled
    adaptive_patch_sizes = None
    if args.adaptive_patches:
        # adaptive_patch_sizes = {
        #     'large': args.large_patch,
        #     'medium': args.medium_patch,
        #     'small': args.small_patch
        # }
        
        # Define adaptive patch sizes
        adaptive_patch_sizes = {
            'large': 896,
            'medium': 768,
            'small': 512
        }
        logger.info(f"Using adaptive patch sizes: {adaptive_patch_sizes}")
    
    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_noisy, img_hr) in enumerate(data_path):
        # --------------------------------
        # (1) img_noisy
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_noisy = util.imread_uint(img_noisy, n_channels=3)
        img_noisy = util.uint2tensor4(img_noisy, data_range)
        img_noisy = img_noisy.to(device)
        
        # Log the image dimensions and selected patch size
        img_height, img_width = img_noisy.shape[2], img_noisy.shape[3]
        if adaptive_patch_sizes is not None:
            if img_height >= adaptive_patch_sizes['large'] and img_width >= adaptive_patch_sizes['large']:
                patch_size = adaptive_patch_sizes['large']
                category = "large"
            elif img_height >= adaptive_patch_sizes['medium'] and img_width >= adaptive_patch_sizes['medium']:
                patch_size = adaptive_patch_sizes['medium']
                category = "medium"
            elif img_height >= adaptive_patch_sizes['small'] and img_width >= adaptive_patch_sizes['small']:
                patch_size = adaptive_patch_sizes['small']
                category = "small"
            else:
                patch_size = min(img_height, img_width)
                category = "direct"
            logger.info(f"Processing image {i+1}: {img_name}, shape=({img_height}×{img_width}), category={category}, patch_size={patch_size}")
        else:
            logger.info(f"Processing image {i+1}: {img_name}, shape=({img_height}×{img_width})")

        # --------------------------------
        # (2) img_dn
        # --------------------------------
        start.record()
        img_dn = forward(
            img_lq=img_noisy, 
            model=model, 
            tile=tile, 
            tile_overlap=args.stride, 
            adaptive_patch_sizes=adaptive_patch_sizes, 
            ensemble=args.ensemble,
            device=device
        )
        end.record()
        torch.cuda.synchronize()
        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds
        img_dn = util.tensor2uint(img_dn, data_range)

        # --------------------------------
        # (3) img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------
        psnr = util.calculate_psnr(img_dn, img_hr, border=border)
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_dn, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        util.imsave(img_dn, os.path.join(save_path, img_name+ext))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"])
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
        
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memory", results[f"{mode}_memory"]))
    logger.info("------> Average runtime of ({}) is : {:.6f} milliseconds".format(
        "test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))
    logger.info("------> Average PSNR: {:.4f} dB".format(results[f"{mode}_ave_psnr"]))

    return results


def main(args):
    utils_logger.logger_info("NTIRE2025-Dn50", log_path="NTIRE2025-Dn50.log")
    logger = logging.getLogger("NTIRE2025-Dn50")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # Print model-specific information
    if args.model_id == 1:
        logger.info(f"Using TimeDiffiT model with the following settings:")
        logger.info(f"  - Stride: {args.stride}")
        logger.info(f"  - Default patch size: {args.patch_size}")
        if args.adaptive_patches:
            logger.info(f"  - Using adaptive patch sizes: Large={args.large_patch}, Medium={args.medium_patch}, Small={args.small_patch}")
        logger.info(f"  - Geometric ensemble: {'Enabled' if args.ensemble else 'Disabled'}")

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if True:
        # --------------------------------
        # restore image
        # --------------------------------

        if args.hybrid_test:
            # inference on the DIV2K and LSDIR test set
            valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="hybrid_test")
            # record PSNR, runtime
            results[model_name] = valid_results
        else:
            # inference on the validation set
            valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
            # record PSNR, runtime
            results[model_name] = valid_results

            if args.include_test:
                # inference on the test set
                test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
                results[model_name].update(test_results)

        input_dim = (3, 256, 256)  # set the input dimension
        activations, num_conv = get_model_activation(model, input_dim)
        activations = activations/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        flops = get_model_flops(model, input_dim, False)
        flops = flops/10**9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        # print(v.keys())
        if args.hybrid_test:
            val_psnr = f"{v['hybrid_test_ave_psnr']:2.2f}"
            val_time = f"{v['hybrid_test_ave_runtime']:3.2f}"
            mem = f"{v['hybrid_test_memory']:2.2f}"
        else:
            val_psnr = f"{v['valid_ave_psnr']:2.2f}"
            val_time = f"{v['valid_ave_runtime']:3.2f}"
            mem = f"{v['valid_memory']:2.2f}"
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2025-Dn50")
    parser.add_argument("--data_dir", default="./NTIRE2025_Challenge/input", type=str)
    parser.add_argument("--save_dir", default="./NTIRE2025_Challenge/results", type=str)
    parser.add_argument("--model_id", default=28, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the DIV2K test set")
    parser.add_argument("--hybrid_test", action="store_true", help="Hybrid test on DIV2K and LSDIR test set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")
    
    # TimeDiffiT model specific arguments
    # parser.add_argument("--patch_size", type=int, default=512, help="Default patch size for processing")
    parser.add_argument("--stride", type=int, default=384, help="Stride for overlapping patches")
    parser.add_argument("--ensemble", action="store_true", help="Use geometric self-ensemble for inference")
    parser.add_argument("--adaptive_patches", action="store_true", help="Use adaptive patch sizes based on image dimensions")
    # parser.add_argument("--large_patch", type=int, default=896, help="Large patch size for large images")
    # parser.add_argument("--medium_patch", type=int, default=768, help="Medium patch size for medium images")
    # parser.add_argument("--small_patch", type=int, default=512, help="Small patch size for small images")

    args = parser.parse_args()
    pprint(args)
    utils_logger.logger_info("NTIRE2025-Dn50", log_path="NTIRE2025-Dn50.log")
    logger = logging.getLogger("NTIRE2025-Dn50")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------
    # Warning about hybrid_test mode
    # --------------------------------
    if args.hybrid_test:
        logger.warning("=" * 80)
        logger.warning("WARNING: In hybrid_test mode, the select_dataset function is currently set to use")
        logger.warning("         the same file for both input and ground truth. This will result in")
        logger.warning("         artificially high PSNR/SSIM metrics as we're comparing the output")
        logger.warning("         against the same noisy input, not a clean ground truth.")
        logger.warning("         The results will be saved correctly, but the metrics are not valid.")
        logger.warning("=" * 80)

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # Print model-specific information
    if args.model_id == 28:
        logger.info(f"Using TimeDiffiT model with the following settings:")
        logger.info(f"  - Stride: {args.stride}")
        logger.info(f"  - Default patch size: 512")
        if args.adaptive_patches:
            logger.info(f"  - Using adaptive patch sizes: Large=896, Medium=768, Small=512")
        logger.info(f"  - Geometric ensemble: {'Enabled' if args.ensemble else 'Disabled'}")

    main(args)