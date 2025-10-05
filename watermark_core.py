# watermark_core.py

import numpy as np
import pywt
from PIL import Image

# --- Parameters ---
ALPHA = 0.15  # The optimized embedding strength (alpha)
LEVEL = 1     # DWT Decomposition Level
WATERMARK_SIZE = 64 # Assuming a small, 64x64 binary watermark

def embed_watermark_dwt_svd(host_img_path, watermark_img_path, alpha):
    """
    Embeds a watermark using the DWT-SVD method in the LL sub-band.
    
    Args:
        host_img_path (str): Path to the host medical image.
        watermark_img_path (str): Path to the binary watermark image (e.g., patient ID/hash).
        alpha (float): The embedding strength determined by the RL optimizer.
        
    Returns:
        tuple: Modified singular values (for extraction key) and the watermarked image data.
    """
    # 1. Load and Normalize Images
    try:
        host_img = np.array(Image.open(host_img_path).convert('L')) / 255.0
        watermark = np.array(Image.open(watermark_img_path).resize((WATERMARK_SIZE, WATERMARK_SIZE)).convert('L'))
        # Normalize watermark to be binary (0 or 1)
        watermark = (watermark > 128).astype(float) 
    except FileNotFoundError:
        print("Error: Host or Watermark image not found.")
        return None, None

    # 2. DWT Decomposition (using 'haar' or 'db1' wavelet)
    coeffs = pywt.wavedec2(host_img, 'haar', level=LEVEL)
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]
    
    # 3. SVD on LL Sub-band (cA)
    U_LL, S_LL, V_LL = np.linalg.svd(cA)
    
    # 4. SVD on Watermark
    U_W, S_W, V_W = np.linalg.svd(watermark)
    
    # 5. Modify Singular Values (Embedding)
    s_host = S_LL.copy()
    s_wm = S_W.copy()
    
    # Modify the host's singular values with the watermark's singular values
    # Truncate s_wm if it's longer than s_host, or pad if smaller (complex case handled here by truncation/matching)
    s_host_len = len(s_host)
    s_wm_len = len(s_wm)
    
    # Simple modulation on the portion that overlaps
    overlap_len = min(s_host_len, s_wm_len)
    s_host[:overlap_len] = s_host[:overlap_len] + alpha * s_wm[:overlap_len]
    
    # Reconstruct the singular value matrix (diagonal matrix)
    S_MODIFIED = np.zeros(U_LL.shape[0])
    S_MODIFIED[:s_host_len] = s_host
    S_MODIFIED = np.diag(S_MODIFIED)
    
    # 6. Inverse SVD to get the modified LL sub-band
    cA_modified = U_LL @ S_MODIFIED @ V_LL
    
    # 7. Inverse DWT (Reconstruction)
    modified_coeffs = list(coeffs)
    modified_coeffs[0] = (cA_modified, cH, cV, cD) if LEVEL == 1 else (cA_modified, coeffs[1:])
    
    # NOTE: The pywt.waverec2 structure needs to be correct for the level.
    # For LEVEL=1, coeffs is (cA, (cH, cV, cD))
    modified_coeffs = [cA_modified, (cH, cV, cD)] # Correct structure for LEVEL=1
    watermarked_img = pywt.waverec2(modified_coeffs, 'haar')
    
    # Normalize to 0-255 and convert to integer type
    watermarked_img = np.clip(watermarked_img * 255, 0, 255).astype(np.uint8)
    
    # The modified singular values are critical for the extraction process (the "key")
    return s_host, watermarked_img

# Example of how to use this (Requires host.png and wm.png in the same directory)
# from PIL import Image
# modified_s, w_img = embed_watermark_dwt_svd('host.png', 'wm.png', ALPHA)
# if w_img is not None:
#     Image.fromarray(w_img).save('watermarked_image_final.png')
#     print("Watermarking complete. Saved watermarked_image_final.png")
