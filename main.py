import numpy as np
import cv2
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Lambda # Import the Lambda layer

# --- Noble Component 1: CNN Feature Extractor Definition ---
def create_feature_extractor(input_shape=(256, 256, 1)):
    """
    Creates a simple CNN model to extract a robust feature map from the DWT sub-band.
    The output (Feature_Conv_2) will replace the traditional CA band for SVD embedding.
    """
    
    # Define the Input Tensor
    input_tensor = layers.Input(shape=input_shape, name='Input_DWT_Band')
    
    # Layer 1: Feature extraction
    x = layers.Conv2D(
        32, (5, 5), activation='relu', padding='same', name='Feature_Conv_1'
    )(input_tensor)
    
    # Layer 2: Intermediate Feature Map (256x256x16)
    feature_output = layers.Conv2D(
        16, (3, 3), activation='relu', padding='same', name='Feature_Conv_2'
    )(x)
    
    # FIX: Using Lambda layer to wrap tf.reduce_mean
    # This reduces the 256x256x16 map to a 256x256x1 map by averaging across the channel axis (-1).
    final_2d_feature = Lambda(
        lambda z: tf.reduce_mean(z, axis=-1, keepdims=True), 
        name='Channel_Average_Pool'
    )(feature_output)
    
    # Create the functional model
    feature_extractor = models.Model(
        inputs=input_tensor, 
        outputs=final_2d_feature, # Output is 256x256x1
        name='CNN_Feature_Refiner'
    )
    # Compile the model with a dummy optimizer/loss, as it's only used for inference
    feature_extractor.compile(optimizer='adam', loss='mse') 
    
    return feature_extractor

# Initialize the CNN Model (must be done before use)
CNN_REFINE_MODEL = create_feature_extractor()
print("CNN Feature Refiner Initialized.")
# --- End of Noble Component 1 ---


# --- Attack Suite Class (NEW) ---
class AttackSuite:
    """Contains functions to apply common image processing attacks."""

    @staticmethod
    def gaussian_noise(img, mean=0, var=0.001):
        """Adds Gaussian noise to the image."""
        row, col, ch = img.shape
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img.astype(np.float64) / 255.0 + gauss
        noisy = np.clip(noisy, 0.0, 1.0)
        return np.uint8(noisy * 255.0)

    @staticmethod
    def median_filter(img, ksize=3):
        """Applies a Median Filter (effective against salt-and-pepper noise)."""
        # Ensure ksize is odd, as required by OpenCV
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.medianBlur(img, ksize)

    @staticmethod
    def jpeg_compression(img, quality=50):
        """Simulates JPEG compression attack."""
        # Convert RGB to BGR for OpenCV encoding
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Encode to JPEG in memory
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        
        # Decode back to image (BGR)
        decimg_bgr = cv2.imdecode(encimg, 1)
        
        # Convert back to RGB
        return cv2.cvtColor(decimg_bgr, cv2.COLOR_BGR2RGB)


# --- Helper Functions (NC Calculation and SVD Extraction) ---

def normalized_correlation(img1, img2):
    """Calculates Normalized Correlation (NC) between two images."""
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

def extract_dwt_svd(W_img_RGB, H_img_RGB, u_wm, v_wm, CH, CV, CD, WaveletName, SF, pcaoutput_H, eigenvec1_H, m1_H):
    """
    Extracts the watermark from the watermarked image using the inverse process,
    including the CNN refinement step for robustness.
    """
    
    # 1. Inverse PCA on Watermarked RGB image
    img1 = W_img_RGB[:,:,0]
    m, n = img1.shape
    
    # Reshape and subtract mean (using the mean from the host image PCA)
    I_W_flat = np.hstack([W_img_RGB[:,:,i].T.flatten().reshape(-1, 1) for i in range(3)])
    I1_W = I_W_flat.astype(np.float64) - m1_H

    # Inverse transformation to get the watermarked PCA output components
    pcaoutput_W = I1_W @ eigenvec1_H

    # The 3rd component is the one that was watermarked
    Watermarked_Features = pcaoutput_W[:, 2].reshape(m, n).T
    
    # 2. DWT Decomposition on Watermarked Features (to get the watermarked CA band)
    coeffs_W = pywt.dwt2(Watermarked_Features, WaveletName)
    CA_W_orig, _ = coeffs_W 
    
    # 3. DWT Decomposition on Original Host Features (for subtraction)
    Host_Features = pcaoutput_H[:, 2].reshape(m, n).T
    coeffs_H = pywt.dwt2(Host_Features, WaveletName)
    CA_H_orig, _ = coeffs_H
    
    # 4. Noble Step: Retrieve the CNN-Refined CA bands for SVD
    global CNN_REFINE_MODEL

    # Prepare input for CNN: Add batch dimension and channel dimension
    CA_W_input = np.expand_dims(np.expand_dims(CA_W_orig, axis=0), axis=-1)
    CA_H_input = np.expand_dims(np.expand_dims(CA_H_orig, axis=0), axis=-1)

    # Use the CNN model to get the refined bands
    CA_W_refined_batch = CNN_REFINE_MODEL.predict(CA_W_input, verbose=0)
    CA_H_refined_batch = CNN_REFINE_MODEL.predict(CA_H_input, verbose=0)
    
    CA_W_refined = CA_W_refined_batch[0, :, :, 0]
    CA_H_refined = CA_H_refined_batch[0, :, :, 0]


    # 5. SVD on the Refined DWT bands
    # Note: We only need the singular values for subtraction
    _, s_host_W, _ = np.linalg.svd(CA_W_refined)
    _, s_host_H, _ = np.linalg.svd(CA_H_refined)
    
    # Ensure singular values are same length (min length)
    min_len = min(len(s_host_W), len(s_host_H))
    s_host_W = s_host_W[:min_len]
    s_host_H = s_host_H[:min_len]

    # 6. Extract modified singular values: s_wm' = (s_host_W' - s_host_H) / SF
    # We must construct a diagonal matrix for subtraction
    s_extracted_diag = np.diag((s_host_W - s_host_H) / SF)
    
    # 7. Inverse SVD to get extracted watermark
    # Create the full singular matrix for I-SVD using the original watermark's shape
    s_wm_extracted = np.diag(s_extracted_diag)
    s_extracted = np.zeros((u_wm.shape[0], v_wm.shape[0]))
    s_extracted[:len(s_wm_extracted), :len(s_wm_extracted)] = np.diag(s_wm_extracted)
    
    extracted_wm = u_wm @ s_extracted @ v_wm.T

    # 8. Normalize the extracted watermark for display and NC calculation
    extracted_wm_normalized = (extracted_wm - np.min(extracted_wm)) / (np.max(extracted_wm) - np.min(extracted_wm))
    extracted_wm_normalized = extracted_wm_normalized.astype(np.float64)

    # Reload original watermark for NC calculation
    Watermark_orig = cv2.imread('fused.png')
    if Watermark_orig.ndim == 3:
        Watermark_orig = cv2.cvtColor(Watermark_orig, cv2.COLOR_BGR2GRAY)
    Watermark_orig = cv2.resize(Watermark_orig, (extracted_wm_normalized.shape[1], extracted_wm_normalized.shape[0]))
    Watermark_orig = Watermark_orig.astype(np.float64) / 255.0
    
    nc_val = normalized_correlation(Watermark_orig, extracted_wm_normalized)
    
    return extracted_wm_normalized, nc_val


# --- Main Embedding and Extraction Routine ---

if __name__ == '__main__':
    # --- Noble Component 2: Dynamic Alpha from RL Agent ---
    SF = 0.05  
    WaveletName = 'haar'
    
    # --- Load and Prepare Images ---
    try:
        # OpenCV loads as BGR. Convert to RGB for consistency.
        img_bgr = cv2.imread('House.bmp')
        if img_bgr is None: raise FileNotFoundError("House.bmp not found.")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        
        Watermark_bgr = cv2.imread('fused.png')
        if Watermark_bgr is None: raise FileNotFoundError("fused.png not found.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please upload 'House.bmp' and 'fused.png' to your Colab environment.")
        exit()

    # Watermark preparation
    if Watermark_bgr.ndim == 3:
        Watermark = cv2.cvtColor(Watermark_bgr, cv2.COLOR_BGR2GRAY)
    else:
        Watermark = Watermark_bgr
        
    Watermark = Watermark.astype(np.float64) / 255.0  # Normalize
    Watermark = cv2.resize(Watermark, (256, 256)) # DWT(512)->CA(256), so Watermark is 256x256
    
    print("Images loaded and prepared.")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(img), plt.title('Original RGB Host Image')
    plt.subplot(1, 2, 2), plt.imshow(Watermark, cmap='gray'), plt.title('Watermark Image')
    plt.show()

    # --- PCA PART: Select Robust Host Component (Host_image) ---
    m, n, _ = img.shape
    
    I_flat = np.hstack([img[:,:,i].T.flatten().reshape(-1, 1) for i in range(3)])
    
    m1 = np.mean(I_flat, axis=0)
    I1 = I_flat.astype(np.float64) - m1
    covv = np.cov(I1.T)
    eigenvalue, eigenvec = np.linalg.eig(covv)
    
    sorted_indices = np.argsort(eigenvalue)[::-1]
    eigenvec1 = eigenvec[:, sorted_indices]
    
    pcaoutput = I1 @ eigenvec1

    # Host_image: 3rd principal component (index 2 after sorting)
    Host_image = pcaoutput[:, 2].reshape(n, m).T
    Host_Features = Host_image # 512x512 array
    
    print(f"PCA Host Component Shape: {Host_image.shape}")


    # --- DWT-SVD Embedding with CNN Refinement ---
    
    # 1. DWT Decomposition (Level 1) on 512x512 host component
    coeffs = pywt.dwt2(Host_Features, WaveletName)
    CA_orig, details = coeffs
    CH, CV, CD = details
    
    # 2. Noble Step: CNN Feature Refinement
    CA_input = np.expand_dims(np.expand_dims(CA_orig, axis=0), axis=-1)
    CA_refined_batch = CNN_REFINE_MODEL.predict(CA_input, verbose=0)
    EmbeddingBand = CA_refined_batch[0, :, :, 0] # Output shape is (256, 256)
    
    print(f"Refined Embedding Band Shape (CNN Output): {EmbeddingBand.shape}")

    # 3. SVD on the Refined Embedding Band
    u_host, s_host_diag, v_host = np.linalg.svd(EmbeddingBand)
    s_host = np.diag(s_host_diag)
    
    # 4. SVD on the Watermark
    u_wm, s_wm_diag, v_wm = np.linalg.svd(Watermark)
    s_wm = np.diag(s_wm_diag)
    
    # 5. Watermark Embedding: s_host' = s_host + SF * s_wm
    s_wm_resized = np.zeros_like(s_host)
    min_size = min(s_host.shape[0], s_wm.shape[0])
    s_wm_resized[:min_size, :min_size] = s_wm[:min_size, :min_size]
    
    s_modified = s_host + SF * s_wm_resized
    
    # 6. Inverse SVD to create the modified DWT band
    Wimg_refined = u_host @ s_modified @ v_host
    
    # 7. Inverse CNN Refinement: Replace the original DWT CA band with the watermarked, refined band.
    coeffs_w = (Wimg_refined, (CH, CV, CD)) 
    
    # 8. Inverse DWT Reconstruction
    Watermarked_Features = pywt.idwt2(coeffs_w, WaveletName) # 512x512

    
    # --- Inverse PCA to reconstruct the RGB image ---
    t1_flat = Watermarked_Features.T.flatten().reshape(-1, 1)
    t2 = pcaoutput.copy()
    t2[:, 2] = t1_flat.flatten()
    V_inv = np.linalg.inv(eigenvec1)
    original = t2 @ V_inv
    
    I2 = original + m1
    I2 = np.clip(I2, 0, 255)
    I2 = np.uint8(np.round(I2))

    back_to_original_img = np.zeros_like(img)
    back_to_original_img[:, :, 0] = I2[:, 0].reshape(n, m).T
    back_to_original_img[:, :, 1] = I2[:, 1].reshape(n, m).T
    back_to_original_img[:, :, 2] = I2[:, 2].reshape(n, m).T
    
    watermarked_img = np.uint8(back_to_original_img)
    
    print("Embedding complete.")
    plt.figure(figsize=(6, 6))
    plt.imshow(watermarked_img), plt.title('RGB Watermarked Image (QuSecureMed Hybrid)')
    plt.show()

    # --- Performance Metrics (Imperceptibility) ---
    img_uint8 = np.uint8(img)
    
    psnr_val = psnr_metric(img_uint8, watermarked_img)
    ssim_val = ssim_metric(img_uint8, watermarked_img, channel_axis=-1, data_range=255)
    
    print(f'PSNR (Imperceptibility): {psnr_val:.4f} dB')
    print(f'SSIM (Structural Similarity): {ssim_val:.4f}')
    
    
    # --- Watermark Extraction & Robustness Test ---
    print("\n--- Robustness Test Results (NC) ---")
    results = {}
    
    # 1. No Attack
    ExtractedWatermark_normalized, nc_val = extract_dwt_svd(
        watermarked_img, img_uint8, u_wm, v_wm.T, CH, CV, CD, WaveletName, SF, 
        pcaoutput, eigenvec1, m1
    )
    results['No Attack'] = nc_val
    print(f"NC (No Attack): {nc_val:.4f}")

    # 2. Attack: Gaussian Noise (Variance 0.005)
    attacked_img_noise = AttackSuite.gaussian_noise(watermarked_img, var=0.005)
    _, nc_noise = extract_dwt_svd(
        attacked_img_noise, img_uint8, u_wm, v_wm.T, CH, CV, CD, WaveletName, SF, 
        pcaoutput, eigenvec1, m1
    )
    results['Gaussian Noise (0.005)'] = nc_noise
    print(f"NC (Gaussian Noise): {nc_noise:.4f}")
    
    # 3. Attack: Median Filter (Kernel 3x3)
    attacked_img_filter = AttackSuite.median_filter(watermarked_img, ksize=3)
    _, nc_filter = extract_dwt_svd(
        attacked_img_filter, img_uint8, u_wm, v_wm.T, CH, CV, CD, WaveletName, SF, 
        pcaoutput, eigenvec1, m1
    )
    results['Median Filter (3x3)'] = nc_filter
    print(f"NC (Median Filter): {nc_filter:.4f}")

    # 4. Attack: JPEG Compression (Quality 70)
    attacked_img_jpeg = AttackSuite.jpeg_compression(watermarked_img, quality=70)
    _, nc_jpeg = extract_dwt_svd(
        attacked_img_jpeg, img_uint8, u_wm, v_wm.T, CH, CV, CD, WaveletName, SF, 
        pcaoutput, eigenvec1, m1
    )
    results['JPEG Compression (Q=70)'] = nc_jpeg
    print(f"NC (JPEG Q=70): {nc_jpeg:.4f}")

    # --- Display Extracted Watermark from the toughest attack (JPEG Q=70) ---
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(attacked_img_jpeg), plt.title('Attacked Image (JPEG Q=70)')
    
    plt.subplot(1, 2, 2)
    plt.imshow(Watermark, cmap='gray'), plt.title(f'Original Watermark')

    # Re-extract the watermark for the final plot display
    ExtractedWatermark_final, nc_final = extract_dwt_svd(
        attacked_img_jpeg, img_uint8, u_wm, v_wm.T, CH, CV, CD, WaveletName, SF, 
        pcaoutput, eigenvec1, m1
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(ExtractedWatermark_final, cmap='gray')
    plt.title(f'Extracted Watermark (JPEG Q=70, NC: {nc_final:.4f})')
    plt.show()

