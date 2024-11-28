import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

def yeni_filter(image, alpha=2):
    # Initialize forward and backward filters
    forward = np.zeros_like(image, dtype=np.float32)
    backward = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape

    # Forward filter 
    for m in range(height):
        for n in range(1, width):
            edge_signal = abs(forward[m, n - 1] - image[m, n])
            A = 1 - (edge_signal / 255.0) ** alpha
            A = max(0, min(A, 1))  
            forward[m, n] = A * forward[m, n - 1] + (1 - A) * image[m, n]

    # Backward filter 
    for m in range(height):
        for n in range(width - 2, -1, -1):
            edge_signal = abs(backward[m, n + 1] - image[m, n])
            A = 1 - (edge_signal / 255.0) ** alpha
            A = max(0, min(A, 1))  
            backward[m, n] = A * backward[m, n + 1] + (1 - A) * image[m, n]

    # local mean
    local_mean = (forward + backward) / 2.0

    return local_mean

def enhance_contrast(image, local_mean, gain=1):
    # Compute high-pass filter response
    detail = image - local_mean
    enhanced = image + gain * detail
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def adaptive_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def unsharp_masking(image, strength=1.5, blur_size=5):
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    detail = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return np.clip(detail, 0, 255).astype(np.uint8)

def calculate_metrics(original, enhanced):
    # Calculate PSNR
    psnr = cv2.PSNR(original, enhanced)
    # Calculate SSIM
    ssim_value, _ = ssim(original, enhanced, full=True)
    return psnr, ssim_value

def compare_methods(input_image_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    # YENI Filter
    local_mean = yeni_filter(input_image, alpha=5)  
    yeni_image = enhance_contrast(input_image, local_mean, gain=0.7)
    # Adaptive Histogram Equalization
    ahe_image = adaptive_histogram_equalization(input_image)
    # Unsharp Masking
    um_image = unsharp_masking(input_image)
    # Metrics
    methods = {"YENI": yeni_image, "AHE": ahe_image, "Unsharp Masking": um_image}
    metrics = {}
    for method, img in methods.items():
        psnr, ssim_value = calculate_metrics(input_image, img)
        metrics[method] = {"PSNR": psnr, "SSIM": ssim_value}
    
    print("Metrics Comparison:")
    for method, metric in metrics.items():
        print(f"{method} - PSNR: {metric['PSNR']:.2f}, SSIM: {metric['SSIM']:.4f}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(input_image, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(yeni_image, cmap="gray")
    plt.title("YENI Filter")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(ahe_image, cmap="gray")
    plt.title("AHE (CLAHE)")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(um_image, cmap="gray")
    plt.title("Unsharp Masking")
    plt.axis("off")
    
    plt.show()
    




compare_methods("images/test1.jpg")

