import cv2
import numpy as np

class Preprocessor:
    def __init__(self, config=None):
        self.config = config if config else {}
        
    def to_grayscale(self, img):
        """Converts BGR image to Grayscale if not already single-channel."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    def denoise(self, img, method='bilateral', d=9, sigma_color=75, sigma_space=75):
        """Applies edge-preserving noise reduction."""
        if method == 'bilateral':
            # Bilateral filter is excellent because it preserves sharp pupil/limbus edges
            return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        elif method == 'gaussian':
            return cv2.GaussianBlur(img, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(img, 5)
        else:
            return img.copy()

    def enhance_contrast(self, img, method='clahe', clip_limit=3.0, tile_grid_size=(8, 8)):
        """Enhances image contrast using CLAHE or Global Histogram Equalization."""
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(img)
        elif method == 'equalize':
            return cv2.equalizeHist(img)
        else:
            return img.copy()

    def suppress_reflections(self, img, threshold=245, dilation_kernel_size=5):
        """Detects bright specular reflections and inpaints them to avoid false edges."""
        # Specular reflections are close to white (intensity near 255)
        _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(mask) == 0:
            return img.copy(), mask
            
        # Dilate mask slightly to cover the transition borders of reflection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        dilated_mask = cv2.dilate(mask, kernel)
        
        # Inpaint using Telea's algorithm (fast and effective for small reflection spots)
        inpainted = cv2.inpaint(img, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted, dilated_mask

    def process(self, img_bgr, denoise_method='bilateral', enhance_method='clahe', suppress_refl=True):
        """Executes full preprocessing pipeline on BGR image."""
        # 1. Convert to grayscale
        gray = self.to_grayscale(img_bgr)
        
        # 2. Specular reflection suppression
        refl_mask = np.zeros_like(gray)
        if suppress_refl:
            gray, refl_mask = self.suppress_reflections(gray)
            
        # 3. Noise reduction
        denoised = self.denoise(gray, method=denoise_method)
        
        # 4. Contrast enhancement
        enhanced = self.enhance_contrast(denoised, method=enhance_method)
        
        return {
            "grayscale": gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "refl_mask": refl_mask
        }

if __name__ == "__main__":
    # Quick self-test
    dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # Add a white spot for reflection
    dummy_img[200:210, 200:210, :] = 255
    
    preprocessor = Preprocessor()
    results = preprocessor.process(dummy_img, suppress_refl=True)
    print("Preprocessor self-test passed!")
    print("Grayscale shape:", results["grayscale"].shape)
    print("Reflection mask non-zero pixels:", cv2.countNonZero(results["refl_mask"]))
