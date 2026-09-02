import cv2
import numpy as np

class RubberSheetNormalizer:
    def __init__(self, config=None):
        self.config = config if config else {}
        norm_config = self.config.get("normalization", {})
        self.width = norm_config.get("width", 512)
        self.height = norm_config.get("height", 64)
        
    def normalize(self, img_gray, xp, yp, rp, xi, yi, ri, mask):
        """Maps circular iris region to a rectangular polar representation of size height x width.
        
        Parameters:
        - img_gray: 2D grayscale image
        - xp, yp, rp: pupil circle center and radius
        - xi, yi, ri: limbus circle center and radius
        - mask: binary iris mask (0 and 255)
        
        Returns:
        - normalized_iris (height x width, uint8)
        - normalized_mask (height x width, uint8)
        """
        # Create r and theta grids
        # r goes from 0 (pupil boundary) to 1 (limbus boundary)
        # theta goes from 0 to 2*pi
        r = np.linspace(0, 1, self.height, dtype=np.float32)
        theta = np.linspace(0, 2 * np.pi, self.width, dtype=np.float32)
        
        # Meshgrid
        r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
        
        # Precompute trigonometric terms
        cos_t = np.cos(theta_grid)
        sin_t = np.sin(theta_grid)
        
        # Coordinates on pupil boundary for each theta
        xp_t = xp + rp * cos_t
        yp_t = yp + rp * sin_t
        
        # Coordinates on limbus boundary for each theta
        xi_t = xi + ri * cos_t
        yi_t = yi + ri * sin_t
        
        # Interpolate between pupil and limbus boundaries
        map_x = (1 - r_grid) * xp_t + r_grid * xi_t
        map_y = (1 - r_grid) * yp_t + r_grid * yi_t
        
        # Cast to float32 for cv2.remap
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Remap grayscale image (using bilinear interpolation)
        normalized_iris = cv2.remap(img_gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Remap mask (using nearest-neighbor to keep it binary)
        normalized_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Ensure mask is strictly binary (0 or 255)
        _, normalized_mask = cv2.threshold(normalized_mask, 127, 255, cv2.THRESH_BINARY)
        
        return normalized_iris, normalized_mask

if __name__ == "__main__":
    # Self test
    test_img = np.ones((480, 640), dtype=np.uint8) * 150
    # Draw simple gradient iris pattern
    for r in range(40, 110):
        cv2.circle(test_img, (320, 240), r, int(100 + (r-40)*1.5), 1)
        
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask, (320, 240), 110, 255, -1)
    cv2.circle(mask, (320, 240), 40, 0, -1)
    
    normalizer = RubberSheetNormalizer()
    norm_img, norm_mask = normalizer.normalize(test_img, 320.0, 240.0, 40.0, 320.0, 240.0, 110.0, mask)
    print("Rubber-sheet Normalizer test:")
    print("Normalized image shape:", norm_img.shape)
    print("Normalized mask shape:", norm_mask.shape)
    print("Mask non-zero fraction:", cv2.countNonZero(norm_mask) / norm_mask.size)
