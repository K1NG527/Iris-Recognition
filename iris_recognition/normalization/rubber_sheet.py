import cv2
import numpy as np

class RubberSheetNormalizer:
    """Daugman's Rubber-Sheet Model for Iris Normalization.
    
    Transforms the circular annular iris region from Cartesian coordinates (x, y)
    to a dimensionless pseudo-polar coordinate system (r, theta), where:
      - r in [0, 1] is the radial dimension (r=0 at pupil boundary, r=1 at limbus boundary)
      - theta in [0, 2*pi] is the angular dimension around the pupil center
    
    Mathematical Formulation (Class Notes Slide 98-99):
      I(x(r, theta), y(r, theta)) -> I(r, theta)
      
      where:
        x(r, theta) = (1 - r) * xp(theta) + r * xl(theta)
        y(r, theta) = (1 - r) * yp(theta) + r * yl(theta)
        
        xp(theta) = xp0 + rp * cos(theta)
        yp(theta) = yp0 + rp * sin(theta)
        xl(theta) = xl0 + rl * cos(theta)
        yl(theta) = yl0 + rl * sin(theta)
        
      (xp0, yp0, rp) = pupillary boundary circle center and radius
      (xl0, yl0, rl) = limbus boundary circle center and radius
    """

    def __init__(self, config=None):
        self.config = config if config else {}
        norm_config = self.config.get("normalization", {})
        self.width = norm_config.get("width", 512)    # Angular resolution (theta samples)
        self.height = norm_config.get("height", 64)   # Radial resolution (r samples)
        
    def normalize(self, img_gray, xp, yp, rp, xi, yi, ri, mask):
        """Maps circular iris region to a rectangular polar representation of size height x width.
        
        Parameters:
        - img_gray: 2D grayscale image
        - xp, yp, rp: pupil circle center (xp, yp) and radius rp (pupillary boundary)
        - xi, yi, ri: limbus circle center (xi, yi) and radius ri (limbus boundary)
        - mask: binary iris mask (255 for valid iris, 0 for occlusions/eyelids)
        
        Returns:
        - normalized_iris (height x width, uint8): unwrapped iris texture strip
        - normalized_mask (height x width, uint8): unwrapped binary mask strip
        """
        # Create r and theta grids
        # r goes from 0 (pupil boundary) to 1 (limbus boundary)
        # theta goes from 0 to 2*pi
        r = np.linspace(0, 1, self.height, dtype=np.float32)
        theta = np.linspace(0, 2 * np.pi, self.width, endpoint=False, dtype=np.float32)
        
        # 2D Meshgrid
        r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
        
        # Precompute trigonometric terms for all angles
        cos_t = np.cos(theta_grid)
        sin_t = np.sin(theta_grid)
        
        # Coordinates on pupil boundary for each theta
        xp_t = xp + rp * cos_t
        yp_t = yp + rp * sin_t
        
        # Coordinates on limbus boundary for each theta
        xi_t = xi + ri * cos_t
        yi_t = yi + ri * sin_t
        
        # Linear interpolation between pupillary and limbus boundaries
        # x(r, theta) = (1 - r) * xp(theta) + r * xl(theta)
        # y(r, theta) = (1 - r) * yp(theta) + r * yl(theta)
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
