import cv2
import numpy as np
from skimage import measure
import os

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.mask = None
        self.labels = None
        self.results = []
        self.current_file_path = None
        self.current_filename = None
        self.sensitivity_used = 50
        self.microns_per_pixel = 1.0 # Default: 1px = 1 unit
        self.unit_name = "px" # Default unit
        
        # Track manual divisors for export
        self.manual_divisors = [] 
        
        # Multi-channel support
        self.channels = [] # List of numpy arrays (H, W) or (H, W, 3)
        self.channel_names = [] # List of strings
        self.current_channel_index = 0
        self.channel_metadata = {}
        self.manual_divisors = [] # Reset manual divisors for new image # {index: {'name': str, 'color': str}}

    def load_image(self, file_path):
        """Loads an image from a file path, supporting CZI and multi-channel TIFF."""
        from tifffile import TiffFile
        
        self.current_file_path = file_path
        self.current_filename = os.path.basename(file_path)
        self.channels = []
        self.channel_names = []
        self.current_channel_index = 0
        self.channel_metadata = {}
        self.manual_divisors = []
        
        if file_path.lower().endswith('.czi'):
            try:
                import czifile
                with czifile.CziFile(file_path) as czi:
                    # CZI data is usually (Time, Scene, Channel, Z, Height, Width, 1)
                    data = czi.asarray()
                    
                    # Try to extract channel names from metadata
                    try:
                        import xml.etree.ElementTree as ET
                        xml_metadata = czi.metadata()
                        root = ET.fromstring(xml_metadata)
                        # Zeiss CZI metadata path for channel names
                        # We look for the display channel names which usually correspond to the data
                        channel_elements = root.findall(".//Channels/Channel")
                        found_names = []
                        for channel in channel_elements:
                            name = channel.get("Name")
                            if name and name not in found_names:
                                found_names.append(name)
                        
                        # Only take as many names as there are data channels later
                        # But we'll store them for now
                        self.channel_names = found_names
                    except Exception as meta_err:
                        print(f"Error reading CZI metadata: {meta_err}")
                    
                    # Squeeze unnecessary dimensions
                    # We want (Channel, Height, Width)
                    data = np.squeeze(data)
                    
                    if data.ndim == 2: # Grayscale
                        self.channels = [data]
                        # If we found names, use the first one, else default
                        if self.channel_names:
                            self.channel_names = [self.channel_names[0]]
                        else:
                            self.channel_names = ["Grayscale"]
                    elif data.ndim == 3:
                        # Could be (C, H, W) or (H, W, C)
                        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                            # (C, H, W)
                            for i in range(data.shape[0]):
                                self.channels.append(data[i])
                        else:
                            # (H, W, C)
                            for i in range(data.shape[2]):
                                self.channels.append(data[:, :, i])
                    
                    # Ensure channel_names length matches channels length
                    if len(self.channel_names) != len(self.channels):
                        if len(self.channel_names) > len(self.channels):
                            # Too many names found in metadata (duplicates or display-only channels)
                            self.channel_names = self.channel_names[:len(self.channels)]
                        else:
                            # Too few names, pad with generic ones
                            for i in range(len(self.channel_names), len(self.channels)):
                                self.channel_names.append(f"Channel {i+1}")
                    
                    # Set the composite or first channel as original_image
                    self._prepare_initial_image()
                    return True
            except Exception as e:
                print(f"Error loading CZI: {e}")
                return False
        
        elif file_path.lower().endswith(('.tif', '.tiff')):
            try:
                with TiffFile(file_path) as tif:
                    data = tif.asarray()
                    # Multi-page or multi-channel TIFF
                    if data.ndim == 3:
                        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                            for i in range(data.shape[0]):
                                self.channels.append(data[i])
                        else:
                            for i in range(data.shape[2]):
                                self.channels.append(data[:, :, i])
                    elif data.ndim == 2:
                        self.channels = [data]
                    
                    if not self.channel_names:
                        self.channel_names = [f"Channel {i+1}" for i in range(len(self.channels))]
                    
                    self._prepare_initial_image()
                    return True
            except Exception as e:
                print(f"Error loading TIFF with tifffile: {e}")
                # Fallback to cv2
        
        # Standard format or fallback
        self.original_image = cv2.imread(file_path)
        if self.original_image is not None:
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.channels = [self.original_image]
            self.channel_names = ["Composite"]
            return True
        return False

    def _prepare_initial_image(self):
        """Converts the raw channel data to 8-bit RGB for display."""
        if not self.channels:
            return
            
        # If we have multiple channels, create a composite for the initial view
        # or just show the first one if it's grayscale.
        # For now, let's just use the first channel as the default 'original_image'
        # but normalize it to 8-bit.
        self.switch_to_channel(0)

    def switch_to_channel(self, index):
        """Switches the active display/analysis image to a specific channel."""
        if index < 0 or index >= len(self.channels):
            return False
            
        self.current_channel_index = index
        data = self.channels[index]
        
        # Normalize to 8-bit
        if data.dtype != np.uint8:
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
        
        # If 2D (grayscale), convert to RGB for consistency in the app
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            
        self.original_image = data
        return True

    def segment_adipocytes(self, sensitivity=0.5, enhance_contrast=True):
        """
        Automated segmentation of adipocytes.
        sensitivity: float between 0 and 1, adjusting the thresholding.
        enhance_contrast: bool, whether to use CLAHE to improve contrast.
        """
        if self.original_image is None:
            return None

        # 1. Grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        # 2. Optional: Enhance contrast using CLAHE
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # 3. Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. Thresholding
        # Adipocytes have light interiors. THRESH_BINARY makes light areas 255 (foreground)
        block_size = 31
        
        # REMAPPING SENSITIVITY:
        # Input sensitivity is 0.0 - 1.0 (from slider 0-100)
        # We remap 0.0 -> 0.5 (original midpoint) to 1.0 -> 1.0 (original max)
        # This focuses the slider on the useful 50-100 range from before.
        remapped_sens = 0.5 + (sensitivity * 0.5)
        
        # c_val: 0.5 -> C=0, 1.0 -> C=20
        c_val = int((remapped_sens - 0.5) * 40)
        
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, block_size, c_val)

        # 5. Morphological cleaning
        kernel = np.ones((3, 3), np.uint8)
        # Fill small holes in the cell interiors and remove small noise
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # 6. Distance transform and Watershed to separate touching cells
        dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        # Higher threshold here makes it more likely to split cells
        _, last_sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
        
        last_sure_fg = np.uint8(last_sure_fg)
        unknown = cv2.subtract(opened, last_sure_fg)

        # marker labeling
        # Optimization: Use 4-connectivity for connected components
        _, markers = cv2.connectedComponents(last_sure_fg, connectivity=4)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply Watershed
        # Optimization: Use a smaller border value if possible
        self.labels = cv2.watershed(self.original_image, markers)
        
        # Create a mask where cells are > 1 (Watershed labels: -1=border, 1=bg, >1=cells)
        self.mask = np.zeros_like(gray, dtype=np.uint8)
        self.mask[self.labels > 1] = 255
        
        # Unify labels: 0=bg, >0=cells
        self.labels[self.labels <= 1] = 0
        self.labels = self.labels.astype(np.int32)
        
        return self.mask

    def get_analysis_results(self):
        """
        Calculates number and area of segmented cells and generates a labeled image.
        Uses self.microns_per_pixel for SI conversion.
        """
        if self.labels is None:
            return [], None

        # Optimization: Use memory-efficient property calculation
        props = measure.regionprops(self.labels)
        self.results = []
        
        # 1. Start with the original image
        labeled_image = self.original_image.copy()
        
        # 2. Overlay the Magenta Mask (255, 0, 255)
        if self.mask is not None:
            mask_rgb = np.zeros_like(labeled_image)
            mask_rgb[self.mask > 0] = [255, 0, 255] # Magenta
            # Blend with 0.3 alpha for export (similar to UI 100/255)
            labeled_image = cv2.addWeighted(labeled_image, 1.0, mask_rgb, 0.3, 0)

        # 3. Draw Manual Divisors (Yellow)
        for pts in self.manual_divisors:
            pts_array = np.array(pts, dtype=np.int32)
            cv2.polylines(labeled_image, [pts_array], isClosed=False, color=(255, 255, 0), thickness=2)

        # 4. Draw Cell IDs and calculate results
        display_id = 1
        for prop in props:
            # Skip background (label 0)
            if prop.label == 0:
                continue
            
            # Filter out very small artifacts
            if prop.area < 100:
                continue

            # Area in physical units squared (e.g. µm²)
            area = prop.area * (self.microns_per_pixel ** 2)
            y0, x0 = prop.centroid
            perimeter = prop.perimeter * self.microns_per_pixel
            
            self.results.append({
                    'label': display_id,
                    'area': area,
                    'centroid': (x0, y0),
                    'perimeter': perimeter
                })
            
            # Draw label on image with yellow color and black outline for high contrast
            text = str(display_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text_color = (0, 255, 255) # Bright Yellow (BGR for OpenCV: (0, 255, 255) is Yellow)
            outline_color = (0, 0, 0) # Black
            thickness = 1
            outline_thickness = 2
            
            # Get text size for centering
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = int(x0 - text_width / 2)
            text_y = int(y0 + text_height / 2)
            
            # Draw black outline first
            cv2.putText(labeled_image, text, (text_x, text_y), font, font_scale, outline_color, outline_thickness)
            # Draw yellow text on top
            cv2.putText(labeled_image, text, (text_x, text_y), font, font_scale, text_color, thickness)
            
            display_id += 1
        
        return self.results, labeled_image

    def apply_manual_border(self, path_points):
        """
        Allows drawing a free-hand border.
        If the path is closed (start and end are close), it fills the interior.
        Otherwise, it acts as a divisor.
        """
        if self.mask is None or len(path_points) < 2:
            return None
        
        # Convert path to numpy array
        pts = np.array(path_points, dtype=np.int32)
        
        # Check if path is closed (distance between start and end < 10 pixels)
        start = pts[0]
        end = pts[-1]
        dist = np.sqrt(np.sum((start - end)**2))
        
        if dist < 10 and len(pts) > 5:
            # Closed path: Trace cell
            # 1. Create a temporary mask for the interior
            interior_mask = np.zeros_like(self.mask, dtype=np.uint8)
            cv2.fillPoly(interior_mask, [pts], 255)
            
            # 2. To ensure it's a separate cell, we must create a 1-pixel gap 
            # between this new area and any existing cells.
            cv2.polylines(self.mask, [pts], isClosed=True, color=0, thickness=1)
            
            # 3. Add the interior (now separated by the gap) to the mask
            self.mask[interior_mask > 0] = 255
        else:
            # Open path: Draw divisor
            # Draw the path as a black line (0) to separate regions
            cv2.polylines(self.mask, [pts], isClosed=False, color=0, thickness=2)
            # Store for visual overlay on export
            self.manual_divisors.append(list(path_points))
        
        # Optimization: Use connectivity=1 for labeling (4-connectivity) which is faster
        self.labels = measure.label(self.mask > 0, connectivity=1)
        self.labels = self.labels.astype(np.int32)
        
        return self.mask

    def delete_cell(self, x, y):
        """
        Deletes a cell at (x, y) and merges its area with neighbors by 
        removing the boundary in the mask.
        """
        if self.labels is None or self.mask is None:
            return None

        # Check coordinates bounds
        h, w = self.labels.shape
        if not (0 <= x < w and 0 <= y < h):
            return None

        target_label = self.labels[y, x]
        if target_label == 0:
            return None

        # 1. Find all pixels of the target cell
        cell_mask = (self.labels == target_label).astype(np.uint8)
        
        # 2. To merge, we fill the cell's area AND its immediate boundary (0s in mask)
        # We dilate by 1 pixel and only fill where the mask was 0.
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(cell_mask, kernel, iterations=1)
        boundary_to_fill = (dilated > 0) & (self.mask == 0)
        
        self.mask[cell_mask > 0] = 255
        self.mask[boundary_to_fill] = 255
        
        # 3. Re-run labeling
        self.labels = measure.label(self.mask > 0, connectivity=1)
        self.labels = self.labels.astype(np.int32)
        
        return self.mask

    def void_area(self, x, y):
        """
        Deletes a cell at (x, y) and removes it from results completely (background).
        """
        if self.labels is None or self.mask is None:
            return None

        # Check coordinates bounds
        h, w = self.labels.shape
        if not (0 <= x < w and 0 <= y < h):
            return None

        target_label = self.labels[y, x]
        if target_label == 0:
            return None

        # 1. Find all pixels of the target cell
        cell_mask = (self.labels == target_label).astype(np.uint8)
        
        # 2. Set this area to background (0) in both mask and labels
        self.mask[cell_mask > 0] = 0
        self.labels[cell_mask > 0] = 0
        
        # 3. Re-run labeling to ensure ID sequence is consistent if needed
        # (Though get_analysis_results handles sequential IDs anyway)
        self.labels = measure.label(self.mask > 0, connectivity=1)
        self.labels = self.labels.astype(np.int32)
        
        return self.mask

    def add_cell(self, x, y):
        """
        Tries to find a region around (x, y) that is currently background (0) 
        and turn it into a cell using flood fill on the processed image.
        Returns (new_mask, success_message, error_message).
        """
        if self.original_image is None or self.labels is None:
            return None, None, "No image analyzed."

        h, w = self.labels.shape
        if not (0 <= x < w and 0 <= y < h):
            return None, None, "Click outside image bounds."

        # If already a cell, do nothing
        if self.labels[y, x] > 0:
            return None, None, "Area is already part of a cell."

        # We use the processed image (thresholded/cleaned) to find the region
        # Re-run thresholding locally or use the existing thresh/opened logic
        # For simplicity and consistency, let's use the same preprocessing as segment_adipocytes
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        sensitivity = 0.6 + (self.sensitivity_used / 100.0) * 0.4
        block_size = 31
        c_val = int((sensitivity - 0.5) * 40)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, block_size, c_val)
        
        # Check if the click is on a "foreground" pixel in the thresholded image
        # If it's on a membrane (0 in thresh), we can't flood fill a cell
        if thresh[y, x] == 0:
            return None, None, "Click is on a cell membrane or dark region. Try clicking the light interior of the cell."

        # Flood fill to find the connected component in thresh
        temp_mask = np.zeros((h + 2, w + 2), np.uint8)
        # We want to fill the '255' region starting at (x, y)
        flood_mask = thresh.copy()
        cv2.floodFill(flood_mask, temp_mask, (x, y), 127) # Use 127 as marker
        
        # The filled area is where flood_mask == 127
        new_cell_area = (flood_mask == 127)
        
        # Check if this new area overlaps significantly with existing cells
        overlap = np.any((self.labels > 0) & new_cell_area)
        if overlap:
            # If it overlaps, it means the thresholded region is already partially counted.
            # This happens if a divisor line was drawn but didn't fully split, or if 
            # the watershed didn't reach this part.
            # We'll allow it but warning.
            pass

        # Limit the size of manually added cells to prevent filling the whole background
        area_px = np.sum(new_cell_area)
        if area_px > (h * w * 0.1): # More than 10% of image
            return None, None, "The detected region is too large (likely not a single cell). Adjust sensitivity or click a more defined area."
        
        if area_px < 50:
            return None, None, "The detected region is too small to be a cell."

        # Apply the new cell to the mask
        self.mask[new_cell_area] = 255
        
        # Re-run labeling
        self.labels = measure.label(self.mask > 0, connectivity=1)
        self.labels = self.labels.astype(np.int32)
        
        return self.mask, f"Added cell with area {area_px} px.", None
