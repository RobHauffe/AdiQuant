# AdiQuant

**Automated histological analysis of adipose tissue for adipocyte number and size determination.**

AdiQuant is a Python-based application designed for the analysis of histological sections of mammalian tissues, specifically focusing on the determination of adipocyte number and size (area). It features an intuitive user interface and supports various image formats including TIF, BMP, JPG, and CZI.

## Features

- **Automated Analysis:** Calculates adipocyte count, area, and size distribution.
- **Broad File Support:** Supports TIF, BMP, JPG, PNG, and Zeiss CZI formats.
- **Manual Correction Tools:**
    - **Draw Border:** Trace cells or draw divisors to split merged cells.
    - **Add Cell:** Click to add uncounted regions (with flood fill).
    - **Void/Delete:** Remove artifacts or incorrect detections.
    - **Undo:** Supports undoing the last 10 manual actions (Ctrl+Z).
- **Scale Calibration:** Interactive tool to define pixel-to-micron ratio using a scale bar.
- **Data Export:** Export results to Excel, copy to clipboard, or save analyzed images with overlays.
- **Visual Feedback:** High-contrast overlays for counted areas, numbered labels, and interactive sensitivity adjustment.

## Methods

AdiQuant employs a multi-step image processing pipeline to identify and quantify adipocytes:

### 1. Preprocessing
The application first converts the input image to grayscale. To account for variations in lighting and staining intensity, **Contrast Limited Adaptive Histogram Equalization (CLAHE)** is applied (clip limit: 2.0, tile grid: 8x8). This is followed by a **Gaussian Blur** (5x5 kernel) to reduce high-frequency noise while preserving cellular boundaries.

### 2. Segmentation
Segmentation is performed using an **Adaptive Gaussian Thresholding** approach. This method calculates thresholds locally for each pixel based on a 31x31 pixel neighborhood, making it robust against uneven background illumination. The sensitivity parameter adjusted by the user directly influences the constant subtracted from the local mean.

### 3. Post-processing and Separation
To refine the initial segmentation:
- **Morphological Operations:** A sequence of closing and opening operations (3x3 kernel) is used to fill small gaps within cell interiors and remove minor noise artifacts.
- **Watershed Transformation:** To separate touching or overlapping adipocytes, the application calculates a **Euclidean Distance Transform** of the binary mask. Local maxima in the distance map serve as markers for the **Watershed algorithm**, which determines the final boundaries (membranes) between adjacent cells.

### 4. Quantification
Once segmented, individual cells are analyzed using the `regionprops` algorithm:
- **Area:** Determined by the total number of pixels within a segmented region, converted to square micrometers ($\mu m^2$) using the user-defined scale calibration.
- **Perimeter:** Calculated by approximating the boundary length of each region.
- **Filtering:** To minimize false positives, objects with an area smaller than 100 pixels (before scaling) are automatically excluded as potential artifacts.

### 5. Manual Refinement
The software provides tools for manual verification and correction. Users can draw free-hand borders to split merged cells (divisors) or trace missing ones. The "Add Cell" tool utilizes a **Flood Fill** algorithm on the preprocessed image to identify connected regions based on local intensity, ensuring that manual additions remain consistent with the automated detection logic.

## Citation

Users are requested to cite **Hauffe et al. 2026 (Placeholder, i will update this once the paper is published)** when using AdiQuant.

## License
This project is licensed under the MIT License – see the LICENSE file for details.

## Credits

**Author:** Dr. Robert Hauffe

**Affiliation:**
[Molecular and Experimental Nutritional Medicine](https://www.uni-potsdam.de/de/mem/index)
University of Potsdam, Germany

## Build Instructions

To build the standalone executable from source, follow these steps:

1. **Install Dependencies:**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run PyInstaller:**
   Use the following command to create the single-file executable (ensure `FunAdipocyte.ico` is in the project root):
   ```bash
   pyinstaller --noconfirm --onefile --windowed --icon "FunAdipocyte.ico" --add-data "FunAdipocyte.ico;." --name "AdiQuant" --hidden-import "scipy.spatial.transform._rotation_groups" main.py
   ```

## Current Build Details (2026-02-09)
- **Version:** 1.0.0
- **SHA-256 Checksum:** `172E686078F285A827BA94A0928F3C9D628FA2799CF48C505DC649E1AEFCDDF3`

To verify the integrity of your `AdiQuant.exe` (located in the `dist` folder), you can run the following command in PowerShell:
```powershell
Get-FileHash -Path "dist\AdiQuant.exe" -Algorithm SHA256 | Format-List
```
Compare the output hash with the one above.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. This product is intended for academic use.

---
*Note: This tool is developed by the Molecular and Experimental Nutritional Medicine group at the University of Potsdam.*
