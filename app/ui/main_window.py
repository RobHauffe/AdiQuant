from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QGroupBox, QSlider,
                             QToolBar, QStatusBar, QMessageBox, QApplication,
                             QSpinBox, QDoubleSpinBox, QInputDialog, QDialog,
                             QCheckBox, QScrollArea, QFrame, QLineEdit, QSplitter,
                             QToolTip, QTabWidget, QFormLayout)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QShortcut, QKeySequence
import pandas as pd
from datetime import datetime
import os
import cv2
from .canvas import ImageCanvas
from ..core.image_processor import ImageProcessor

class HelpLabel(QLabel):
    """Custom label for help icons that shows tooltips immediately on hover."""
    def __init__(self, text, parent=None):
        super().__init__("?", parent)
        self.help_text = text
        self.setFixedSize(18, 18)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #bdc3c7;
                color: white;
                border-radius: 9px;
                font-weight: bold;
                font-size: 12px;
            }
            QLabel:hover {
                background-color: #7f8c8d;
            }
        """)

    def enterEvent(self, event):
        # Show tooltip immediately at the label's position
        pos = self.mapToGlobal(QPoint(self.width(), 0))
        QToolTip.showText(pos, self.help_text, self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AdiQuant - Histological Image Analysis")
        
        # Version Information
        self.version = "1.0.0"
        self.creation_date = "2026-02-09"
        self.author = "Dr. Robert Hauffe"
        self.processor = ImageProcessor()
        self.undo_stack = []
        self.max_undo = 10
        
        # UI Setup
        self.setup_ui()
        self.setup_connections()
        self.setup_shortcuts()
        
        # Show maximized at the end to prevent sizing flicker
        self.showMaximized()

    def setup_ui(self):
        self.apply_styles()
        self.help_widgets = {} # Store help buttons to toggle visibility
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # Left Side: Filename and Canvas
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        self.lbl_filename = QLabel("No image loaded")
        self.lbl_filename.setObjectName("lbl_filename")
        self.lbl_filename.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50; padding: 5px;")
        self.lbl_filename.setAlignment(Qt.AlignmentFlag.AlignLeft)
        left_layout.addWidget(self.lbl_filename)
        
        self.canvas = ImageCanvas()
        left_layout.addWidget(self.canvas)
        
        self.splitter.addWidget(left_panel)

        # Right Side: Controls and Results (with ScrollArea)
        right_panel = QWidget()
        right_panel.setMinimumWidth(380)
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        right_layout = QVBoxLayout(scroll_content)
        right_layout.setContentsMargins(5, 5, 5, 5)
        scroll_area.setWidget(scroll_content)
        
        right_panel_layout.addWidget(scroll_area)
        self.splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (Canvas: 75%, Controls: 25%)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)

        # 1. File Group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        file_btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open Image")
        self.btn_open.setObjectName("btn_open")
        file_btn_layout.addWidget(self.btn_open)
        file_btn_layout.addStretch()
        file_btn_layout.addWidget(self.create_help_label("Open a histology image (TIF, JPG, BMP, etc.) to begin analysis."))
        file_layout.addLayout(file_btn_layout)
        
        right_layout.addWidget(file_group)

        # 2. Analysis Group
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QVBoxLayout(analysis_group)
        
        sens_label_layout = QHBoxLayout()
        sens_label_layout.addWidget(QLabel("Sensitivity (1-100):"))
        sens_label_layout.addStretch()
        sens_label_layout.addWidget(self.create_help_label("Adjust how strictly cells are identified. Higher = stricter/smaller cells, Lower = looser/larger cells."))
        analysis_layout.addLayout(sens_label_layout)

        sens_hbox = QHBoxLayout()
        self.slider_sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.slider_sensitivity.setRange(1, 100)
        self.slider_sensitivity.setValue(50)
        
        self.spin_sensitivity = QSpinBox()
        self.spin_sensitivity.setRange(1, 100)
        self.spin_sensitivity.setValue(50)
        self.spin_sensitivity.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_sensitivity.setFixedWidth(60)
        
        sens_hbox.addWidget(self.slider_sensitivity)
        sens_hbox.addWidget(self.spin_sensitivity)
        analysis_layout.addLayout(sens_hbox)

        clahe_hbox = QHBoxLayout()
        self.chk_enhance = QCheckBox("Enhance Contrast (CLAHE)")
        self.chk_enhance.setChecked(True)
        clahe_hbox.addWidget(self.chk_enhance)
        clahe_hbox.addStretch()
        clahe_hbox.addWidget(self.create_help_label("Use CLAHE to improve contrast between cell membranes and interiors."))
        analysis_layout.addLayout(clahe_hbox)
        
        analyze_btn_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("Run Analysis")
        self.btn_analyze.setObjectName("btn_analyze")
        analyze_btn_layout.addWidget(self.btn_analyze)
        analyze_btn_layout.addStretch()
        analyze_btn_layout.addWidget(self.create_help_label("Run the automated adipocyte detection based on current sensitivity."))
        analysis_layout.addLayout(analyze_btn_layout)
        
        right_layout.addWidget(analysis_group)
        self.analysis_group = analysis_group # Store for toggling

        # 3. Scale Calibration Group
        scale_group = QGroupBox("Scale and Calibration")
        scale_layout = QVBoxLayout(scale_group)
        
        scale_input_hbox = QHBoxLayout()
        scale_input_hbox.addWidget(QLabel("Scale (µm/px):"))
        self.spin_pixel_size = QDoubleSpinBox()
        self.spin_pixel_size.setRange(0.0001, 1000.0)
        self.spin_pixel_size.setValue(1.0)
        self.spin_pixel_size.setDecimals(4)
        self.spin_pixel_size.setSingleStep(0.1)
        self.spin_pixel_size.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_pixel_size.setFixedWidth(80)
        scale_input_hbox.addWidget(self.spin_pixel_size)
        scale_input_hbox.addStretch()
        scale_input_hbox.addWidget(self.create_help_label("Manual scale entry. 1.0 means results will be in pixels."))
        scale_layout.addLayout(scale_input_hbox)
        
        calibrate_btn_layout = QHBoxLayout()
        self.btn_calibrate = QPushButton("Calibrate from Scale Bar")
        self.btn_calibrate.setCheckable(True)
        calibrate_btn_layout.addWidget(self.btn_calibrate)
        calibrate_btn_layout.addStretch()
        help_calibrate = self.create_help_label("Draw a line over a known distance (e.g. scale bar) to calculate µm/px automatically.")
        self.help_widgets['calibrate'] = help_calibrate
        calibrate_btn_layout.addWidget(help_calibrate)
        scale_layout.addLayout(calibrate_btn_layout)
        
        right_layout.addWidget(scale_group)

        # 4. Manual Correction Tools
        self.tools_group = QGroupBox("Manual Correction Tools")
        tools_layout = QVBoxLayout(self.tools_group)
        
        pan_layout = QHBoxLayout()
        self.btn_pan = QPushButton("Pan / Zoom")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setChecked(True)
        pan_layout.addWidget(self.btn_pan)
        pan_layout.addStretch()
        pan_layout.addWidget(self.create_help_label("Standard mode: Drag to pan, scroll wheel to zoom."))
        tools_layout.addLayout(pan_layout)
        
        draw_layout = QHBoxLayout()
        self.btn_draw = QPushButton("Draw Border")
        self.btn_draw.setCheckable(True)
        draw_layout.addWidget(self.btn_draw)
        draw_layout.addStretch()
        help_draw = self.create_help_label(
            "Draw Border Tool:\n\n"
            "1. Trace Cell: Click and draw a closed shape to manually define a cell interior.\n"
            "2. Draw Divisor: Draw a line to split two merged cells. Unlike the trace tool, "
            "leave the path open to act as a separator.\n\n"
            "The yellow lines will remain visible as a reference."
        )
        self.help_widgets['draw'] = help_draw
        draw_layout.addWidget(help_draw)
        tools_layout.addLayout(draw_layout)
        
        delete_layout = QHBoxLayout()
        self.btn_delete = QPushButton("Delete and Merge")
        self.btn_delete.setCheckable(True)
        delete_layout.addWidget(self.btn_delete)
        delete_layout.addStretch()
        help_delete = self.create_help_label("Click on a cell to delete it and merge its area into neighbors.")
        self.help_widgets['delete'] = help_delete
        delete_layout.addWidget(help_delete)
        tools_layout.addLayout(delete_layout)

        void_layout = QHBoxLayout()
        self.btn_void = QPushButton("Void / Clear Area")
        self.btn_void.setCheckable(True)
        void_layout.addWidget(self.btn_void)
        void_layout.addStretch()
        help_void = self.create_help_label("Void: Click to remove tissue artifacts.")
        self.help_widgets['void'] = help_void
        void_layout.addWidget(help_void)
        tools_layout.addLayout(void_layout)

        add_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add Cell")
        self.btn_add.setCheckable(True)
        add_layout.addWidget(self.btn_add)
        add_layout.addStretch()
        help_add = self.create_help_label("Click inside an uncounted cell to try and add it to the results manually.")
        self.help_widgets['add'] = help_add
        add_layout.addWidget(help_add)
        tools_layout.addLayout(add_layout)

        undo_layout = QHBoxLayout()
        self.btn_undo = QPushButton("Undo Action")
        self.btn_undo.setEnabled(False)
        undo_layout.addWidget(self.btn_undo)
        undo_layout.addStretch()
        help_undo = self.create_help_label("Undo the last 10 manual alterations (divisor, delete, void). Shortcut: Ctrl+Z")
        self.help_widgets['undo'] = help_undo
        undo_layout.addWidget(help_undo)
        tools_layout.addLayout(undo_layout)
        
        right_layout.addWidget(self.tools_group)

        # 5. Export Group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        save_img_layout = QHBoxLayout()
        self.btn_save_image = QPushButton("Save Analyzed Image")
        save_img_layout.addWidget(self.btn_save_image)
        save_img_layout.addStretch()
        save_img_layout.addWidget(self.create_help_label("Save the current image with all cell labels and masks as a new file."))
        export_layout.addLayout(save_img_layout)
        
        save_excel_layout = QHBoxLayout()
        self.btn_save_excel = QPushButton("Save as Excel")
        save_excel_layout.addWidget(self.btn_save_excel)
        save_excel_layout.addStretch()
        save_excel_layout.addWidget(self.create_help_label("Export all cell data (ID, Area, Perimeter) and a summary to an Excel file."))
        export_layout.addLayout(save_excel_layout)
        
        copy_clip_layout = QHBoxLayout()
        self.btn_copy_clipboard = QPushButton("Copy to Clipboard")
        copy_clip_layout.addWidget(self.btn_copy_clipboard)
        copy_clip_layout.addStretch()
        copy_clip_layout.addWidget(self.create_help_label("Copy the summary and results table to the clipboard for pasting into other apps."))
        export_layout.addLayout(copy_clip_layout)
        
        right_layout.addWidget(export_group)

        # 6. Results Group
        self.results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(self.results_group)
        
        self.lbl_count = QLabel("Cell Count: -")
        results_layout.addWidget(self.lbl_count)
        
        self.lbl_avg_area = QLabel("Avg Area: -")
        results_layout.addWidget(self.lbl_avg_area)
        
        self.lbl_avg_perimeter = QLabel("Avg Perimeter: -")
        results_layout.addWidget(self.lbl_avg_perimeter)
        
        self.lbl_total_area = QLabel("Total Area: -")
        results_layout.addWidget(self.lbl_total_area)
        
        self.lbl_sensitivity_res = QLabel("Sensitivity Setting: -")
        results_layout.addWidget(self.lbl_sensitivity_res)
        
        self.table_results = QTableWidget(0, 3)
        self.table_results.setMinimumHeight(350) # Ensure table is tall enough to be useful
        self.table_results.setHorizontalHeaderLabels(["ID", "Area", "Perimeter"])
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_results.horizontalHeader().setMinimumSectionSize(90)
        results_layout.addWidget(self.table_results)
        
        right_layout.addWidget(self.results_group)

        # 7. About Group
        about_group = QGroupBox("Information")
        about_layout = QVBoxLayout(about_group)
        self.btn_about = QPushButton("About AdiQuant")
        about_layout.addWidget(self.btn_about)
        right_layout.addWidget(about_group)

        # Toolbar
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    def setup_connections(self):
        self.btn_open.clicked.connect(self.open_image)
        self.btn_analyze.clicked.connect(self.run_analysis)
        
        self.btn_pan.clicked.connect(lambda: self.set_tool_mode("pan"))
        self.btn_draw.clicked.connect(lambda: self.set_tool_mode("draw"))
        self.btn_delete.clicked.connect(lambda: self.set_tool_mode("delete"))
        self.btn_void.clicked.connect(lambda: self.set_tool_mode("void"))
        self.btn_add.clicked.connect(lambda: self.set_tool_mode("add"))
        self.btn_calibrate.clicked.connect(lambda: self.set_tool_mode("calibrate"))
        self.btn_undo.clicked.connect(self.undo_action)
        
        self.spin_pixel_size.valueChanged.connect(self.update_scale)
        
        # Link Sliders and SpinBoxes
        self.slider_sensitivity.valueChanged.connect(self.spin_sensitivity.setValue)
        self.spin_sensitivity.valueChanged.connect(self.slider_sensitivity.setValue)
        
        self.canvas.border_drawn.connect(self.apply_border)
        self.canvas.calibration_drawn.connect(self.finish_calibration)
        self.canvas.point_clicked.connect(self.handle_canvas_click)
        self.btn_save_image.clicked.connect(self.export_image)
        self.btn_save_excel.clicked.connect(self.export_excel)
        self.btn_copy_clipboard.clicked.connect(self.copy_to_clipboard)
        self.btn_about.clicked.connect(self.show_about)

    def setup_shortcuts(self):
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_action)
        
        self.cycle_shortcut = QShortcut(QKeySequence("C"), self)
        self.cycle_shortcut.activated.connect(self.cycle_channels)

    def cycle_channels(self):
        """Cycles through available image channels."""
        if not self.processor.channels or len(self.processor.channels) <= 1:
            return
            
        next_idx = (self.processor.current_channel_index + 1) % len(self.processor.channels)
        # self.combo_channel.setCurrentIndex(next_idx) # Removed combo_channel
        self.change_channel(next_idx)

    def change_channel(self, index):
        """Handles switching between image channels."""
        if self.processor.switch_to_channel(index):
            self.canvas.set_image(self.processor.original_image)
            self.statusBar.showMessage(f"Switched to channel: {self.processor.channel_names[index]}")

    def update_scale(self, value):
        self.processor.microns_per_pixel = value
        # If the user manually changes the scale, we assume they are working in µm 
        # unless they have specifically calibrated otherwise.
        if self.processor.unit_name == "px":
            self.processor.unit_name = "µm"
        
        # Always update UI to reflect potential unit change in headers/labels
        self.update_results_table()

    def handle_canvas_click(self, x, y):
        if self.canvas.mode == "delete":
            self.delete_cell(x, y)
        elif self.canvas.mode == "void":
            self.void_area(x, y)
        elif self.canvas.mode == "add":
            self.add_cell(x, y)

    def apply_border(self, path):
        if self.processor.original_image is None:
            return
            
        # Save state for undo (includes current mask)
        self.save_for_undo()
        
        # We need to track that this undo step also added a divisor line
        self.undo_stack[-1]['added_divisor'] = True
        
        self.processor.apply_manual_border(path)
        self.canvas.add_divisor_path(path)
        self.update_results_table()

    def delete_cell(self, x, y):
        """Merges a clicked cell with its neighbors by filling the boundary."""
        if self.processor.labels is None:
            return
            
        label = self.processor.labels[int(y), int(x)]
        if label > 0:
            self.save_for_undo()
            
            import numpy as np
            # To "merge", we find the label's area and dilate it slightly
            # to bridge the watershed gap to neighboring cells.
            cell_mask = (self.processor.labels == label).astype(np.uint8) * 255
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(cell_mask, kernel, iterations=1)
            
            # Update the global mask to include the dilated area (filling the gap)
            self.processor.mask[dilated > 0] = 255
            
            # Re-label to reflect the merge
            from skimage import measure
            self.processor.labels = measure.label(self.processor.mask > 0, connectivity=1)
            self.update_results_table()
            self.statusBar.showMessage(f"Merged cell {label} with neighbors.")

    def void_area(self, x, y):
        """Completely removes a cell from the mask (sets to background)."""
        if self.processor.labels is None:
            return
            
        label = self.processor.labels[int(y), int(x)]
        if label > 0:
            self.save_for_undo()
            
            # Remove the entire labeled region from the mask
            self.processor.mask[self.processor.labels == label] = 0
            
            # Re-label to update IDs
            from skimage import measure
            self.processor.labels = measure.label(self.processor.mask > 0, connectivity=1)
            self.update_results_table()
            self.statusBar.showMessage(f"Voided cell {label}.")
        else:
            # If no label was clicked, fall back to a small circular void
            self.save_for_undo()
            cv2.circle(self.processor.mask, (int(x), int(y)), 10, 0, -1)
            from skimage import measure
            self.processor.labels = measure.label(self.processor.mask > 0, connectivity=1)
            self.update_results_table()

    def add_cell(self, x, y):
        if self.processor.original_image is None:
            return
            
        self.save_for_undo()
        # Basic seed-fill or similar to add cell
        # (Placeholder for complex addition logic)
        self.statusBar.showMessage("Manual cell addition (Simplified)")
        self.update_results_table()

    def save_for_undo(self):
        if self.processor.mask is not None:
            # Store mask AND metadata about the state
            state = {
                'mask': self.processor.mask.copy(),
                'divisors': self.processor.manual_divisors.copy(), # Store processor's divisor list
                'added_divisor': False
            }
            self.undo_stack.append(state)
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)
            self.btn_undo.setEnabled(True)

    def undo_action(self):
        if self.undo_stack:
            state = self.undo_stack.pop()
            self.processor.mask = state['mask']
            self.processor.manual_divisors = state.get('divisors', []) # Restore processor's divisor list
            
            # If the action being undone added a visual divisor, remove it from canvas
            if state.get('added_divisor'):
                self.canvas.remove_last_divisor()
            
            from skimage import measure
            self.processor.labels = measure.label(self.processor.mask > 0, connectivity=1)
            self.update_results_table()
            
            if not self.undo_stack:
                self.btn_undo.setEnabled(False)
            self.statusBar.showMessage("Undo successful")

    def run_analysis(self):
        if self.processor.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        sensitivity = self.slider_sensitivity.value() / 100.0
        enhance = self.chk_enhance.isChecked()
        
        self.statusBar.showMessage("Analyzing adipocytes...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            self.processor.segment_adipocytes(sensitivity, enhance)
            self.update_results_table()
            self.statusBar.showMessage("Analysis complete.")
        finally:
            QApplication.restoreOverrideCursor()

    def update_results_table(self):
        results, labeled_img = self.processor.get_analysis_results()
        if labeled_img is not None:
            self.canvas.set_image(labeled_img, is_analysis=True)
        self.canvas.set_mask(self.processor.mask)
        
        unit = self.processor.unit_name
        # Update table headers with units
        self.table_results.setHorizontalHeaderLabels(["ID", f"Area ({unit}²)", f"Perimeter ({unit})"])
        
        self.table_results.setRowCount(len(results))
        total_area = 0
        perimeters = []
        
        for i, res in enumerate(results):
            self.table_results.setItem(i, 0, QTableWidgetItem(str(res['label'])))
            self.table_results.setItem(i, 1, QTableWidgetItem(f"{res['area']:.2f}"))
            self.table_results.setItem(i, 2, QTableWidgetItem(f"{res['perimeter']:.2f}"))
            total_area += res['area']
            perimeters.append(res['perimeter'])
            
        count = len(results)
        self.lbl_count.setText(f"Cell Count: {count}")
        self.lbl_total_area.setText(f"Total Area: {total_area:.2f} {unit}²")
        
        if count > 0:
            avg_area = total_area / count
            avg_perim = sum(perimeters) / count
            self.lbl_avg_area.setText(f"Avg Area: {avg_area:.2f} {unit}²")
            self.lbl_avg_perimeter.setText(f"Avg Perimeter: {avg_perim:.2f} {unit}")
        else:
            self.lbl_avg_area.setText(f"Avg Area: - ({unit}²)")
            self.lbl_avg_perimeter.setText(f"Avg Perimeter: - ({unit})")
            
        self.lbl_sensitivity_res.setText(f"Sensitivity Setting: {self.slider_sensitivity.value()}")

    def set_tool_mode(self, mode):
        self.canvas.set_mode(mode)
        # Uncheck other buttons
        self.btn_pan.setChecked(mode == "pan")
        self.btn_draw.setChecked(mode == "draw")
        self.btn_delete.setChecked(mode == "delete")
        self.btn_void.setChecked(mode == "void")
        self.btn_add.setChecked(mode == "add")
        self.btn_calibrate.setChecked(mode == "calibrate")
        
        msg = {
            "pan": "Mode: Pan & Zoom",
            "draw": "Mode: Draw Border (Closed=Cell, Open=Divisor)",
            "delete": "Mode: Delete and Merge (Click cell)",
            "void": "Mode: Void / Clear Area (Click artifact)",
            "add": "Mode: Add Cell (Click interior)",
            "calibrate": "Mode: Calibrate (Draw line over scale bar)"
        }.get(mode, "")
        self.statusBar.showMessage(msg)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.tif *.tiff *.czi *.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            # Clear previous analysis data
            self.processor.mask = None
            self.processor.labels = None
            self.processor.results = []
            self.processor.manual_divisors = []
            self.canvas.clear_overlays() # Clear canvas drawings
            self.table_results.setRowCount(0)
            self.lbl_count.setText("Cell Count: 0")
            self.lbl_total_area.setText("Total Area: -")
            self.lbl_avg_area.setText("Avg Area: -")
            self.lbl_avg_perimeter.setText("Avg Perimeter: -")
            
            if self.processor.load_image(file_path):
                self.canvas.set_image(self.processor.original_image)
                self.lbl_filename.setText(f"File: {os.path.basename(file_path)}")
                self.statusBar.showMessage(f"Loaded: {file_path}")
                self.undo_stack = []
                self.btn_undo.setEnabled(False)
                
                # Update UI for multi-channel if needed (though buttons are hidden)
                if len(self.processor.channels) > 1:
                    self.statusBar.showMessage(f"Loaded {len(self.processor.channels)} channels. Press 'C' to cycle.")
                
                # Update UI to show default units even before analysis
                self.update_results_table()
            else:
                QMessageBox.critical(self, "Error", "Could not load image.")

    def finish_calibration(self, pixel_length):
        if pixel_length <= 0:
            return
            
        # Create a custom dialog for calibration
        dialog = QDialog(self)
        dialog.setWindowTitle("Calibration")
        layout = QVBoxLayout(dialog)
        
        form_layout = QHBoxLayout()
        
        # Distance input
        dist_layout = QVBoxLayout()
        dist_layout.addWidget(QLabel("Real distance:"))
        spin_dist = QDoubleSpinBox()
        spin_dist.setRange(0.001, 100000.0)
        spin_dist.setValue(100.0)
        dist_layout.addWidget(spin_dist)
        form_layout.addLayout(dist_layout)
        
        # Unit input
        unit_layout = QVBoxLayout()
        unit_layout.addWidget(QLabel("Unit name:"))
        edit_unit = QLineEdit(self.processor.unit_name if self.processor.unit_name != "px" else "µm")
        unit_layout.addWidget(edit_unit)
        form_layout.addLayout(unit_layout)
        
        layout.addLayout(form_layout)
        
        # Buttons
        btn_box = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(dialog.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dialog.reject)
        btn_box.addStretch()
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dist = spin_dist.value()
            unit = edit_unit.text().strip() or "µm"
            
            microns_per_pixel = dist / pixel_length
            self.processor.unit_name = unit
            self.processor.microns_per_pixel = microns_per_pixel
            
            # Block signals temporarily to prevent double update_results_table call
            self.spin_pixel_size.blockSignals(True)
            self.spin_pixel_size.setValue(microns_per_pixel)
            self.spin_pixel_size.blockSignals(False)
            
            # Force update of labels and results table to reflect new unit
            self.update_results_table()
            self.statusBar.showMessage(f"Calibrated: {microns_per_pixel:.4f} {unit}/px")
            self.set_tool_mode("pan")

    def export_image(self):
        if self.processor.original_image is None:
            return
            
        # Default folder is the image folder
        default_dir = os.path.dirname(self.processor.current_file_path) if self.processor.current_file_path else ""
        default_path = os.path.join(default_dir, f"analyzed_{self.processor.current_filename}") if self.processor.current_filename else ""
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Analyzed Image", default_path, "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)"
        )
        if file_path:
            # We save the labeled image from results
            _, labeled_img = self.processor.get_analysis_results()
            if labeled_img is not None:
                cv2.imwrite(file_path, cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR))
                self.statusBar.showMessage(f"Image saved to {file_path}")
                QMessageBox.information(self, "Success", f"Analyzed image successfully saved to:\n{file_path}")

    def export_excel(self):
        if not self.processor.results:
            QMessageBox.warning(self, "Warning", "No results to export. Run analysis first.")
            return
            
        # Default folder is the image folder
        default_dir = os.path.dirname(self.processor.current_file_path) if self.processor.current_file_path else ""
        default_filename = os.path.splitext(self.processor.current_filename)[0] + "_results.xlsx" if self.processor.current_filename else "results.xlsx"
        default_path = os.path.join(default_dir, default_filename)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel Results", default_path, "Excel Files (*.xlsx)"
        )
        if file_path:
            df_details = pd.DataFrame(self.processor.results)
            # Rename columns to include units
            unit = self.processor.unit_name
            df_details = df_details.rename(columns={
                'label': 'ID',
                'area': f'Area ({unit}²)',
                'perimeter': f'Perimeter ({unit})'
            })
            # Drop centroid from Excel export for simplicity
            if 'centroid' in df_details.columns:
                df_details = df_details.drop(columns=['centroid'])
            
            # Create summary data
            total_area = sum(r['area'] for r in self.processor.results)
            avg_area = df_details[f'Area ({unit}²)'].mean()
            
            summary_data = [
                ["Original Image:", self.processor.current_filename or "N/A"],
                ["Total Cell Count:", len(self.processor.results)],
                ["Average Area:", f"{avg_area:.2f}"],
                ["Total Area:", f"{total_area:.2f}"],
                ["Unit:", unit],
                ["Sensitivity:", self.slider_sensitivity.value()],
                ["Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                [], # Empty row
            ]
            
            try:
                # Write to single sheet
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # Write summary manually first
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, index=False, header=False, sheet_name='AdiQuant Results')
                    
                    # Write details starting after summary
                    df_details.to_excel(writer, index=False, sheet_name='AdiQuant Results', startrow=len(summary_data))
                
                self.statusBar.showMessage(f"Results exported to {file_path}")
                QMessageBox.information(self, "Success", f"Excel results successfully exported to:\n{file_path}")
                
                # Automatically open the file
                os.startfile(file_path)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export Excel: {str(e)}")

    def copy_to_clipboard(self):
        if not self.processor.results:
            return
            
        unit = self.processor.unit_name
        # Create a text representation of the results
        text = "AdiQuant Analysis Results\n"
        text += f"File: {self.processor.current_filename}\n"
        text += f"Cell Count: {len(self.processor.results)}\n"
        text += f"Total Area: {sum(r['area'] for r in self.processor.results):.2f} {unit}²\n\n"
        text += f"ID\tArea ({unit}²)\tPerimeter ({unit})\n"
        
        for res in self.processor.results:
            text += f"{res['label']}\t{res['area']:.2f}\t{res['perimeter']:.2f}\n"
            
        QApplication.clipboard().setText(text)
        self.statusBar.showMessage("Results copied to clipboard.")
        QMessageBox.information(self, "Success", "Analysis results copied to clipboard.")

    def show_about(self):
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About AdiQuant")
        about_dialog.setMinimumSize(500, 600)
        layout = QVBoxLayout(about_dialog)
        
        # Title
        title_label = QLabel("AdiQuant")
        title_label.setStyleSheet("font-size: 26px; font-weight: bold; color: #2c3e50;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        tabs = QTabWidget()
        
        # Tab 1: Get Started
        general_tab = QWidget()
        gen_layout = QVBoxLayout(general_tab)
        
        start_text = QLabel(
            "<h3>Workflow Guide</h3>"
            "<b>1. Load Image:</b> Open your histology slide (TIF, JPG, PNG, or CZI).<br><br>"
            "<b>2. Calibrate:</b> Use <i>Calibrate from Scale Bar</i> to draw a line over your scale bar and enter its real length and unit (e.g., µm).<br><br>"
            "<b>3. Analysis Settings:</b> Adjust the <i>Sensitivity</i> slider. Lower values detect larger/looser cells; higher values are stricter.<br><br>"
            "<b>4. Run Analysis:</b> Click <i>Run Analysis</i> to automatically segment adipocytes. Results will appear in the table below.<br><br>"
            "<b>5. Manual Refinement:</b> Use <i>Draw Border</i> to split cells or <i>Delete/Merge</i> to fix over-segmentation. Use <i>Void</i> to clear artifacts.<br><br>"
            "<b>6. Export Results:</b> Save the analyzed image with overlays or export the full dataset to a single-sheet Excel report.<br><br>"
            "<b>7. Shortcuts:</b> Use <b>Ctrl+Z</b> to undo manual corrections and <b>Scroll Wheel</b> to zoom."
        )
        start_text.setWordWrap(True)
        gen_layout.addWidget(start_text)
        gen_layout.addStretch()
        tabs.addTab(general_tab, "Get Started")
        
        # Tab 2: Methodology
        method_tab = QWidget()
        method_layout = QVBoxLayout(method_tab)
        
        method_text = QLabel(
            "<h3>Quantification Method</h3>"
            "<b>Preprocessing:</b><br>"
            "Applies grayscale conversion and optional <i>CLAHE</i> (Contrast Limited Adaptive Histogram Equalization) to enhance cellular boundaries.<br><br>"
            "<b>Segmentation:</b><br>"
            "Uses an adaptive thresholding and <i>Watershed Algorithm</i> to separate individual adipocytes based on membrane intensity.<br><br>"
            "<b>Physical Scaling:</b><br>"
            "Converts pixel-based area and perimeter to physical units (e.g., µm², µm) using the calibrated scale factor.<br><br>"
            "<b>Manual Corrections:</b><br>"
            "Integrates manual splitting and merging directly into the label matrix, ensuring real-time updates to statistics."
        )
        method_text.setWordWrap(True)
        method_layout.addWidget(method_text)
        method_layout.addStretch()
        tabs.addTab(method_tab, "Methodology")
        
        # Tab 3: Credits
        credits_tab = QWidget()
        cred_layout = QVBoxLayout(credits_tab)
        
        info_layout = QFormLayout()
        info_layout.addRow("Version:", QLabel(self.version))
        info_layout.addRow("Created:", QLabel(self.creation_date))
        info_layout.addRow("Author:", QLabel(self.author))
    
        
        # License Info
        license_label = QLabel("This software is free for academic and non-commercial research use. Commercial use requires a separate license agreement. See the LICENSE file for details.")
        license_label.setWordWrap(True)
        license_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        info_layout.addRow("License:", license_label)
        
        cred_layout.addLayout(info_layout)
        
        desc_label = QLabel("\nAutomated adipocyte quantification for\nprecise histological analysis and research.")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("font-style: italic; color: #7f8c8d;")
        cred_layout.addWidget(desc_label)
        
        cred_layout.addStretch()
        tabs.addTab(credits_tab, "Credits")
        
        layout.addWidget(tabs)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(about_dialog.accept)
        layout.addWidget(close_btn)
        
        about_dialog.exec()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dcdde1;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #2f3640;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2471a3;
            }
            QPushButton:checked {
                background-color: #2ecc71;
            }
            QPushButton#btn_open {
                background-color: #2ecc71;
            }
            QPushButton#btn_open:hover {
                background-color: #27ae60;
            }
            QPushButton#btn_analyze {
                background-color: #9b59b6;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton#btn_analyze:hover {
                background-color: #8e44ad;
            }
            QLabel#lbl_filename {
                background-color: #ffffff;
                border-bottom: 1px solid #dcdde1;
            }
            QTableWidget {
                border: 1px solid #dcdde1;
                gridline-color: #f5f6fa;
                selection-background-color: #3498db;
            }
            QHeaderView::section {
                background-color: #f5f6fa;
                padding: 4px;
                border: none;
                font-weight: bold;
            }
        """)

    def create_help_label(self, text):
        return HelpLabel(text, self)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

