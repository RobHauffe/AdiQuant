from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsEllipseItem
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import QPixmap, QImage, QPen, QColor, QPainter, QPainterPath, QCursor
import numpy as np

class ImageCanvas(QGraphicsView):
    # Signal emitted when a manual border path is drawn
    border_drawn = Signal(list) # List of (x, y) tuples
    # Signal emitted when a calibration line is drawn
    calibration_drawn = Signal(float) # pixel_length
    # Signal emitted when a point is clicked (for deletion)
    point_clicked = Signal(float, float) # (x, y)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Mask for Adipocytes (Green)
        self.adipocyte_mask_item = QGraphicsPixmapItem()
        self.adipocyte_mask_item.setOpacity(0.5)
        self.scene.addItem(self.adipocyte_mask_item)

        # Legacy alias for backward compatibility or simple usage
        self.mask_item = self.adipocyte_mask_item

        # self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.drawing = False
        self.current_path_item = None
        self.path_points = []
        self.mode = "pan" 
        self.divisors = [] # List of lists of (x, y) points for permanent display

    def set_image(self, numpy_image, is_analysis=False):
        """Sets the main image from a numpy array (RGB)."""
        height, width, channel = numpy_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(numpy_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_img)
        self.pixmap_item.setPixmap(pixmap)
        
        if not is_analysis:
            self.scene.setSceneRect(QRectF(0, 0, width, height))
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            # Clear divisors only when a fresh original image is loaded
            self.clear_divisors()

    def clear_divisors(self):
        """Removes all manual divisor lines from the scene."""
        protected_items = [self.pixmap_item, self.adipocyte_mask_item, self.mask_item]
        for item in self.scene.items():
            if (isinstance(item, QGraphicsLineItem) or isinstance(item, QGraphicsPathItem)) and item not in protected_items:
                self.scene.removeItem(item)
        self.divisors = []

    def add_divisor_path(self, points):
        """Adds a permanent visual path for a border."""
        if len(points) < 2:
            return
            
        path = QPainterPath()
        path.moveTo(points[0][0], points[0][1])
        for p in points[1:]:
            path.lineTo(p[0], p[1])
            
        path_item = QGraphicsPathItem(path)
        # Yellow is very visible on both blue (DAPI) and gray (unstained/H&E) backgrounds
        pen = QPen(QColor(255, 255, 0)) 
        pen.setWidth(2)
        path_item.setPen(pen)
        self.scene.addItem(path_item)
        self.divisors.append(path_item) # Store the item itself for easy removal

    def remove_last_divisor(self):
        """Removes the most recently added divisor line."""
        if self.divisors:
            last_item = self.divisors.pop()
            self.scene.removeItem(last_item)

    def set_mask(self, mask_array, color=(255, 0, 255)):
        """
        Sets the adipocyte segmentation mask overlay.
        Default color changed to Magenta/Purple for better contrast on green/blue backgrounds.
        """
        if mask_array is None:
            self.adipocyte_mask_item.setPixmap(QPixmap())
            return
            
        height, width = mask_array.shape
        colored_mask = np.zeros((height, width, 4), dtype=np.uint8)
        r, g, b = color
        # 100 alpha (slightly more transparent for better visibility of underlying tissue)
        colored_mask[mask_array > 0] = [r, g, b, 100] 
        
        q_img = QImage(colored_mask.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(q_img)
        self.adipocyte_mask_item.setPixmap(pixmap)

    def wheelEvent(self, event):
        """Handle zooming with the mouse wheel."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

    def keyPressEvent(self, event):
        """Handle panning with arrow keys."""
        pan_step = 20
        if event.key() == Qt.Key.Key_Left:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - pan_step)
        elif event.key() == Qt.Key.Key_Right:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + pan_step)
        elif event.key() == Qt.Key.Key_Up:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - pan_step)
        elif event.key() == Qt.Key.Key_Down:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + pan_step)
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if self.mode in ["draw", "calibrate"] and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            # Get position in scene coordinates
            scene_pos = self.mapToScene(event.pos())
            
            # Store as QPointF for precise QGraphicsItem positioning
            self.start_point = scene_pos
            # Store as integer tuples for the image processor
            self.path_points = [(int(scene_pos.x()), int(scene_pos.y()))]
            
            self.current_path_item = None
        elif self.mode in ["delete", "void", "add"] and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.point_clicked.emit(scene_pos.x(), scene_pos.y())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        
        if self.mode in ["draw", "calibrate"] and self.drawing:
            # 1. Ignore points outside the actual image area
            if not self.scene.sceneRect().contains(scene_pos):
                return
            
            # 2. Strict threshold to ignore jitter and early (0,0) reports
            dx = scene_pos.x() - self.start_point.x()
            dy = scene_pos.y() - self.start_point.y()
            if (dx*dx + dy*dy) < 25: # 5 pixels squared
                return

            if self.mode == "draw":
                new_pt = (int(scene_pos.x()), int(scene_pos.y()))
                
                # Only update if we actually have a new pixel coordinate
                if new_pt != self.path_points[-1]:
                    # Prevent adding (0,0) or near-zero coordinates if they aren't the start point
                    # This is the most common source of the "top-left" line
                    if len(self.path_points) > 1 and (abs(new_pt[0]) < 1 and abs(new_pt[1]) < 1):
                        return
                        
                    self.path_points.append(new_pt)
                    
                    # 3. Rebuild path from scratch to ensure no "ghost" points to (0,0) exist
                    path = QPainterPath()
                    path.moveTo(self.start_point)
                    for pt in self.path_points[1:]:
                        path.lineTo(QPointF(pt[0], pt[1]))
                    
                    if self.current_path_item is None:
                        self.current_path_item = QGraphicsPathItem(path)
                        pen = QPen(QColor(255, 255, 0)) # Bright Yellow for live drawing
                        pen.setWidth(2)
                        pen.setCosmetic(True)
                        self.current_path_item.setPen(pen)
                        self.scene.addItem(self.current_path_item)
                    else:
                        self.current_path_item.setPath(path)
                        
            elif self.mode == "calibrate":
                if self.current_path_item is None:
                    self.current_path_item = QGraphicsLineItem(
                        self.start_point.x(), self.start_point.y(),
                        scene_pos.x(), scene_pos.y()
                    )
                    pen = QPen(QColor(241, 196, 15)) # Sun Flower Yellow
                    pen.setWidth(2)
                    pen.setCosmetic(True)
                    self.current_path_item.setPen(pen)
                    self.scene.addItem(self.current_path_item)
                else:
                    self.current_path_item.setLine(
                        self.start_point.x(), self.start_point.y(),
                        scene_pos.x(), scene_pos.y()
                    )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode in ["draw", "calibrate"] and self.drawing and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            
            scene_pos = self.mapToScene(event.pos())
            
            if self.mode == "draw":
                if len(self.path_points) > 1:
                    self.border_drawn.emit(self.path_points)
            elif self.mode == "calibrate":
                # Calculate pixel length
                dx = scene_pos.x() - self.start_point.x()
                dy = scene_pos.y() - self.start_point.y()
                pixel_length = (dx**2 + dy**2)**0.5
                self.calibration_drawn.emit(pixel_length)
            
            # Remove the temporary drawing
            if self.current_path_item:
                self.scene.removeItem(self.current_path_item)
                self.current_path_item = None
            self.path_points = []
        else:
            super().mouseReleaseEvent(event)

    def set_mode(self, mode):
        self.mode = mode
        if mode == "pan":
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setMouseTracking(False)
        elif mode in ["draw", "calibrate"]:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setMouseTracking(False)
        else: # e.g., "delete"
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setMouseTracking(False)

    def enterEvent(self, event):
        super().enterEvent(event)

    def leaveEvent(self, event):
        super().leaveEvent(event)

    def clear_overlays(self):
        """Removes all manual divisor lines from the scene."""
        for item in self.divisors:
            if item in self.scene.items():
                self.scene.removeItem(item)
        self.divisors = []
        self.mask_item.setPixmap(QPixmap()) # Clear the mask overlay as well
