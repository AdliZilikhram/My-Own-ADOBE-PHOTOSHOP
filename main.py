import sys
import cv2
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QAction,
    QLabel, QSlider, QLineEdit, QPushButton, QMessageBox, QFileDialog, QComboBox,
    QColorDialog, QInputDialog, QStackedLayout, QCheckBox, QSplashScreen
)
from PyQt5.QtCore import Qt, QRect, QSize, QPoint, QPointF, QTimer
from PyQt5.QtGui import( QPainter, QPixmap, QColor, QImage, QIcon, QPen, QIntValidator, QPolygon, QFont, QFontDatabase,
                        QImageWriter, QTransform, QMovie
)

from PyQt5.QtGui import QFontMetrics, QPen
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import color
from skimage.transform import resize

class SplashScreen(QWidget):
    def __init__(self, gif_path, width=800, height=500, duration=5000):
        super().__init__()
        
        # Set window flags for frameless and transparent background
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(width, height)

        # Create layout for the GIF
        layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Load the GIF
        self.movie = QMovie(gif_path)
        self.movie.setScaledSize(self.size())  # Resize the GIF to match the splash screen size
        self.label.setMovie(self.movie)

        layout.addWidget(self.label)
        self.setLayout(layout)

        # Start the GIF animation
        self.movie.start()

        # Automatically close the splash screen after 'duration' milliseconds
        QTimer.singleShot(duration, self.close)

def dehaze(image, omega=0.95, t0=0.1):
    """
    Perform dehazing on an input image.
    Args:
        image (np.ndarray): Input foggy or hazy image (BGR format).
        omega (float): Controls the amount of haze removal (default is 0.95).
        t0 (float): Lower bound for transmission map (default is 0.1).
    Returns:
        np.ndarray: Dehazed image (BGR format).
    """
    def dark_channel(image, size=15):
        """Compute the dark channel of the image."""
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        return cv2.erode(min_channel, kernel)

    def estimate_atmospheric_light(image, dark_channel):
        """Estimate atmospheric light based on the dark channel."""
        num_pixels = image.shape[0] * image.shape[1]
        num_brightest = max(num_pixels // 1000, 1)
        dark_vec = dark_channel.ravel()
        indices = np.argpartition(dark_vec, -num_brightest)[-num_brightest:]
        brightest = image.reshape(-1, 3)[indices]
        return np.max(brightest, axis=0)

    # Convert BGR to RGB for consistency in processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize the image to [0, 1] range
    norm_image = image / 255.0

    # Compute dark channel
    dark = dark_channel(norm_image)

    # Estimate atmospheric light
    A = estimate_atmospheric_light(norm_image, dark)

    # Estimate transmission map
    transmission = 1 - omega * dark / np.max(A)

    # Refine the transmission map using a guided filter
    transmission = cv2.ximgproc.guidedFilter(
        guide=norm_image.astype(np.float32),
        src=transmission.astype(np.float32),
        radius=60,
        eps=0.0001
    )

    # Ensure transmission is not too low
    transmission = np.clip(transmission, t0, 1)

    # Recover the scene radiance
    J = (norm_image - A) / transmission[:, :, np.newaxis] + A
    J = np.clip(J, 0, 1)

    # Convert back to 8-bit image and return in BGR format
    dehazed_image = (J * 255).astype(np.uint8)
    return cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR)

class MiniCanvasWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mini Canvas")
        self.resize(300, 300)  # Adjust the size as needed

        # Enable the window to move independently
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        # Add a QLabel to display the canvas
        self.canvas_label = QLabel(self)
        self.canvas_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas_label)

    def update_canvas(self, pixmap):
        self.canvas_label.setPixmap(pixmap.scaled(self.canvas_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class GridTool:
    def __init__(self, parent_canvas):
        self.parent_canvas = parent_canvas
        self.show_grid = False
        self.show_ruler = False
        self.ruler_color = QColor(150, 150, 150)  # Default ruler color
        self.tick_color = QColor(100, 100, 100)   # Default tick color
        self.number_color = QColor(50, 50, 50)

    def toggle_grid(self):
        """Toggle the visibility of the grid."""
        self.show_grid = not self.show_grid
        self.parent_canvas.update()  # Trigger a repaint of the canvas

    def toggle_ruler(self):
        """Toggle the visibility of the ruler."""
        self.show_ruler = not self.show_ruler
        self.parent_canvas.update()  # Trigger a repaint of the canvas

    def draw_grid(self, painter):
        """Draw the grid if it's enabled."""
        if not self.show_grid:
            return
        painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.DashLine))
        spacing = 20  # Grid spacing
        for x in range(self.parent_canvas.canvas_x, 
                       self.parent_canvas.canvas_x + self.parent_canvas.width, spacing):
            painter.drawLine(x, self.parent_canvas.canvas_y, 
                             x, self.parent_canvas.canvas_y + self.parent_canvas.height)
        for y in range(self.parent_canvas.canvas_y, 
                       self.parent_canvas.canvas_y + self.parent_canvas.height, spacing):
            painter.drawLine(self.parent_canvas.canvas_x, y, 
                             self.parent_canvas.canvas_x + self.parent_canvas.width, y)

    def draw_ruler(self, painter):
        """Draw the ruler with corrected positions."""
        if not self.show_ruler:
            return

        ruler_thickness = 20
        small_tick_size = 5
        large_tick_size = 10
        tick_interval = 10
        number_interval = 50

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(self.number_color))

        # Top ruler (horizontal)
        painter.fillRect(self.parent_canvas.canvas_x, self.parent_canvas.canvas_y,
                        self.parent_canvas.width, ruler_thickness, self.ruler_color)
        for x in range(0, self.parent_canvas.width, tick_interval):
            x_position = x + self.parent_canvas.canvas_x
            if x % number_interval == 0:
                painter.drawLine(x_position, self.parent_canvas.canvas_y,
                                x_position, self.parent_canvas.canvas_y + large_tick_size)
                painter.drawText(x_position + 2, self.parent_canvas.canvas_y + 15, str(x))
            else:
                painter.drawLine(x_position, self.parent_canvas.canvas_y,
                                x_position, self.parent_canvas.canvas_y + small_tick_size)

        # Left ruler (vertical)
        painter.fillRect(self.parent_canvas.canvas_x, self.parent_canvas.canvas_y,
                        ruler_thickness, self.parent_canvas.height, self.ruler_color)
        for y in range(0, self.parent_canvas.height, tick_interval):
            y_position = y + self.parent_canvas.canvas_y
            if y % number_interval == 0:
                painter.drawLine(self.parent_canvas.canvas_x, y_position,
                                self.parent_canvas.canvas_x + large_tick_size, y_position)
                painter.drawText(self.parent_canvas.canvas_x + 5, y_position + 5, str(y))
            else:
                painter.drawLine(self.parent_canvas.canvas_x, y_position,
                                self.parent_canvas.canvas_x + small_tick_size, y_position)


class TextTool:
    def __init__(self, canvas):
        self.canvas = canvas
        self.current_text = ""
        self.text_objects = []  # Stores all finalized text
        self.current_font = QFont("Arial", 12)
        self.pen_color = QColor(Qt.black)
        self.pen_opacity = 1.0  # New: Add opacity setting
        self.is_active = False
        self.text_rect = None  # Rectangle for the dashed box

    def add_text(self, position, text, font, color, opacity):
        text_object = {
            'position': position,
            'text': text,
            'font': self.current_font,
            'color': color,
            'opacity': opacity,
        }
        self.text_objects.append(text_object)
        self.canvas.save_state()  # Save state after adding text
        self.canvas.update()

    def set_text_settings(self, color, font_name, size, opacity):
        """Update text settings."""
        self.pen_color = QColor(color)
        self.current_font = QFont(font_name, size)
        self.pen_opacity = opacity

    def start_typing(self, event):
        """Start text input."""
        self.start_pos = event.pos()
        self.current_text = ""
        self.is_active = True
        self.update_text_rect()  # Update rectangle size

    def update_text_rect(self):
        """Calculate and update the dashed rectangle size."""
        if self.start_pos:
            font_metrics = QFontMetrics(self.current_font)
            text_height = font_metrics.height()
            self.text_rect = QRect(
                self.start_pos.x(),
                self.start_pos.y() - text_height,  # Align rectangle top with font baseline
                font_metrics.averageCharWidth() * 10,  # Estimate width (adjust as needed)
                text_height + 5,  # Add padding
            )

    def finalize_text(self):
        """Finalize the current text input and store it."""
        if self.current_text:
            text_obj = {
                "position": self.start_pos,
                "text": self.current_text,
                "font": self.current_font,
                "color": self.pen_color,
                "opacity": self.pen_opacity,
            }
            self.text_objects.append(text_obj)
            self.current_text = ""
            self.start_pos = None
            self.text_rect = None  # Clear the rectangle
            self.is_active = False

    def handle_key_press(self, event):
        """Handles key press events for text input."""
        if event.key() == Qt.Key_Backspace:
            self.current_text = self.current_text[:-1]
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.finalize_text()
            self.canvas.save_state()
        else:
            self.current_text += event.text()
            self.update_text_rect()  # Update rectangle when text changes

    def render_text(self, painter):
        """Render finalized text and current text in progress."""
        # Draw finalized text
        for text_obj in self.text_objects:
            painter.setFont(text_obj["font"])
            painter.setPen(text_obj["color"])
            painter.setOpacity(text_obj["opacity"])
            painter.drawText(text_obj["position"], text_obj["text"])

        # Draw current text in progress
        if self.is_active and self.current_text:
            painter.setFont(self.current_font)
            painter.setPen(self.pen_color)
            painter.setOpacity(self.pen_opacity)
            painter.drawText(self.start_pos, self.current_text)

        # Draw dashed rectangle for typing
        if self.text_rect:
            pen = QPen(Qt.DashLine)
            pen.setColor(self.pen_color)
            painter.setPen(pen)
            painter.setOpacity(1.0)  # Dashed rectangle should always be fully opaque
            painter.drawRect(self.text_rect)

class ShapeTool:
    def __init__(self, canvas):
        self.canvas = canvas
        self.start_pos = None
        self.shape_type = "Rectangle"  # Default shape type
        self.pen_color = Qt.black
        self.pen_width = 1
        self.pen_opacity = 1.0
        self.pen_style = Qt.SolidLine  # Default pen style

    def set_pen_settings(self, color, width, opacity, style=Qt.SolidLine):
        """Update pen settings."""
        self.pen_color = color
        self.pen_width = width
        self.pen_opacity = opacity
        self.pen_style = style

    def set_shape_type(self, shape_type):
        """Set the type of shape to draw."""
        self.shape_type = shape_type

    def map_to_canvas(self, widget_pos):
        """Map widget position to canvas position."""
        offset_x = self.canvas.canvas_x
        offset_y = self.canvas.canvas_y
        scale_x = self.canvas.width / self.canvas.pixmap.width()
        scale_y = self.canvas.height / self.canvas.pixmap.height()
        canvas_x = (widget_pos.x() - offset_x) / scale_x
        canvas_y = (widget_pos.y() - offset_y) / scale_y
        canvas_x = max(0, min(canvas_x, self.canvas.pixmap.width()))
        canvas_y = max(0, min(canvas_y, self.canvas.pixmap.height()))
        return QPoint(int(canvas_x), int(canvas_y))

    def start_drawing(self, event):
        """Start drawing a shape."""
        self.start_pos = self.map_to_canvas(event.pos())

    def continue_drawing(self, event):
        """Preview the shape while dragging the mouse."""
        if self.start_pos:
            current_pos = self.map_to_canvas(event.pos())
            self.canvas.overlay_pixmap.fill(Qt.transparent)  # Clear overlay pixmap

            painter = QPainter(self.canvas.overlay_pixmap)
            painter.setOpacity(self.pen_opacity)
            pen = QPen(self.pen_color, self.pen_width, self.pen_style)
            painter.setPen(pen)

            self.draw_shape(painter, self.start_pos, current_pos)
            painter.end()
            self.canvas.update()

    def end_drawing(self, event):
        """Finalize the shape."""
        if self.start_pos:
            current_pos = self.map_to_canvas(event.pos())

            painter = QPainter(self.canvas.pixmap)
            painter.setOpacity(self.pen_opacity)
            pen = QPen(self.pen_color, self.pen_width, self.pen_style)
            painter.setPen(pen)

            self.draw_shape(painter, self.start_pos, current_pos)
            painter.end()

            #self.canvas.overlay_pixmap.fill(Qt.transparent)  # Clear overlay
            self.start_pos = None
            self.canvas.update()

    def draw_shape(self, painter, start_pos, current_pos):
        """Draw the selected shape."""
        rect = QRect(start_pos, current_pos)
        if self.shape_type == "Rectangle":
            painter.drawRect(rect)
        elif self.shape_type == "Ellipse":
            painter.drawEllipse(rect)
        elif self.shape_type == "Circle":
            # Enforce a square rect for a circle
            size = min(rect.width(), rect.height())
            painter.drawEllipse(QRect(rect.topLeft(), QSize(size, size)))
        elif self.shape_type == "Square":
            # Enforce a square rect for a square
            size = min(rect.width(), rect.height())
            painter.drawRect(QRect(rect.topLeft(), QSize(size, size)))
        elif self.shape_type == "Triangle":
            # Draw an isosceles triangle
            points = [
                QPoint(rect.center().x(), rect.top()),  # Top-center
                QPoint(rect.bottomLeft()),  # Bottom-left
                QPoint(rect.bottomRight()),  # Bottom-right
            ]
            painter.drawPolygon(QPolygon(points))
        elif self.shape_type == "Line":
            painter.drawLine(start_pos, current_pos)
        elif self.shape_type == "DashLine":
            # Use a dashed pen for drawing the line
            dashed_pen = QPen(self.pen_color, self.pen_width, Qt.DashLine)
            painter.setPen(dashed_pen)
            painter.drawLine(start_pos, current_pos)

class DrawingTool:
    def __init__(self, canvas):
        self.canvas = canvas  # Reference to the main canvas
        #self.is_drawing = False  # Track whether drawing is active
        self.last_pos = None  # Last mouse position
        self.pen_color = Qt.black
        self.pen_width = 1
        self.pen_opacity = 1.0
        self.pen_style = Qt.SolidLine

    def map_to_canvas(self, widget_pos):
        """Map widget position to canvas position, accounting for offsets and scaling."""
        # Account for canvas offsets
        offset_x = self.canvas.canvas_x
        offset_y = self.canvas.canvas_y
        
        # Scaling factors
        scale_x = self.canvas.width / self.canvas.pixmap.width()
        scale_y = self.canvas.height / self.canvas.pixmap.height()
        
        # Adjust position to account for canvas position and scaling
        canvas_x = (widget_pos.x() - offset_x) / scale_x
        canvas_y = (widget_pos.y() - offset_y) / scale_y

        # Clamp values to canvas bounds
        canvas_x = max(0, min(canvas_x, self.canvas.pixmap.width()))
        canvas_y = max(0, min(canvas_y, self.canvas.pixmap.height()))

        return QPoint(int(canvas_x), int(canvas_y))

    def set_pen_settings(self, color, width, opacity, style=Qt.SolidLine):
        self.pen_color = color
        self.pen_width = width
        self.pen_opacity = opacity
        self.pen_style = style

    def start_drawing(self, event):
        """Start the drawing process."""
        self.is_drawing = True
        self.last_pos = self.map_to_canvas(event.pos())

    def continue_drawing(self, event):
        """Continue drawing as the mouse moves."""
        if self.is_drawing:
            current_pos = self.map_to_canvas(event.pos())
            painter = QPainter(self.canvas.overlay_pixmap)
            painter.setPen(QPen(self.pen_color, self.pen_width, self.pen_style))  # Customize pen as needed
            painter.setOpacity(self.pen_opacity)
            painter.drawLine(self.last_pos, current_pos)
            painter.end()
            self.last_pos = current_pos
            self.canvas.update()
            
    def end_drawing(self):
        """Finalize the drawing process."""
        if self.is_drawing:
            # Commit the overlay drawing to the main pixmap
            painter = QPainter(self.canvas.pixmap)
            painter.drawPixmap(0, 0, self.canvas.overlay_pixmap)
            painter.end()

            # Clear the overlay pixmap
            #self.canvas.overlay_pixmap.fill(Qt.transparent)
            self.is_drawing = False
            self.last_pos = None
            self.canvas.update()
        
class EraseTool:
    def __init__(self, canvas):
        self.canvas = canvas  # Reference to the main canvas
        self.last_pos = None  # Last mouse position
        self.pen_width = 10  # Default eraser size
        self.pen_style = Qt.SolidLine

    def map_to_canvas(self, widget_pos):
        """Map widget position to canvas position, accounting for offsets and scaling."""
        offset_x = self.canvas.canvas_x
        offset_y = self.canvas.canvas_y
        scale_x = self.canvas.width / self.canvas.pixmap.width()
        scale_y = self.canvas.height / self.canvas.pixmap.height()
        canvas_x = (widget_pos.x() - offset_x) / scale_x
        canvas_y = (widget_pos.y() - offset_y) / scale_y
        canvas_x = max(0, min(canvas_x, self.canvas.pixmap.width()))
        canvas_y = max(0, min(canvas_y, self.canvas.pixmap.height()))
        return QPoint(int(canvas_x), int(canvas_y))

    def start_erasing(self, event):
        """Start the erasing process."""
        self.is_erasing = True
        self.last_pos = self.map_to_canvas(event.pos())

    def continue_erasing(self, event):
        """Erase continuously as the mouse moves."""
        if self.is_erasing:
            current_pos = self.map_to_canvas(event.pos())
            painter = QPainter(self.canvas.overlay_pixmap)  # Use the overlay_pixmap
            painter.setPen(QPen(Qt.white, self.pen_width, self.pen_style))  # Erase with white
            painter.drawLine(self.last_pos, current_pos)
            painter.end()
            self.last_pos = current_pos
            self.canvas.update()  # Update the canvas display

    def end_erasing(self):
        """Finalize the erasing process."""
        if self.is_erasing:
            # Commit the overlay erasing to the main pixmap
            painter = QPainter(self.canvas.pixmap)
            painter.drawPixmap(0, 0, self.canvas.overlay_pixmap)
            painter.end()

            # Clear the overlay pixmap
            #self.canvas.overlay_pixmap.fill(Qt.transparent)
            self.is_erasing = False
            self.last_pos = None
            self.canvas.update()

class LeftPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(100)
        self.setStyleSheet("background-color: rgb(220, 220, 220);")

class TopPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        self.setStyleSheet("background-color: rgb(200, 200, 200);")


class CanvasApp(QWidget):
    def __init__(self):
        super().__init__()
        self.canvas_width = 800
        self.canvas_height = 600
        self.init_ui()
        

    def init_ui(self):
        self.setWindowTitle("Canvas Size Selector")
        self.setWindowIcon(QIcon(r'fairy tail.png'))
        self.setGeometry(700, 300, 400, 300)

        # Layouts
        main_layout = QVBoxLayout()
        width_layout = QHBoxLayout()
        height_layout = QHBoxLayout()

        # Preview canvas
        self.preview_label = QLabel(self)
        self.preview_pixmap = QPixmap(200, 150)  # Initial preview size
        self.preview_pixmap.fill(Qt.white)
        self.preview_label.setPixmap(self.preview_pixmap)

        # Widgets for width
        self.width_label = QLabel("Width:")
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(100, 1200)
        self.width_slider.setValue(800)
        self.width_slider.valueChanged.connect(self.update_preview)

        self.width_input = QLineEdit()
        self.width_input.setText("800")
        self.width_input.returnPressed.connect(self.set_width_from_input)

        # Widgets for height
        self.height_label = QLabel("Height:")
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(100, 800)
        self.height_slider.setValue(600)
        self.height_slider.valueChanged.connect(self.update_preview)

        self.height_input = QLineEdit()
        self.height_input.setText("600")
        self.height_input.returnPressed.connect(self.set_height_from_input)

        # Confirm button
        self.confirm_button = QPushButton("Create Canvas")
        self.confirm_button.clicked.connect(self.open_canvas_window)

        # Assemble layouts
        width_layout.addWidget(self.width_label)
        width_layout.addWidget(self.width_slider)
        width_layout.addWidget(self.width_input)

        height_layout.addWidget(self.height_label)
        height_layout.addWidget(self.height_slider)
        height_layout.addWidget(self.height_input)

        main_layout.addWidget(self.preview_label)
        main_layout.addLayout(width_layout)
        main_layout.addLayout(height_layout)
        main_layout.addWidget(self.confirm_button)

        self.setLayout(main_layout)
        self.update_preview()  # Initialize the preview

    def update_preview(self):
        # Update width and height from sliders
        self.canvas_width = self.width_slider.value()
        self.canvas_height = self.height_slider.value()

        # Update width and height input fields
        self.width_input.setText(str(self.canvas_width))
        self.height_input.setText(str(self.canvas_height))

        # Calculate scaled dimensions for the preview
        preview_width = int(self.canvas_width * 0.2)
        preview_height = int(self.canvas_height * 0.2)

        # Update preview canvas
        self.preview_pixmap = QPixmap(preview_width, preview_height)
        self.preview_pixmap.fill(Qt.white)
        painter = QPainter(self.preview_pixmap)
        painter.setPen(QColor("black"))
        painter.drawRect(0, 0, preview_width - 1, preview_height - 1)  # Draw a border
        painter.end()
        self.preview_label.setPixmap(self.preview_pixmap)

    def set_width_from_input(self):
        try:
            value = int(self.width_input.text())
            if 100 <= value <= 1920:
                self.width_slider.setValue(value)
                self.update_preview()
            else:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a width between 100 and 1920.")

    def set_height_from_input(self):
        try:
            value = int(self.height_input.text())
            if 100 <= value <= 1080:
                self.height_slider.setValue(value)
                self.update_preview()
            else:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a height between 100 and 1080.")

    def open_canvas_window(self):
        self.canvas_window = CanvasWindow(self.canvas_width, self.canvas_height)
        self.canvas_window.show()
        self.close()  # Close the size selector window

class CanvasWindow(QMainWindow):
    def __init__(self, width, height):
        super().__init__()
        self.setWindowIcon(QIcon(r'fairy tail.png'))
        self.width = width
        self.height = height
        self.grid_tool = GridTool(self)
        self.current_scale_factor = 1.0
        self.init_ui()
        
        self.container()
        
        self.pixmap.fill(Qt.white)
        self.overlay_pixmap = QPixmap(self.width, self.height)  # Separate overlay layer
        self.overlay_pixmap.fill(Qt.transparent)  # Transparent layer for drawing
        
        # History stacks
        self.undo_stack = []  # To store previous states
        self.redo_stack = []  # To store undone states
        self.image_rect = None  # Track the image position and size
        self.image_selected = False  # Whether the image is selected
        
        self.canvas_x = 100  # Initial x-offset for the canvas
        self.canvas_y = 100  # Initial y-offset for the canvas
        self.drag_mode = False  # Track whether drag mode is active
        self.scale_mode = False  # Scale mode flag
        self.image_x = None  # X-position of the image relative to the canvas
        self.image_y = None  # Y-position of the image relative to the canvas
        self.image_pixmap = None 

        # Define the left panel rectangle
        self.background = QRect(0, 0, 1950, 980)
        self.left_panel_rect = QRect(0, 90, 59, 980)
        self.right_panel_rect = QRect(1400, 120, 500, 880)
        self.top_panel_rect = QRect(0, 20, 1950, 50)
        
        # Manage multiple images
        self.images = []  # List of dictionaries to store image details
        self.selected_image_index = None  # Track selected image
        self.anchor_clicked=False
        
        self.drawing_tool = DrawingTool(self)  # Initialize the drawing tool
        self.erase_tool = EraseTool(self)
        self.shape_tool = ShapeTool(self)
        self.text_tool = TextTool(self)
        self.drawing_mode = False
        self.eraser_mode = False
        self.shape_mode = False
        self.text_mode = False
        
        # Cropping related attributes
        self.crop_mode = False
        self.crop_start_point = None
        self.crop_end_point = None
        self.crop_rect = None 
        
    def init_ui(self):
        self.setWindowTitle("Canvas")
        self.setGeometry(0, 0, 1980, 980)
        self.pixmap = QPixmap(self.width, self.height)
        self.pixmap.fill(Qt.white)  # Fill canvas with white color

        # Menubar setup
        self.setup_menu()

        # Layouts
        main_layout = QVBoxLayout()
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Add top panel
        top_panel = TopPanel(self)
        main_layout.addWidget(top_panel)

        # Add canvas area with left panel
        canvas_layout = QHBoxLayout()
        left_panel = LeftPanel(self)
        canvas_layout.addWidget(left_panel)

        # Add the canvas (a QWidget with the pixmap drawn on it)
        canvas_widget = QWidget(self)
        canvas_widget.setMinimumSize(self.width, self.height)
        canvas_layout.addWidget(canvas_widget)

        main_layout.addLayout(canvas_layout)

        # Drag Mode Button
        self.drag_button = QPushButton(self)
        drag_icon = QIcon("move.png")
        self.drag_button.setIcon(drag_icon)
        self.drag_button.setGeometry(10, 100, 40, 40)
        self.drag_button.setIconSize(self.drag_button.size())
        self.drag_button.setStyleSheet("background: transparent; border: none;")
        self.drag_button.setCheckable(True)
        self.drag_button.clicked.connect(self.toggle_drag_mode)

        # Undo and Redo buttons
        self.undo_button = QPushButton(self)
        undo_icon = QIcon("undo.png")
        self.undo_button.setIcon(undo_icon)
        self.undo_button.setGeometry(60, 30, 40, 40)
        self.undo_button.setIconSize(self.undo_button.size())
        self.undo_button.setStyleSheet("background: transparent; border: none;")
        self.undo_button.clicked.connect(self.undo_action)

        self.redo_button = QPushButton(self)
        redo_icon = QIcon("redo.png")
        self.redo_button.setIcon(redo_icon)
        self.redo_button.setGeometry(110, 30, 40, 40)
        self.redo_button.setIconSize(self.redo_button.size())
        self.redo_button.setStyleSheet("background: transparent; border: none;")
        self.redo_button.clicked.connect(self.redo_action)
        
        # Scale Mode Button
        self.scale_button = QPushButton(self)
        scale_icon = QIcon("scale.png")  # Replace with your scale icon file path
        self.scale_button.setIcon(scale_icon)
        self.scale_button.setGeometry(10, 150, 40, 40)  # Position below the drag button
        self.scale_button.setIconSize(self.scale_button.size())
        self.scale_button.setStyleSheet("background: transparent; border: none;")
        self.scale_button.setCheckable(True)
        self.scale_button.clicked.connect(self.toggle_scale_mode)
        
        self.draw_button = QPushButton(self)
        draw_icon = QIcon("draw.png")  # Replace with your draw icon file path
        self.draw_button.setIcon(draw_icon)
        self.draw_button.setGeometry(10, 200, 40, 40)  # Position below the scale button
        self.draw_button.setIconSize(self.draw_button.size())
        self.draw_button.setStyleSheet("background: transparent; border: none;")
        self.draw_button.setCheckable(True)
        self.draw_button.clicked.connect(self.toggle_drawing_mode)
        
        # Eraser button
        self.eraser_button = QPushButton(self)
        eraser_icon = QIcon("eraser.png")  # Replace with your eraser icon file path
        self.eraser_button.setIcon(eraser_icon)
        self.eraser_button.setGeometry(10, 250, 40, 40)  # Position below the draw button
        self.eraser_button.setIconSize(self.eraser_button.size())
        self.eraser_button.setStyleSheet("background: transparent; border: none;")
        self.eraser_button.setCheckable(True)
        self.eraser_button.clicked.connect(self.toggle_eraser_mode)
        
        # Shape Tool Button
        self.shape_button = QPushButton(self)
        shape_icon = QIcon("shapes.png")  # Replace with your shape icon file path
        self.shape_button.setIcon(shape_icon)
        self.shape_button.setGeometry(10, 300, 40, 40)  # Position below the eraser button
        self.shape_button.setIconSize(self.shape_button.size())
        self.shape_button.setStyleSheet("background: transparent; border: none;")
        self.shape_button.setCheckable(True)
        self.shape_button.clicked.connect(self.toggle_shape_mode)

        self.setup_shape_dropdown()
                
        # Pen type dropdown
        self.pen_label = QLabel("Pen Type:", self)
        self.pen_label.setGeometry(190, 35, 60, 30)  # Adjust positioning as needed

        self.pen_dropdown = QComboBox(self)
        self.pen_dropdown.setGeometry(260, 35, 120, 30)  # Adjust positioning as needed
        self.pen_dropdown.addItems(["Pencil", "Brush", "Highlighter", "Marker", "Calligraphy"])
        self.pen_dropdown.currentTextChanged.connect(self.change_pen_type)
        
        # Color picker button
        self.color_button = QPushButton(self)
        color_icon = QIcon("color.png")
        self.color_button.setIcon(color_icon)
        self.color_button.setGeometry(400, 28, 40, 40)  # Adjust positioning as needed
        self.color_button.setIconSize(self.color_button.size())
        self.color_button.clicked.connect(self.choose_color)
        
        # Size
        self.size_label = QLabel("Size:", self)
        self.size_label.setGeometry(460, 30, 60, 30)  # Adjust positioning as needed
        
        # Pen Size Slider
        self.size_slider = QSlider(Qt.Horizontal, self)
        self.size_slider.setGeometry(500, 35, 120, 30)  # Adjust positioning as needed
        self.size_slider.setRange(1, 50)  # Pen size range
        self.size_slider.setValue(5)      # Default pen size
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(5)

        # Value Box (QLineEdit)
        self.size_value_box = QLineEdit(self)
        self.size_value_box.setGeometry(620, 35, 50, 30)  # Position beside the slider
        self.size_value_box.setText("5")  # Default value
        self.size_value_box.setValidator(QIntValidator(1, 50))  # Restrict input to valid range

        # Add connections
        self.size_slider.valueChanged.connect(self.update_value_box)
        self.size_value_box.textChanged.connect(self.update_slider)
        
        # Opacity
        self.opacity_label = QLabel("Opacity:", self)
        self.opacity_label.setGeometry(680, 30, 60, 30)  # Adjust positioning as needed
        
        # Opacity Slider
        self.opacity_slider = QSlider(Qt.Horizontal, self)
        self.opacity_slider.setGeometry(740, 35, 120, 30)  # Adjust positioning as needed
        self.opacity_slider.setRange(0, 100)  # Opacity range from 0% to 100%
        self.opacity_slider.setValue(100)    # Default opacity (fully opaque)
        self.opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.opacity_slider.setTickInterval(10)

        # Opacity Value Box (QLineEdit)
        self.opacity_value_box = QLineEdit(self)
        self.opacity_value_box.setGeometry(860, 35, 50, 30)  # Position beside the slider
        self.opacity_value_box.setText("100")  # Default value (100%)
        self.opacity_value_box.setValidator(QIntValidator(0, 100))  # Restrict input to valid range

        # Add connections
        self.opacity_slider.valueChanged.connect(self.update_opacity_value_box)
        self.opacity_value_box.textChanged.connect(self.update_opacity_slider)

        self.text_button = QPushButton(self)
        text_icon = QIcon("text.png")  # Replace with your text icon file path
        self.text_button.setIcon(text_icon)
        self.text_button.setGeometry(10, 350, 40, 40)  # Position below the shape button
        self.text_button.setIconSize(self.text_button.size())
        self.text_button.setStyleSheet("background: transparent; border: none;")
        self.text_button.setCheckable(True)
        self.text_button.clicked.connect(self.toggle_text_mode)
        
        self.font_label = QLabel("Font :", self)
        self.font_label.setGeometry(1065, 35, 60, 30)  # Adjust positioning as needed
        
        self.font_dropdown = QComboBox(self)
        self.font_dropdown.setGeometry(1100, 35, 150, 30)  # Adjust position and size as needed
        self.populate_fonts()  # Populate with available fonts
        self.font_dropdown.currentIndexChanged.connect(self.set_font)  # Handle font changes
        
        self.color_button.clicked.connect(self.choose_text_color)
        self.size_slider.valueChanged.connect(lambda value: self.change_text_size(value))
        self.opacity_slider.valueChanged.connect(lambda value: self.change_text_opacity(value))
        
        # Crop Mode Button
        self.crop_button = QPushButton(self)
        crop_icon = QIcon("crop.png")  # Replace with your crop icon file path
        self.crop_button.setIcon(crop_icon)
        self.crop_button.setGeometry(10, 400, 40, 40)  # Position below other buttons
        self.crop_button.setIconSize(self.crop_button.size())
        self.crop_button.setStyleSheet("background: transparent; border: none;")
        self.crop_button.setCheckable(True)
        self.crop_button.clicked.connect(self.toggle_crop_mode)
        
        # Delete Image Button
        self.delete_button = QPushButton(self)
        delete_icon = QIcon("bin.png")  # Replace with your delete icon file path
        self.delete_button.setIcon(delete_icon)
        self.delete_button.setGeometry(10, 450, 40, 40)  # Adjust position as needed
        self.delete_button.setIconSize(self.delete_button.size())
        self.delete_button.setStyleSheet("background: transparent; border: none;")
        self.delete_button.clicked.connect(self.delete_selected_image)
        
        # Flip Horizontal Button
        self.flip_horizontal_button = QPushButton(self)
        flip_horizontal_icon = QIcon("horizontal.png")  # Replace with your icon file path
        self.flip_horizontal_button.setIcon(flip_horizontal_icon)
        self.flip_horizontal_button.setGeometry(10, 500, 40, 40)  # Adjust position as needed
        self.flip_horizontal_button.setIconSize(self.flip_horizontal_button.size())
        self.flip_horizontal_button.setStyleSheet("background: transparent; border: none;")
        self.flip_horizontal_button.clicked.connect(self.flip_horizontal)

        # Flip Vertical Button
        self.flip_vertical_button = QPushButton(self)
        flip_vertical_icon = QIcon("vertical.png")  # Replace with your icon file path
        self.flip_vertical_button.setIcon(flip_vertical_icon)
        self.flip_vertical_button.setGeometry(10, 550, 40, 40)  # Adjust position as needed
        self.flip_vertical_button.setIconSize(self.flip_vertical_button.size())
        self.flip_vertical_button.setStyleSheet("background: transparent; border: none;")
        self.flip_vertical_button.clicked.connect(self.flip_vertical)
        
        # Zoom In Button
        self.zoom_in_button = QPushButton(self)
        zoom_in_icon = QIcon("zoom-in.png")  # Replace with your icon file path
        self.zoom_in_button.setIcon(zoom_in_icon)
        self.zoom_in_button.setGeometry(10, 600, 40, 40)  # Adjust position
        self.zoom_in_button.setIconSize(self.zoom_in_button.size())
        self.zoom_in_button.setStyleSheet("background: transparent; border: none;")
        self.zoom_in_button.clicked.connect(self.zoom_in)

        # Zoom Out Button
        self.zoom_out_button = QPushButton(self)
        zoom_out_icon = QIcon("zoom-out.png")  # Replace with your icon file path
        self.zoom_out_button.setIcon(zoom_out_icon)
        self.zoom_out_button.setGeometry(10, 650, 40, 40)  # Adjust position
        self.zoom_out_button.setIconSize(self.zoom_out_button.size())
        self.zoom_out_button.setStyleSheet("background: transparent; border: none;")
        self.zoom_out_button.clicked.connect(self.zoom_out)

        # Reset Zoom Button
        self.reset_zoom_button = QPushButton(self)
        reset_zoom_icon = QIcon("reset.png")  # Replace with your icon file path
        self.reset_zoom_button.setIcon(reset_zoom_icon)
        self.reset_zoom_button.setGeometry(10, 700, 40, 40)  # Adjust position
        self.reset_zoom_button.setIconSize(self.reset_zoom_button.size())
        self.reset_zoom_button.setStyleSheet("background: transparent; border: none;")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        
        
    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        import_action = QAction("Import Image", self)
        import_action.triggered.connect(self.import_image)
        file_menu.addAction(import_action)
        
        # Export as PNG action
        export_action = QAction("Export as PNG", self)
        export_action.triggered.connect(self.export_as_png)
        file_menu.addAction(export_action)
        
        # Save Project action
        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        # Open Project action
        open_project_action = QAction("Open Project", self)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)
        
        view_menu = menubar.addMenu("View")
        
        # thumbnails toggle
        thumbnails_action = QAction("Thumbnails", self, checkable=True)
        thumbnails_action.triggered.connect(self.show_mini_canvas)
        view_menu.addAction(thumbnails_action)
        
        # Gridline toggle
        gridline_action = QAction("GridLines", self, checkable=True)
        gridline_action.triggered.connect(self.grid_tool.toggle_grid)
        view_menu.addAction(gridline_action)
        
        # Ruler toggle
        ruler_action = QAction("Ruler", self, checkable=True)
        ruler_action.triggered.connect(self.grid_tool.toggle_ruler)
        view_menu.addAction(ruler_action)
        
        window_menu = menubar.addMenu("Window")
        
        # New Action to Launch the New Window Sequence
        new_window_action = QAction("New Window", self)
        new_window_action.triggered.connect(self.launch_new_window_sequence)
        window_menu.addAction(new_window_action)

        
    def container(self):
        # Create the container widget for buttons and stack layout
        self.container = QWidget(self)
        self.container.setGeometry(1400, 70, 500, 900)  # Position and size of the container

        # Layout for the container
        self.container_layout = QVBoxLayout()
        self.container.setLayout(self.container_layout)

        # Create the stacked layout
        self.stacked_layout = QStackedLayout()

        # Add pages to the stacked layout
        self.page1 = self.create_page1()
        self.page2 = self.create_page2()
        self.page3 = self.create_page3()
        self.stacked_layout.addWidget(self.page1)
        self.stacked_layout.addWidget(self.page2)
        self.stacked_layout.addWidget(self.page3)

        # Create navigation buttons
        self.btn1 = QPushButton("Image Editor")
        self.btn2 = QPushButton("Filter")
        self.btn3 = QPushButton("3D Representation")

        # Connect buttons to change pages
        self.btn1.clicked.connect(lambda: self.stacked_layout.setCurrentIndex(0))
        self.btn2.clicked.connect(lambda: self.stacked_layout.setCurrentIndex(1))
        self.btn3.clicked.connect(lambda: self.stacked_layout.setCurrentIndex(2))

        # Add navigation buttons in a horizontal layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn1)
        button_layout.addWidget(self.btn2)
        button_layout.addWidget(self.btn3)

        # Add the button layout and stacked layout to the container's layout
        self.container_layout.addLayout(button_layout)
        self.container_layout.addLayout(self.stacked_layout)

    def create_page1(self):
        """Create and return the first page with RGB histogram."""
        page = QWidget()
        layout = QVBoxLayout()
        
        # Button for standard histogram equalization
        equalize_button = QPushButton("Equalize Histogram")
        equalize_button.clicked.connect(self.apply_histogram_equalization)
        layout.addWidget(equalize_button)

        # Slider for CLAHE (Adaptive Histogram Equalization)
        clahe_label = QLabel("CLAHE Clip Limit: 2.0")
        self.clahe_slider = QSlider(Qt.Horizontal)
        self.clahe_slider.setRange(1, 40)  # Clip limit range: 0.1 to 4.0 (scaled by 10)
        self.clahe_slider.setValue(20)  # Default clip limit = 2.0
        self.clahe_slider.valueChanged.connect(lambda: self.apply_clahe(clahe_label))

        layout.addWidget(clahe_label)
        layout.addWidget(self.clahe_slider)
        
        # Piecewise Linear Transformation Slider
        piecewise_label = QLabel("Piecewise Point: 128")
        self.piecewise_slider = QSlider(Qt.Horizontal)
        self.piecewise_slider.setRange(0, 255)  # Point for piecewise linear transformation
        self.piecewise_slider.setValue(128)     # Default midpoint value
        self.piecewise_slider.valueChanged.connect(lambda: self.update_piecewise_image(piecewise_label))

        layout.addWidget(piecewise_label)
        layout.addWidget(self.piecewise_slider)

        
        # Erosion Slider
        erosion_label = QLabel("Erosion Kernel Size: 1")
        self.erosion_slider = QSlider(Qt.Horizontal)
        self.erosion_slider.setRange(1, 20)  # Kernel size from 1x1 to 20x20
        self.erosion_slider.setValue(1)
        self.erosion_slider.valueChanged.connect(lambda: self.update_erosion_image(erosion_label))

        layout.addWidget(erosion_label)
        layout.addWidget(self.erosion_slider)

        # Dilation Slider
        dilation_label = QLabel("Dilation Kernel Size: 1")
        self.dilation_slider = QSlider(Qt.Horizontal)
        self.dilation_slider.setRange(1, 20)  # Kernel size from 1x1 to 20x20
        self.dilation_slider.setValue(1)
        self.dilation_slider.valueChanged.connect(lambda: self.update_dilation_image(dilation_label))

        layout.addWidget(dilation_label)
        layout.addWidget(self.dilation_slider)
        
        # Add slider and label for thresholding
        threshold_label = QLabel("Threshold Value: 127")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)  # Threshold values range from 0 to 255
        self.threshold_slider.setValue(127)  # Default threshold value
        self.threshold_slider.valueChanged.connect(lambda: self.update_thresholding_image(threshold_label))

        layout.addWidget(threshold_label)  # Add threshold label to layout
        layout.addWidget(self.threshold_slider)  # Add threshold slider to layout
                
        # Add slider and label for sharpening
        sharp_label = QLabel("Sharpening Strength: 0.0")
        self.sharp_slider = QSlider(Qt.Horizontal)
        self.sharp_slider.setRange(0, 20)  # Represents 1.0 to 2.0 in steps of 0.1
        self.sharp_slider.setValue(0)  # Default value: 1.0
        self.sharp_slider.valueChanged.connect(lambda: self.update_sharpening_image(sharp_label))

        layout.addWidget(sharp_label)  # Add sharpening label to layout
        layout.addWidget(self.sharp_slider)  # Add sharpening slider to layout
            
            # Add buttons for bit-plane slicing
        bit_plane_label = QLabel("Bit-plane Slicing")
        bit_plane_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(bit_plane_label)

        bit_plane_button_layout = QHBoxLayout()  # Horizontal layout for bit-plane buttons

        for i in range(8):  # Create 8 buttons for bit-plane slicing
            button = QPushButton(f"Bit {i}")
            button.clicked.connect(lambda _, bit=i: self.bit_plane_slicing(bit))  # Connect button to slicing function
            bit_plane_button_layout.addWidget(button)

        layout.addLayout(bit_plane_button_layout)
        
        # Add slider and label for gamma value
        self.gamma_label = QLabel("Gamma: 1.0")
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 50)  # Gamma values scaled by 10 (0.1 to 5.0)
        self.gamma_slider.setValue(10)  # Default gamma = 1.0
        self.gamma_slider.valueChanged.connect(self.update_powerlaw_image)

        # Add the slider and label to the main layout
        layout.addWidget(self.gamma_label)  # Add label to layout
        layout.addWidget(self.gamma_slider)  # Add slider to layout

        self.image_label = QLabel("_______HISTOGRAM________")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Add checkable buttons for RGB channels
        self.check_red = QCheckBox("Red")
        self.check_green = QCheckBox("Green")
        self.check_blue = QCheckBox("Blue")

        # Arrange checkboxes vertically
        checkbox_layout = QVBoxLayout()
        checkbox_layout.addWidget(self.check_red)
        checkbox_layout.addWidget(self.check_green)
        checkbox_layout.addWidget(self.check_blue)
        layout.addLayout(checkbox_layout)

        # Add the Matplotlib canvas for the histogram
        self.figure_histogram = Figure()
        self.histogram_canvas = FigureCanvas(self.figure_histogram)
        layout.addWidget(self.histogram_canvas)

        # Connect checkboxes to update histogram
        self.check_red.stateChanged.connect(self.update_histogram)
        self.check_green.stateChanged.connect(self.update_histogram)
        self.check_blue.stateChanged.connect(self.update_histogram)

        page.setLayout(layout)
        return page
    
    def create_page2(self):
        """Create and return the second page."""
        page = QWidget()
        layout = QVBoxLayout()
        
        # --- Dehazing Button ---
        dehaze_button = QPushButton("Dehaze Image")
        dehaze_button.clicked.connect(self.apply_dehaze)
        layout.addWidget(dehaze_button)
        
        # --- Glitch Art Generator ---
        glitch_button = QPushButton("Apply Glitch Art Effect")
        glitch_button.clicked.connect(self.apply_glitch_effect)  # Connect to glitch effect function
        layout.addWidget(glitch_button)
        
        # --- Gaussian Blur ---
        gaussian_label = QLabel("Gaussian Blur - Kernel Size: 5")
        gaussian_label.setAlignment(Qt.AlignCenter)
        self.gaussian_slider = QSlider(Qt.Horizontal)
        self.gaussian_slider.setRange(1, 31)
        self.gaussian_slider.setSingleStep(2)
        self.gaussian_slider.setValue(5)
        layout.addWidget(gaussian_label)
        layout.addWidget(self.gaussian_slider)
        self.gaussian_slider.valueChanged.connect(lambda value: self.apply_gaussian_blur(value, gaussian_label))

        # --- Median Filter ---
        median_label = QLabel("Median Filter - Kernel Size: 5")
        median_label.setAlignment(Qt.AlignCenter)
        self.median_slider = QSlider(Qt.Horizontal)
        self.median_slider.setRange(1, 31)
        self.median_slider.setSingleStep(2)
        self.median_slider.setValue(5)
        layout.addWidget(median_label)
        layout.addWidget(self.median_slider)
        self.median_slider.valueChanged.connect(lambda value: self.apply_median_filter(value, median_label))

        # --- Bilateral Filter ---
        bilateral_label = QLabel("Bilateral Filter - Diameter: 9")
        bilateral_label.setAlignment(Qt.AlignCenter)
        self.bilateral_slider = QSlider(Qt.Horizontal)
        self.bilateral_slider.setRange(1, 15)
        self.bilateral_slider.setValue(9)
        layout.addWidget(bilateral_label)
        layout.addWidget(self.bilateral_slider)
        self.bilateral_slider.valueChanged.connect(lambda value: self.apply_bilateral_filter(value, bilateral_label))

        # --- Unsharp Masking (Sharpening) ---
        unsharp_label = QLabel("Unsharp Masking - Strength: 1.5")
        unsharp_label.setAlignment(Qt.AlignCenter)
        self.unsharp_slider = QSlider(Qt.Horizontal)
        self.unsharp_slider.setRange(1, 50)
        self.unsharp_slider.setValue(15)
        layout.addWidget(unsharp_label)
        layout.addWidget(self.unsharp_slider)
        self.unsharp_slider.valueChanged.connect(lambda value: self.apply_unsharp_mask(value / 10, unsharp_label))

        # --- Laplacian Filter ---
        laplacian_label = QLabel("Laplacian Filter - Kernel Size: 3")
        laplacian_label.setAlignment(Qt.AlignCenter)
        self.laplacian_slider = QSlider(Qt.Horizontal)
        self.laplacian_slider.setRange(1, 7)
        self.laplacian_slider.setSingleStep(2)
        self.laplacian_slider.setValue(3)
        layout.addWidget(laplacian_label)
        layout.addWidget(self.laplacian_slider)
        self.laplacian_slider.valueChanged.connect(lambda value: self.apply_laplacian_filter(value, laplacian_label))

        # --- Sobel Filter ---
        sobel_label = QLabel("Sobel Filter - Kernel Size: 3")
        sobel_label.setAlignment(Qt.AlignCenter)
        self.sobel_slider = QSlider(Qt.Horizontal)
        self.sobel_slider.setRange(3, 15)
        self.sobel_slider.setSingleStep(2)
        self.sobel_slider.setValue(3)
        layout.addWidget(sobel_label)
        layout.addWidget(self.sobel_slider)
        self.sobel_slider.valueChanged.connect(lambda value: self.apply_sobel_filter(value, sobel_label))
            
        # --- Contour Detection Sliders ---
        contour_label = QLabel("Contour Detection")
        contour_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(contour_label)

        # Threshold slider for contouring
        self.contour_threshold_label = QLabel("Threshold Value: 127")
        self.contour_threshold_slider = QSlider(Qt.Horizontal)
        self.contour_threshold_slider.setRange(0, 255)
        self.contour_threshold_slider.setValue(127)
        layout.addWidget(self.contour_threshold_label)
        layout.addWidget(self.contour_threshold_slider)

        # Connect slider to the contour update method
        self.contour_threshold_slider.valueChanged.connect(self.update_contours)

        # --- Canny Edge Detection ---
        canny_label = QLabel("Canny Edge Detection")
        canny_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(canny_label)

        # Lower threshold slider
        self.canny_lower_label = QLabel("Lower Threshold: 100")
        self.canny_lower_slider = QSlider(Qt.Horizontal)
        self.canny_lower_slider.setRange(0, 255)
        self.canny_lower_slider.setValue(100)
        layout.addWidget(self.canny_lower_label)
        layout.addWidget(self.canny_lower_slider)

        # Upper threshold slider
        self.canny_upper_label = QLabel("Upper Threshold: 200")
        self.canny_upper_slider = QSlider(Qt.Horizontal)
        self.canny_upper_slider.setRange(0, 255)
        self.canny_upper_slider.setValue(200)
        layout.addWidget(self.canny_upper_label)
        layout.addWidget(self.canny_upper_slider)

        # Connect Canny sliders to the processing function
        self.canny_lower_slider.valueChanged.connect(self.apply_canny_edge_detection)
        self.canny_upper_slider.valueChanged.connect(self.apply_canny_edge_detection)

        # --- Prewitt Edge Detection ---
        prewitt_label = QLabel("Prewitt Edge Detection")
        prewitt_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(prewitt_label)

        # Prewitt edge detection is static, no slider needed
        prewitt_button = QPushButton("Apply Prewitt Edge Detection")
        prewitt_button.clicked.connect(self.apply_prewitt_edge_detection)
        layout.addWidget(prewitt_button)

        # --- Sobel Edge Detection ---
        self.sobel_label_title = QLabel("Sobel Edge Detection")
        self.sobel_label_title.setAlignment(Qt.AlignCenter)
        self.sobel_label = QLabel("Sobel Kernel Size: 3")
        self.sobel_slider = QSlider(Qt.Horizontal)
        self.sobel_slider.setRange(3, 15)
        self.sobel_slider.setSingleStep(2)
        self.sobel_slider.setValue(3)
        layout.addWidget(self.sobel_label_title)
        layout.addWidget(self.sobel_label)
        layout.addWidget(self.sobel_slider)

        # Connect Sobel slider to the processing function
        self.sobel_slider.valueChanged.connect(self.apply_sobel_edge_detection)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def create_page3(self):
        """Create and return the third page with 3D visualization."""
        page = QWidget()
        layout = QVBoxLayout()

        label = QLabel("Page 3")
        label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(label)

        # Add button to show 3D representation
        self.show_3d_button = QPushButton("Show 3D Representation")
        self.show_3d_button.clicked.connect(self.show_3d_representation)
        layout.addWidget(self.show_3d_button)

        self.figure_3d = plt.figure()
        self.representation_canvas = FigureCanvas(self.figure_3d)
        layout.addWidget(self.representation_canvas)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw the background panels
        painter.fillRect(self.background, QColor("black"))
        painter.fillRect(self.left_panel_rect, QColor(220, 220, 220))
        painter.fillRect(self.right_panel_rect, QColor(220, 220, 220))
        painter.fillRect(self.top_panel_rect, QColor(200, 200, 200))

        # Set the clipping region to restrict drawing to the canvas
        canvas_clip_rect = QRect(self.canvas_x, self.canvas_y, self.width, self.height)
        painter.setClipRect(canvas_clip_rect)

        # Draw the static canvas
        painter.drawPixmap(self.canvas_x, self.canvas_y, self.pixmap)

        # Draw all images
        for index, img_data in enumerate(self.images):
            painter.drawPixmap(img_data['x'], img_data['y'], img_data['pixmap'])
            # Highlight if selected
            if index == self.selected_image_index:
                painter.setPen(QColor("blue"))
                painter.drawRect(img_data['x'], img_data['y'], img_data['pixmap'].width(), img_data['pixmap'].height())

        for i, img_data in enumerate(self.images):
            painter.drawPixmap(img_data['x'], img_data['y'], img_data['pixmap'])

            # Highlight selected image
            if i == self.selected_image_index:
                painter.setPen(QColor("blue"))
                painter.drawRect(img_data['rect'])
                
            if i == self.selected_image_index:
                # Draw anchor points
                rect = img_data['rect']
                anchors = [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]
                painter.setBrush(QColor("blue"))
                for anchor in anchors:
                    painter.drawEllipse(anchor, 5, 5)
                    
        if self.crop_mode and self.crop_rect:
            painter.setPen(QPen(Qt.DashLine))
            painter.setBrush(QColor(255, 255, 255, 50))  # Semi-transparent
            painter.drawRect(self.crop_rect)
                    
        # Draw the overlay (drawings)
        painter.drawPixmap(self.canvas_x, self.canvas_y, self.overlay_pixmap)

        # Render text from the TextTool
        self.text_tool.render_text(painter)
        
        # Draw grid and ruler
        self.grid_tool.draw_grid(painter)
        self.grid_tool.draw_ruler(painter)
        
        self.update_mini_canvas()
    
        painter.end()
        
    def export_as_png(self):
        """Export the current canvas as a PNG image."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Export as PNG", "", "PNG Files (*.png);;All Files (*)")
        if file_path:
            if not file_path.endswith(".png"):
                file_path += ".png"

            # Create a temporary QPixmap to draw the combined state
            export_pixmap = QPixmap(self.pixmap.size())
            export_pixmap.fill(Qt.white)  # Ensure the background is white

            painter = QPainter(export_pixmap)
            painter.drawPixmap(0, 0, self.pixmap)  # Draw the main canvas

            # Draw all imported images onto the export_pixmap
            for img in self.images:
                painter.drawPixmap(img['x'], img['y'], img['pixmap'])

            # Draw any additional overlay elements (shapes, lines)
            painter.drawPixmap(0, 0, self.overlay_pixmap)

            # Draw text objects
            for text_obj in self.text_tool.text_objects:
                font = QFont()
                font.setFamily(text_obj['font']['family'])
                font.setPointSize(text_obj['font']['size'])
                font.setBold(text_obj['font']['bold'])
                font.setItalic(text_obj['font']['italic'])
                painter.setFont(font)
                painter.setPen(QColor(text_obj['color']))
                painter.setOpacity(text_obj['opacity'])
                painter.drawText(text_obj['position'], text_obj['text'])

            painter.end()

            # Save the combined pixmap to the specified file
            if export_pixmap.save(file_path, "PNG"):
                QMessageBox.information(self, "Export Successful", f"Image saved to {file_path}")
            else:
                QMessageBox.warning(self, "Export Failed", "Could not save the image.")

                    
    def save_project(self):
        """Save the current project to a file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Adli Files (*.adli);;All Files (*)")
        if file_path:
            if not file_path.endswith(".adli"):
                file_path += ".adli"

            try:
                # Create a temporary folder to save images
                project_folder = os.path.join(os.path.dirname(file_path), "project_resources")
                os.makedirs(project_folder, exist_ok=True)

                # Save the main canvas as an image
                canvas_image_path = os.path.join(project_folder, "canvas.png")
                self.pixmap.save(canvas_image_path, "PNG")

                # Save the overlay pixmap
                overlay_image_path = os.path.join(project_folder, "overlay.png")
                self.overlay_pixmap.save(overlay_image_path, "PNG")

                # Save individual images
                image_data = []
                for i, img in enumerate(self.images):
                    image_path = os.path.join(project_folder, f"image_{i}.png")
                    img['pixmap'].save(image_path, "PNG")
                    image_data.append({
                        'path': image_path,
                        'x': img['x'],
                        'y': img['y'],
                        'rect': img['rect'],
                    })

                # Serialize text objects with font attributes
                serialized_text_objects = [
                    {
                        'position': text_obj['position'],
                        'text': text_obj['text'],
                        'font': {
                            'family': text_obj['font'].family(),
                            'size': text_obj['font'].pointSize(),
                            'bold': text_obj['font'].bold(),
                            'italic': text_obj['font'].italic(),
                        },
                        'color': text_obj['color'],
                        'opacity': text_obj['opacity'],
                    }
                    for text_obj in self.text_tool.text_objects
                ]

                # Save project metadata
                project_data = {
                    'canvas_image_path': canvas_image_path,
                    'overlay_image_path': overlay_image_path,
                    'images': image_data,
                    'text_objects': serialized_text_objects,  # Save serialized text objects
                }

                with open(file_path, 'wb') as file:
                    pickle.dump(project_data, file)

                QMessageBox.information(self, "Save Successful", f"Project saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"An error occurred while saving: {e}")

    def open_project(self):
        """Open a saved project file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Adli Files (*.adli);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'rb') as file:
                    project_data = pickle.load(file)

                # Restore the main canvas
                canvas_image = QImage(project_data['canvas_image_path'])
                self.pixmap = QPixmap.fromImage(canvas_image)

                # Restore the overlay
                overlay_image = QImage(project_data['overlay_image_path'])
                self.overlay_pixmap = QPixmap.fromImage(overlay_image)

                # Restore imported images
                self.images = []
                for img_data in project_data['images']:
                    pixmap = QPixmap(img_data['path'])
                    self.images.append({
                        'pixmap': pixmap,
                        'x': img_data['x'],
                        'y': img_data['y'],
                        'rect': img_data['rect'],
                    })

                # Restore text objects with fonts
                self.text_tool.text_objects = [
                    {
                        'position': text_obj['position'],
                        'text': text_obj['text'],
                        'font': QFont(
                            text_obj['font']['family'],
                            text_obj['font']['size'],
                            QFont.Bold if text_obj['font']['bold'] else QFont.Normal,
                            text_obj['font']['italic']
                        ),
                        'color': text_obj['color'],
                        'opacity': text_obj['opacity'],
                    }
                    for text_obj in project_data['text_objects']
                ]

                # Update the canvas
                self.update()

                QMessageBox.information(self, "Open Successful", f"Project loaded from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Open Failed", f"An error occurred while opening: {e}")


    def save_state(self):
        """Save the current state to the undo stack."""
        state = {
            'images': [  # Save the images
                {
                    'pixmap': img['pixmap'].copy(),
                    'original_pixmap': img['original_pixmap'].copy(),
                    'x': img['x'],
                    'y': img['y'],
                    'rect': QRect(img['rect']),
                }
                for img in self.images
            ],
            'selected_image_index': self.selected_image_index,
            'pixmap': self.pixmap.copy(),  # Save the current canvas pixmap
            'overlay_pixmap': self.overlay_pixmap.copy(),  # Save the overlay layer
            'text_objects': [  # Save text objects
            {
                'position': text_obj['position'],
                'text': text_obj['text'],
                'font': text_obj['font'],
                'color': text_obj['color'],
                'opacity': text_obj['opacity'],
            }
            for text_obj in self.text_tool.text_objects
        ],
            # Save slider values
        'threshold_value': self.threshold_slider.value(),
        'sharp_value': self.sharp_slider.value(),
        'gamma_value': self.gamma_slider.value()
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()
        self.update_histogram()

        # Clear redo stack only if not in drag/scale mode
        if not self.drag_mode and not self.anchor_clicked and not self.drawing_mode:
            self.redo_stack.clear()

    def undo_action(self):
        if self.undo_stack:
            # Save the current state to the redo stack
            current_state = {
                'images': [  # Save the images
                    {
                        'pixmap': img['pixmap'].copy(),
                        'original_pixmap': img['original_pixmap'].copy(),
                        'x': img['x'],
                        'y': img['y'],
                        'rect': QRect(img['rect']),
                    }
                    for img in self.images
                ],
                'selected_image_index': self.selected_image_index,
                'pixmap': self.pixmap.copy(),  # Save the current canvas pixmap
                'overlay_pixmap': self.overlay_pixmap.copy(),  # Save the overlay layer
                'text_objects': [
                {
                    'position': text_obj['position'],
                    'text': text_obj['text'],
                    'font': text_obj['font'],
                    'color': text_obj['color'],
                    'opacity': text_obj['opacity'],
                }
                for text_obj in self.text_tool.text_objects
            ],
                # Save slider values
            'threshold_value': self.threshold_slider.value(),
            'sharp_value': self.sharp_slider.value(),
            'gamma_value': self.gamma_slider.value()
            }
            self.redo_stack.append(current_state)

            # Restore the last state from the undo stack
            last_state = self.undo_stack.pop()
            self.images = last_state['images']
            self.selected_image_index = last_state['selected_image_index']
            self.pixmap = last_state['pixmap']  # Restore the pixmap
            self.overlay_pixmap = last_state['overlay_pixmap']  # Restore the overlay layer
            self.text_tool.text_objects = last_state['text_objects']  # Restore text objects
            
            # Restore slider values
            self.threshold_slider.setValue(last_state.get('threshold_value', 127))
            self.sharp_slider.setValue(last_state.get('sharp_value', 0))
            self.gamma_slider.setValue(last_state.get('gamma_value', 10))

            self.update()
            self.update_histogram()

    def redo_action(self):
        if self.redo_stack:
            # Save the current state to the undo stack
            current_state = {
                'images': [
                    {
                        'pixmap': img['pixmap'].copy(),
                        'original_pixmap': img['original_pixmap'].copy(),
                        'x': img['x'],
                        'y': img['y'],
                        'rect': QRect(img['rect']),
                    }
                    for img in self.images
                ],
                'selected_image_index': self.selected_image_index,
                'pixmap': self.pixmap.copy(),
                'overlay_pixmap': self.overlay_pixmap.copy(),
                'text_objects': [
                {
                    'position': text_obj['position'],
                    'text': text_obj['text'],
                    'font': text_obj['font'],
                    'color': text_obj['color'],
                    'opacity': text_obj['opacity'],
                }
                for text_obj in self.text_tool.text_objects
            ],
                # Save slider values
            'threshold_value': self.threshold_slider.value(),
            'sharp_value': self.sharp_slider.value(),
            'gamma_value': self.gamma_slider.value()
            }
            self.undo_stack.append(current_state)

            # Restore the next state from the redo stack
            next_state = self.redo_stack.pop()
            self.images = next_state['images']
            self.selected_image_index = next_state['selected_image_index']
            self.pixmap = next_state['pixmap']  # Restore the pixmap
            self.overlay_pixmap = next_state['overlay_pixmap']
            self.text_tool.text_objects = next_state['text_objects']  # Restore text objects
            
            # Restore slider values
            self.threshold_slider.setValue(next_state.get('threshold_value', 127))
            self.sharp_slider.setValue(next_state.get('sharp_value', 0))
            self.gamma_slider.setValue(next_state.get('gamma_value', 10))
            
            self.update()
            self.update_histogram()


    def import_image(self):
        self.save_state()  # Save state for undo functionality
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # Load and convert the image
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "Error", "Could not load the image.")
                return

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # Scale and center the image
            if w > self.width or h > self.height:
                pixmap = pixmap.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            x = self.canvas_x + (self.width - pixmap.width()) // 2
            y = self.canvas_y + (self.height - pixmap.height()) // 2
            
            

            # Add to the images list with both 'pixmap' and 'original_pixmap'
            self.images.append({
                'pixmap': pixmap,
                'original_pixmap': pixmap.copy(),  # Use .copy() to avoid references
                'x': x,
                'y': y,
                'rect': QRect(x, y, pixmap.width(), pixmap.height())
            })
            
            self.gamma_slider.setValue(10)

            self.update()


    def add_background_panels(self):
        self.background = QRect(0, 0, 1950, 980)
        self.left_panel_rect = QRect(0, 60, 80, 980)
        self.right_panel_rect = QRect(1400, 120, 500, 880)
        self.top_panel_rect = QRect(0, 0, 1950, 50)

    def toggle_drag_mode(self):
        self.drag_mode = not self.drag_mode
        if self.drag_mode:
            self.drag_button.setStyleSheet("""
                QPushButton {
                    background-color: green;
                    color: white;
                    border-radius: 5px;
                    border: 1px solid black;
                }
            """)
            self.statusBar().showMessage("Drag mode enabled", 3000)
        else:
            self.drag_button.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                    border-radius: 5px;
                    border: 1px solid black;
                }
                QPushButton:hover {
                    background-color: darkgray;
                }
                QPushButton:pressed {
                    background-color: gray;
                }
            """)
            self.statusBar().showMessage("Drag mode disabled", 3000)

        
    def start_drag(self, event):
        """Start the dragging process."""
        self.image_selected = True
        self.last_mouse_position = event.pos()

    def perform_drag(self, event):
        """Handle dragging of the image."""
        delta = event.pos() - self.last_mouse_position  # Calculate movement
        if self.image_pixmap and self.image_selected:  # Ensure image exists and is selected
            # Update the image position
            self.image_x += delta.x()
            self.image_y += delta.y()

            # Update the bounding rectangle
            self.image_rect.moveTo(self.image_x, self.image_y)

        self.last_mouse_position = event.pos()  # Update the last mouse position
        self.update()  # Redraw the canvas

    def end_drag(self):
        """End the dragging process."""
        self.image_selected = False  # Reset selection state
        
    def get_clicked_anchor(self, pos, rect):
        """Detect if a specific anchor point is clicked."""
        anchors = {
            "top-left": rect.topLeft(),
            "top-right": rect.topRight(),
            "bottom-left": rect.bottomLeft(),
            "bottom-right": rect.bottomRight(),
        }
        for name, anchor in anchors.items():
            if QRect(anchor.x() - 5, anchor.y() - 5, 10, 10).contains(pos):
                return name
        return None

    def scale_image(self, pos, img_data):
        """Scale the image dynamically based on anchor and mouse movement."""
        rect = img_data['rect']
        anchor = self.anchor_clicked

        # Update the rectangle based on the anchor point
        if anchor == "top-left":
            rect.setTopLeft(pos)
        elif anchor == "top-right":
            rect.setTopRight(pos)
        elif anchor == "bottom-left":
            rect.setBottomLeft(pos)
        elif anchor == "bottom-right":
            rect.setBottomRight(pos)

        # Calculate new dimensions
        new_width = rect.width()
        new_height = rect.height()

        # Prevent too small dimensions
        if new_width > 10 and new_height > 10:
            # Enforce aspect ratio if needed
            aspect_ratio = img_data['original_pixmap'].width() / img_data['original_pixmap'].height()
            if anchor in ["top-left", "top-right", "bottom-left", "bottom-right"]:
                new_height = int(new_width / aspect_ratio)

            # Scale the image
            img_data['pixmap'] = img_data['original_pixmap'].scaled(
                new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            # Update the rectangle
            img_data['rect'] = QRect(rect.topLeft(), QSize(new_width, new_height))

        self.update()
        
    def toggle_scale_mode(self):
        self.scale_mode = not getattr(self, "scale_mode", False)  # Toggle scale_mode
        if self.scale_mode:
            self.scale_button.setStyleSheet("""
                QPushButton {
                    background-color: blue;
                    color: white;
                    border-radius: 5px;
                    border: 1px solid black;
                }
            """)
            self.drag_mode = False  # Disable drag mode if scaling mode is enabled
            self.drag_button.setChecked(False)
            self.statusBar().showMessage("Scaling mode enabled", 3000)
        else:
            self.scale_button.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                    border-radius: 5px;
                    border: 1px solid black;
                }
                QPushButton:hover {
                    background-color: darkgray;
                }
                QPushButton:pressed {
                    background-color: gray;
                }
            """)
            self.statusBar().showMessage("Scaling mode disabled", 3000)
            
    def toggle_drawing_mode(self):
        self.drawing_mode = not self.drawing_mode
        self.save_state()
        if self.drawing_mode:
            self.draw_button.setStyleSheet("""
                QPushButton {
                    background-color: orange;
                    color: white;
                    border-radius: 5px;
                    border: 1px solid black;
                }
            """)
            self.drag_mode = False
            self.scale_mode = False
            self.drag_button.setChecked(False)
            self.scale_button.setChecked(False)
            self.statusBar().showMessage("Drawing mode enabled", 3000)
        else:
            self.draw_button.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                    border-radius: 5px;
                    border: 1px solid black;
                }
                QPushButton:hover {
                    background-color: darkgray;
                }
                QPushButton:pressed {
                    background-color: gray;
                }
            """)
            self.statusBar().showMessage("Drawing mode disabled", 3000)
            
    def change_pen_type(self, pen_type):
        """Update the drawing tool based on selected pen type."""
        if pen_type == "Pencil":
            self.drawing_tool.set_pen_settings(color=Qt.black, width=1, opacity=1.0)
        elif pen_type == "Brush":
            self.drawing_tool.set_pen_settings(color=Qt.black, width=5, opacity=0.8)
        elif pen_type == "Highlighter":
            self.drawing_tool.set_pen_settings(color=Qt.yellow, width=10, opacity=0.5)
        elif pen_type == "Marker":
            self.drawing_tool.set_pen_settings(color=Qt.red, width=8, opacity=1.0)
        elif pen_type == "Calligraphy":
            self.drawing_tool.set_pen_settings(color=Qt.black, width=6, opacity=0.9, style=Qt.SolidLine)
        elif pen_type == "ShapeTool":
            self.shape_tool.set_pen_settings(color=Qt.blue, width=3, opacity=0.8, style=Qt.DashLine)
            
    def choose_color(self):
        """Open a color dialog to choose the pen color."""
        color = QColorDialog.getColor(initial=self.drawing_tool.pen_color, parent=self)
        if color.isValid():  # Check if the user selected a color
            self.drawing_tool.set_pen_settings(
                color=color,
                width=self.drawing_tool.pen_width,
                opacity=self.drawing_tool.pen_opacity,
                style=self.drawing_tool.pen_style,
            )
            self.shape_tool.set_pen_settings(
                color=color,
                width=self.shape_tool.pen_width,
                opacity=self.shape_tool.pen_opacity,
                style=self.shape_tool.pen_style,
            )
            
    def update_value_box(self, value):
        """Update the value box when the slider changes."""
        self.size_value_box.setText(str(value))
        self.drawing_tool.pen_width = value  # Update the pen width dynamically
        self.shape_tool.pen_width = value 

    def update_slider(self, value):
        """Update the slider when the value box changes."""
        if value.isdigit():  # Ensure valid input
            pen_size = int(value)
            self.size_slider.setValue(pen_size)
            self.drawing_tool.pen_width = pen_size  # Update the pen width
            self.shape_tool.pen_width = pen_size

    def set_pen_settings(self, color=Qt.black, width=1, opacity=1.0, style=Qt.SolidLine):
        """Update the pen settings."""
        self.pen_color = color
        self.pen_width = width  # Use the slider's value here
        self.pen_opacity = opacity
        self.pen_style = style
        
    def update_opacity_value_box(self, value):
        """Update the opacity value box when the slider changes."""
        self.opacity_value_box.setText(str(value))
        self.drawing_tool.pen_opacity = value / 100.0  # Update opacity dynamically (0.0 to 1.0)
        self.shape_tool.pen_opacity = value / 100.0

    def update_opacity_slider(self, value):
        """Update the opacity slider when the value box changes."""
        if value.isdigit():  # Ensure valid input
            opacity = int(value)
            self.opacity_slider.setValue(opacity)
            self.drawing_tool.pen_opacity = opacity / 100.0  # Update opacity dynamically
            self.shape_tool.pen_opacity = opacity / 100.0

    def toggle_eraser_mode(self):
        self.eraser_mode = not self.eraser_mode
        self.drawing_mode = False  # Disable drawing mode
        self.save_state()
        print(f"Eraser mode: {self.eraser_mode}, Drawing mode: {self.drawing_mode}")
        if self.eraser_mode:
            self.eraser_button.setStyleSheet("""
                QPushButton {
                    background-color: red;
                    color: white;
                    border-radius: 5px;
                    border: 1px solid black;
                }
            """)
            self.drag_mode = False
            self.scale_mode = False
            self.drag_button.setChecked(False)
            self.scale_button.setChecked(False)
            self.statusBar().showMessage("Drawing mode enabled", 3000)
        else:
            self.eraser_button.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                    border-radius: 5px;
                    border: 1px solid black;
                }
                QPushButton:hover {
                    background-color: darkgray;
                }
                QPushButton:pressed {
                    background-color: gray;
                }
            """)
            self.statusBar().showMessage("eraser mode disabled", 3000)
            
    def toggle_shape_mode(self):
        self.shape_mode = not self.shape_mode
        self.drawing_mode = False
        self.eraser_mode = False
        self.save_state()

        if self.shape_mode:
            self.shape_button.setStyleSheet("""
                QPushButton {
                    background-color: yellow;
                    color: white;
                    border-radius: 5px;
                    border: 1px solid black;
                }
            """)
            self.statusBar().showMessage("Shape mode enabled", 3000)
        else:
            self.shape_button.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                    border-radius: 5px;
                    border: 1px solid black;
                }
                QPushButton:hover {
                    background-color: darkgray;
                }
                QPushButton:pressed {
                    background-color: gray;
                }
            """)
            self.statusBar().showMessage("Shape mode disabled", 3000)
            
    def setup_shape_dropdown(self):
        """Create and configure the shape dropdown menu."""
        self.shape_dropdown = QComboBox(self)
        self.shape_dropdown.setGeometry(930, 35, 120, 30)  # Position next to the shape button

        # Add shape options with icons
        shapes = [
            ("Rectangle", "rectangle.png"),
            ("Ellipse", "ellipse.png"),
            ("Circle", "circle.png"),
            ("Square", "square.png"),
            ("Triangle", "triangle.png"),
            ("Line", "line.png"),
            ("DashLine", "dashline.png"),
        ]

        for shape_name, icon_path in shapes:
            icon = QIcon(icon_path)
            self.shape_dropdown.addItem(icon, shape_name)

        # Connect dropdown selection to shape tool
        self.shape_dropdown.currentTextChanged.connect(self.update_shape_type)

    def update_shape_type(self, shape_name):
        """Update the shape type in the ShapeTool based on dropdown selection."""
        self.shape_tool.set_shape_type(shape_name)
        
    def toggle_text_mode(self):
        self.text_mode = not self.text_mode
        self.drawing_mode = False
        self.eraser_mode = False
        self.shape_mode = False
        self.save_state()

        if self.text_mode:
            self.text_button.setStyleSheet("""
                QPushButton {
                    background-color: purple;
                    color: white;
                    border-radius: 5px;
                    border: 1px solid black;
                }
            """)
            self.statusBar().showMessage("Text mode enabled", 3000)
        else:
            self.text_button.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                    border-radius: 5px;
                    border: 1px solid black;
                }
                QPushButton:hover {
                    background-color: darkgray;
                }
                QPushButton:pressed {
                    background-color: gray;
                }
            """)
            self.statusBar().showMessage("Text mode disabled", 3000)
            
    def update_text_settings(self, color=None, font_name="Arial", size=14, opacity=1.0):
        """Update the settings for the text tool."""
        color = color or self.text_tool.pen_color
        font_name = font_name or self.text_tool.font.family()
        size = size or self.text_tool.font.pointSize()
        opacity = opacity or self.text_tool.pen_opacity

        self.text_tool.set_text_settings(color=color, font_name=font_name, size=size, opacity=opacity)
        
    def choose_text_color(self):
        """Open a color dialog to choose the text color."""
        color = QColorDialog.getColor(initial=self.text_tool.pen_color, parent=self)
        if color.isValid():
            self.update_text_settings(color=color)

    def change_text_size(self, size):
        """Update text size from a slider or input box."""
        self.update_text_settings(size=size)

    def change_text_opacity(self, value):
        """Update text opacity dynamically."""
        opacity = value / 100.0  # Convert from 0-100 slider to 0.0-1.0
        self.update_text_settings(opacity=opacity)
        
    def populate_fonts(self):
        """Populate the font dropdown with available system fonts."""
        font_database = QFontDatabase()
        fonts = font_database.families()
        self.font_dropdown.addItems(fonts)
        
    def set_font(self):
        """Set the selected font for the TextTool."""
        selected_font = self.font_dropdown.currentText()
        font = QFont(selected_font, self.size_slider.value())  # Default size
        self.text_tool.current_font = font  # Update the TextTool's font
        
    def toggle_crop_mode(self):
        """Toggle crop mode."""
        self.crop_mode = not self.crop_mode
        if self.crop_mode:
            self.crop_button.setStyleSheet("""
                QPushButton {
                    background-color: cyan;
                    color: white;
                    border-radius: 5px;
                    border: 1px solid black;
                }
            """)
            self.statusBar().showMessage("Crop mode enabled", 3000)
        else:
            self.crop_button.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                    border-radius: 5px;
                    border: 1px solid black;
                }
                QPushButton:hover {
                    background-color: darkgray;
                }
                QPushButton:pressed {
                    background-color: gray;
                }
            """)
            self.statusBar().showMessage("Crop mode disabled", 3000)

    def perform_crop(self):
        if self.selected_image_index is None or not self.crop_rect:
            return
        
        # Save the current state before cropping
        self.save_state()
        
        img_data = self.images[self.selected_image_index]
        pixmap = img_data['pixmap']
        rect = self.crop_rect.translated(-img_data['x'], -img_data['y'])  # Adjust for image position
        cropped_pixmap = pixmap.copy(rect)
        img_data['pixmap'] = cropped_pixmap
        img_data['rect'] = QRect(img_data['x'], img_data['y'], cropped_pixmap.width(), cropped_pixmap.height())
        
    def delete_selected_image(self):
        if self.selected_image_index is not None and 0 <= self.selected_image_index < len(self.images):
            self.save_state()  # Save the state for undo functionality
            del self.images[self.selected_image_index]  # Remove the selected image
            self.selected_image_index = None  # Reset the selected index
            self.update()  # Refresh the canvas
        else:
            QMessageBox.warning(self, "Warning", "No image selected for deletion.")
            
    def show_mini_canvas(self):
        if not hasattr(self, 'mini_canvas_window') or self.mini_canvas_window is None:
            self.mini_canvas_window = MiniCanvasWindow(self)
            self.mini_canvas_window.show()

        # Update the mini canvas with the current main canvas state
        self.update_mini_canvas()

    def update_mini_canvas(self):
        if hasattr(self, 'mini_canvas_window') and self.mini_canvas_window.isVisible():
            # Create a pixmap of the current canvas state
            canvas_pixmap = QPixmap(self.width, self.height)
            self.render(canvas_pixmap)
            self.mini_canvas_window.update_canvas(canvas_pixmap)
            
    def flip_horizontal(self):
        if self.selected_image_index is not None and 0 <= self.selected_image_index < len(self.images):
            self.save_state()  # Save state for undo functionality
            img_data = self.images[self.selected_image_index]
            img_data['pixmap'] = img_data['pixmap'].transformed(QTransform().scale(-1, 1))  # Flip horizontally
            self.update()  # Refresh the canvas
        else:
            QMessageBox.warning(self, "Warning", "No image selected to flip horizontally.")
            
    def flip_vertical(self):
        if self.selected_image_index is not None and 0 <= self.selected_image_index < len(self.images):
            self.save_state()  # Save state for undo functionality
            img_data = self.images[self.selected_image_index]
            img_data['pixmap'] = img_data['pixmap'].transformed(QTransform().scale(1, -1))  # Flip vertically
            self.update()  # Refresh the canvas
        else:
            QMessageBox.warning(self, "Warning", "No image selected to flip vertically.")
            
    def zoom_in(self):
        self.scale_canvas(1.1)  # Scale up by 10%

    def zoom_out(self):
        self.scale_canvas(0.9)  # Scale down by 10%

    def reset_zoom(self):
        self.scale_canvas(1, reset=True)  # Reset to original scale

    def scale_canvas(self, factor, reset=False):
        if reset:
            self.current_scale_factor = 1.0
        else:
            self.current_scale_factor *= factor

        # Scale images
        for img in self.images:
            original_pixmap = img['original_pixmap']
            scaled_pixmap = original_pixmap.scaled(
                int(original_pixmap.width() * self.current_scale_factor),
                int(original_pixmap.height() * self.current_scale_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            img['pixmap'] = scaled_pixmap
            img['x'] = self.canvas_x + (self.width - scaled_pixmap.width()) // 2
            img['y'] = self.canvas_y + (self.height - scaled_pixmap.height()) // 2
            img['rect'] = QRect(img['x'], img['y'], scaled_pixmap.width(), scaled_pixmap.height())

        # Scale the drawing canvas
        self.pixmap = self.pixmap.scaled(
            int(self.width * self.current_scale_factor),
            int(self.height * self.current_scale_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Scale overlay pixmap (for drawings, shapes, etc.)
        self.overlay_pixmap = self.overlay_pixmap.scaled(
            int(self.width * self.current_scale_factor),
            int(self.height * self.current_scale_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Update the canvas
        self.update()
        self.update_histogram()
            
    def update_histogram(self):
        """Update the RGB histogram based on the selected channels."""
        if self.selected_image_index is None or not self.images:
            print("No image selected or no images loaded.")
            return

        # Get the selected image
        selected_image = self.images[self.selected_image_index]['pixmap']

        # Convert QPixmap to QImage
        q_image = selected_image.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB888)

        # Get dimensions and stride
        width, height = q_image.width(), q_image.height()
        stride = q_image.bytesPerLine()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())

        # Convert the QImage to a numpy array while handling the stride
        img_array = np.array(ptr).reshape((height, stride))[:, :width * 3]
        img_array = img_array.reshape((height, width, 3))

        # Clear previous plot on the histogram figure
        self.figure_histogram.clear()
        ax = self.figure_histogram.add_subplot(111)

        # Plot histograms for selected channels
        if self.check_red.isChecked():
            hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
            ax.plot(hist, color='r', label='Red')
        if self.check_green.isChecked():
            hist = cv2.calcHist([img_array], [1], None, [256], [0, 256])
            ax.plot(hist, color='g', label='Green')
        if self.check_blue.isChecked():
            hist = cv2.calcHist([img_array], [2], None, [256], [0, 256])
            ax.plot(hist, color='b', label='Blue')

        ax.set_xlim([0, 256])
        ax.legend()
        self.histogram_canvas.draw()

    def resize_pixmap(self, source_pixmap, target_size):
        """Resize the source QPixmap to the target size."""
        return source_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        
    def update_powerlaw_image(self):
        if self.selected_image_index is None or not self.images:
            return

        self.save_state()

        # Get the original image and current size
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert original_pixmap to OpenCV format
        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB32)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        cv_image = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # Extract RGB channels

        # Get gamma value from the slider
        gamma_value = self.gamma_slider.value() / 10.0
        self.gamma_label.setText(f"Gamma: {gamma_value:.1f}")

        # Apply gamma correction
        inv_gamma = 1.0 / gamma_value
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        gamma_corrected_image = cv2.LUT(cv_image, table)

        # Convert back to QPixmap
        gamma_corrected_image = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB)
        h, w, ch = gamma_corrected_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(gamma_corrected_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_pixmap = QPixmap.fromImage(qt_image)

        # Resize to match the current cropped and scaled size
        resized_pixmap = self.resize_pixmap(new_pixmap, current_size)

        # Update the displayed pixmap
        selected_image['pixmap'] = resized_pixmap

        self.update_histogram()
        self.update()


        
    def bit_plane_slicing(self, bit):
        if self.selected_image_index is None or not self.images:
            QMessageBox.warning(self, "No Image", "Please load and select an image first.")
            return

        self.save_state()

        # Get the original image and current size
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert original_pixmap to OpenCV format
        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB888)

        width, height = q_image.width(), q_image.height()
        stride = q_image.bytesPerLine()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        img_array = np.array(ptr).reshape((height, stride))[:, :width * 3]
        img_array = img_array.reshape((height, width, 3))

        # Separate into RGB channels
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        # Apply bit-plane slicing
        red_bit_plane = ((red_channel >> bit) & 1) * 255
        green_bit_plane = ((green_channel >> bit) & 1) * 255
        blue_bit_plane = ((blue_channel >> bit) & 1) * 255

        # Combine channels back into an RGB image
        sliced_image = np.stack([red_bit_plane, green_bit_plane, blue_bit_plane], axis=-1).astype(np.uint8)

        # Convert back to QPixmap
        h, w, ch = sliced_image.shape
        bytes_per_line = ch * w
        sliced_q_image = QImage(sliced_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_pixmap = QPixmap.fromImage(sliced_q_image)

        # Resize to match the current cropped and scaled size
        resized_pixmap = self.resize_pixmap(new_pixmap, current_size)

        # Update the displayed pixmap
        selected_image['pixmap'] = resized_pixmap

        self.update_histogram()
        self.update()


    def update_sharpening_image(self, sharp_label):
        if self.selected_image_index is None or not self.images:
            return  # No image selected or no images loaded

        self.save_state()  # Save the state for undo before making changes

        # Get the selected image
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert QImage to OpenCV format
        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB32)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        cv_image = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # Extract RGB channels

        # Get sharpening level from slider (convert scaled integer back to float)
        sharp_value = self.sharp_slider.value() / 10.0
        sharp_label.setText(f"Sharpening Strength: {sharp_value:.1f}")

        if sharp_value == 0:
            # Display the original image when sharpening strength is 0
            resized_pixmap = self.resize_pixmap(original_pixmap, current_size)
            self.images[self.selected_image_index]['pixmap'] = resized_pixmap
        else:
            # Create dynamic sharpening kernel
            kernel_size = 3  # Fixed kernel size
            center_value = 1 + (8 * sharp_value)  # Neutral brightness kernel
            kernel = np.ones((kernel_size, kernel_size)) * -sharp_value
            kernel[kernel_size // 2, kernel_size // 2] = center_value

            # Apply sharpening filter
            sharp_image = cv2.filter2D(cv_image, -1, kernel)

            # Convert back to QPixmap
            sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB)
            h, w, ch = sharp_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(sharp_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            sharp_pixmap = QPixmap.fromImage(qt_image)

            # Resize the sharpened image to match the current dimensions
            resized_pixmap = self.resize_pixmap(sharp_pixmap, current_size)

            # Update the selected image's displayed pixmap
            self.images[self.selected_image_index]['pixmap'] = resized_pixmap

        self.update()
        self.update_histogram()

        
    def update_thresholding_image(self, threshold_label):
        if self.selected_image_index is None or not self.images:
            return  # No image selected or no images loaded

        self.save_state()

        # Get the selected image
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert QImage to OpenCV format
        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB32)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        cv_image = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # Extract RGB channels

        # Get threshold value from slider
        threshold_value = self.threshold_slider.value()
        threshold_label.setText(f"Threshold Value: {threshold_value}")

        # Apply thresholding to each channel (R, G, B)
        channels = cv2.split(cv_image)  # Split the image into R, G, B channels
        thresholded_channels = [
            cv2.threshold(channel, threshold_value, 255, cv2.THRESH_BINARY)[1]
            for channel in channels
        ]
        thresholded_image = cv2.merge(thresholded_channels)  # Merge the thresholded channels back

        # Convert back to QPixmap
        thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2RGB)
        h, w, ch = thresholded_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(thresholded_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        threshold_pixmap = QPixmap.fromImage(qt_image)

        # Resize the thresholded image to match the current dimensions
        resized_pixmap = self.resize_pixmap(threshold_pixmap, current_size)

        # Update the selected image's displayed pixmap
        self.images[self.selected_image_index]['pixmap'] = resized_pixmap

        self.update()
        self.update_histogram()
        
    def update_erosion_image(self, erosion_label):
        if self.selected_image_index is None or not self.images:
            return  # No image selected or no images loaded

        self.save_state()

        # Get the original image and current size
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert QImage to OpenCV format
        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB32)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        cv_image = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # Extract RGB channels

        # Get kernel size from slider
        kernel_size = self.erosion_slider.value()
        erosion_label.setText(f"Erosion Kernel Size: {kernel_size}")

        # Apply erosion
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_image = cv2.erode(cv_image, kernel)

        # Convert back to QPixmap
        eroded_image = cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB)
        h, w, ch = eroded_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(eroded_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_pixmap = QPixmap.fromImage(qt_image)

        # Resize to match the current cropped and scaled size
        resized_pixmap = self.resize_pixmap(new_pixmap, current_size)

        # Update the displayed pixmap
        selected_image['pixmap'] = resized_pixmap

        self.update_histogram()
        self.update()


    def update_dilation_image(self, dilation_label):
        if self.selected_image_index is None or not self.images:
            return  # No image selected or no images loaded

        self.save_state()

        # Get the original image and current size
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert QImage to OpenCV format
        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB32)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        cv_image = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # Extract RGB channels

        # Get kernel size from slider
        kernel_size = self.dilation_slider.value()
        dilation_label.setText(f"Dilation Kernel Size: {kernel_size}")

        # Apply dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(cv_image, kernel)

        # Convert back to QPixmap
        dilated_image = cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB)
        h, w, ch = dilated_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(dilated_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_pixmap = QPixmap.fromImage(qt_image)

        # Resize to match the current cropped and scaled size
        resized_pixmap = self.resize_pixmap(new_pixmap, current_size)

        # Update the displayed pixmap
        selected_image['pixmap'] = resized_pixmap

        self.update_histogram()
        self.update()
        
    def apply_canny_edge_detection(self):
        """Apply Canny edge detection dynamically while maintaining image size."""
        if self.selected_image_index is None or not self.images:
            return
        
        self.save_state()

        lower_thresh = self.canny_lower_slider.value()
        upper_thresh = self.canny_upper_slider.value()

        self.canny_lower_label.setText(f"Lower Threshold: {lower_thresh}")
        self.canny_upper_label.setText(f"Upper Threshold: {upper_thresh}")

        # Get the selected image
        selected_image = self.images[self.selected_image_index]['original_pixmap']
        q_image = selected_image.toImage().convertToFormat(QImage.Format_RGB888)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        stride = q_image.bytesPerLine()
        img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, stride))[:, :width * 3].reshape((height, width, 3))

        # Convert to grayscale and apply Canny
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, lower_thresh, upper_thresh)

        # Convert edges to QPixmap and resize
        edges_qimage = QImage(edges.data, edges.shape[1], edges.shape[0], edges.strides[0], QImage.Format_Grayscale8)
        new_pixmap = QPixmap.fromImage(edges_qimage)

        # Resize the pixmap to the current display size
        current_display_size = self.images[self.selected_image_index]['pixmap'].size()
        resized_pixmap = self.resize_pixmap(new_pixmap, current_display_size)

        # Update the displayed pixmap
        self.images[self.selected_image_index]['pixmap'] = resized_pixmap
        self.update()
        self.update_histogram()

    def apply_prewitt_edge_detection(self):
        """Apply Prewitt edge detection dynamically while maintaining image size."""
        if self.selected_image_index is None or not self.images:
            return

        # Get the selected image
        selected_image = self.images[self.selected_image_index]['original_pixmap']
        q_image = selected_image.toImage().convertToFormat(QImage.Format_RGB888)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        stride = q_image.bytesPerLine()
        img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, stride))[:, :width * 3].reshape((height, width, 3))

        # Convert to grayscale
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Prewitt filters
        prewitt_x = cv2.filter2D(gray_image, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
        prewitt_y = cv2.filter2D(gray_image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
        prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

        # Convert combined Prewitt edges to QPixmap
        prewitt_qimage = QImage(prewitt_combined.data, prewitt_combined.shape[1], prewitt_combined.shape[0],
                                prewitt_combined.strides[0], QImage.Format_Grayscale8)
        new_pixmap = QPixmap.fromImage(prewitt_qimage)

        # Resize the pixmap to the current display size
        current_display_size = self.images[self.selected_image_index]['pixmap'].size()
        resized_pixmap = self.resize_pixmap(new_pixmap, current_display_size)

        # Update the displayed pixmap
        self.images[self.selected_image_index]['pixmap'] = resized_pixmap
        self.update()
        self.update_histogram()

    def apply_sobel_edge_detection(self):
        """Apply Sobel edge detection dynamically while maintaining image size."""
        if self.selected_image_index is None or not self.images:
            return
        
        self.save_state()

        kernel_size = self.sobel_slider.value()
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure the kernel size is odd

        self.sobel_label.setText(f"Kernel Size: {kernel_size}")

        # Get the selected image
        selected_image = self.images[self.selected_image_index]['original_pixmap']
        q_image = selected_image.toImage().convertToFormat(QImage.Format_RGB888)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        stride = q_image.bytesPerLine()
        img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, stride))[:, :width * 3].reshape((height, width, 3))

        # Convert to grayscale and apply Sobel
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # Calculate gradient magnitude
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

        # Convert to QPixmap and resize
        sobel_qimage = QImage(sobel_combined.data, sobel_combined.shape[1], sobel_combined.shape[0],
                            sobel_combined.strides[0], QImage.Format_Grayscale8)
        new_pixmap = QPixmap.fromImage(sobel_qimage)

        # Resize the pixmap to the current display size
        current_display_size = self.images[self.selected_image_index]['pixmap'].size()
        resized_pixmap = self.resize_pixmap(new_pixmap, current_display_size)

        # Update the displayed pixmap
        self.images[self.selected_image_index]['pixmap'] = resized_pixmap
        self.update()
        self.update_histogram()

    def update_piecewise_image(self, piecewise_label):
        if self.selected_image_index is None or not self.images:
            return

        self.save_state()

        # Get the selected image and current size
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert QPixmap to OpenCV format
        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB32)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        cv_image = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # Extract RGB channels

        # Get the piecewise point value from the slider
        piecewise_point = self.piecewise_slider.value()
        piecewise_label.setText(f"Piecewise Point: {piecewise_point}")

        # Avoid division by zero by clamping the value
        if piecewise_point == 0:
            piecewise_point = 1
        elif piecewise_point == 255:
            piecewise_point = 254

        # Perform piecewise linear transformation
        lookup_table = np.array([
            min(max((i * 255) // piecewise_point, 0), 255) if i <= piecewise_point
            else min(max(128 + (i - piecewise_point) * 127 // (255 - piecewise_point), 0), 255)
            for i in range(256)
        ]).astype(np.uint8)

        # Apply the lookup table to the image
        transformed_image = cv2.LUT(cv_image, lookup_table)

        # Convert back to QPixmap
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        h, w, ch = transformed_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(transformed_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_pixmap = QPixmap.fromImage(qt_image)

        # Resize to match the current cropped and scaled size
        resized_pixmap = self.resize_pixmap(new_pixmap, current_size)

        # Update the displayed pixmap
        selected_image['pixmap'] = resized_pixmap

        self.update_histogram()
        self.update()

    def apply_histogram_equalization(self):
        """Apply standard histogram equalization to the selected image, preserving the original color."""
        if self.selected_image_index is None or not self.images:
            QMessageBox.warning(self, "No Image", "Please load and select an image first.")
            return

        self.save_state()

        # Get the selected image and convert to OpenCV format
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB888)
        width, height = q_image.width(), q_image.height()
        stride = q_image.bytesPerLine()  # Correctly account for padding
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        color_image = np.array(ptr).reshape((height, stride))[:, :width * 3].reshape((height, width, 3))

        # Apply histogram equalization to each channel
        channels = cv2.split(color_image)
        equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
        equalized_image = cv2.merge(equalized_channels)

        # Convert back to QPixmap
        h, w, ch = equalized_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(equalized_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_pixmap = QPixmap.fromImage(qt_image)

        # Resize and update the displayed image
        resized_pixmap = self.resize_pixmap(new_pixmap, current_size)
        selected_image['pixmap'] = resized_pixmap

        self.update()
        self.update_histogram()

    def apply_clahe(self, clahe_label):
        """Apply adaptive histogram equalization (CLAHE) with adjustable clip limit, preserving the original color."""
        if self.selected_image_index is None or not self.images:
            QMessageBox.warning(self, "No Image", "Please load and select an image first.")
            return

        self.save_state()

        # Get the selected image and convert to OpenCV format
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB888)
        width, height = q_image.width(), q_image.height()
        stride = q_image.bytesPerLine()  # Correctly account for padding
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        color_image = np.array(ptr).reshape((height, stride))[:, :width * 3].reshape((height, width, 3))

        # Get clip limit from slider
        clip_limit = self.clahe_slider.value() / 10.0
        clahe_label.setText(f"CLAHE Clip Limit: {clip_limit:.1f}")

        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        clahe_channels = [clahe.apply(channel) for channel in cv2.split(color_image)]
        clahe_image = cv2.merge(clahe_channels)

        # Convert back to QPixmap
        h, w, ch = clahe_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(clahe_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_pixmap = QPixmap.fromImage(qt_image)

        # Resize and update the displayed image
        resized_pixmap = self.resize_pixmap(new_pixmap, current_size)
        selected_image['pixmap'] = resized_pixmap

        self.update()
        self.update_histogram()
        
    def update_contours(self):
        """Update contours dynamically as the slider is adjusted."""
        if self.selected_image_index is None or not self.images:
            return

        # Get the selected image
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert QPixmap to OpenCV format
        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB32)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        cv_image = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # Extract RGB channels

        # Get threshold value from slider
        threshold_value = self.contour_threshold_slider.value()
        self.contour_threshold_label.setText(f"Threshold Value: {threshold_value}")

        # Convert to grayscale and apply thresholding
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original color image
        contour_image = cv_image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green contours

        # Convert back to QPixmap
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
        h, w, ch = contour_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(contour_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_pixmap = QPixmap.fromImage(qt_image)

        # Resize and update the displayed image
        resized_pixmap = self.resize_pixmap(new_pixmap, current_size)
        selected_image['pixmap'] = resized_pixmap

        self.update()
        self.update_histogram()
        
    def show_3d_representation(self):
        """Display a 3D representation of the selected image."""
        if self.selected_image_index is None or not self.images:
            QMessageBox.warning(self, "Warning", "No image selected.")
            return

        # Get the selected image
        selected_image = self.images[self.selected_image_index]['pixmap']
        image = selected_image.toImage()
        image = image.convertToFormat(QImage.Format.Format_RGB32)

        # Convert QImage to numpy array
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        img_array = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

        # Convert to grayscale
        img_gray = color.rgb2gray(img_array[:, :, :3])

        # Resize for performance
        img_resized = resize(img_gray, (200, 200), anti_aliasing=True)

        # Create meshgrid for X, Y, Z
        x = np.linspace(0, img_resized.shape[1], img_resized.shape[1])
        y = np.linspace(0, img_resized.shape[0], img_resized.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = img_resized

        # Clear previous plot on the 3D figure
        self.figure_3d.clear()
        ax = self.figure_3d.add_subplot(111, projection='3d')

        # Plot 3D surface
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        self.figure_3d.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        ax.set_title("3D Representation of Image Intensity")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Intensity")

        # Refresh the 3D canvas
        self.representation_canvas.draw()
        
    def apply_gaussian_blur(self, kernel_size, label):
        if kernel_size % 2 == 0:
            kernel_size += 1
        label.setText(f"Gaussian Blur - Kernel Size: {kernel_size}")
        if self.selected_image_index is not None:
            image = self.get_current_image()
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            self.update_image_display(blurred)
            
        self.update()
        self.update_histogram()

    def apply_median_filter(self, kernel_size, label):
        if kernel_size % 2 == 0:
            kernel_size += 1
        label.setText(f"Median Filter - Kernel Size: {kernel_size}")
        if self.selected_image_index is not None:
            image = self.get_current_image()
            filtered = cv2.medianBlur(image, kernel_size)
            self.update_image_display(filtered)
            
        self.update()
        self.update_histogram()

    def apply_bilateral_filter(self, diameter, label):
        label.setText(f"Bilateral Filter - Diameter: {diameter}")
        if self.selected_image_index is not None:
            image = self.get_current_image()
            filtered = cv2.bilateralFilter(image, diameter, 75, 75)
            self.update_image_display(filtered)
            
        self.update()
        self.update_histogram()

    def apply_unsharp_mask(self, strength, label):
        label.setText(f"Unsharp Masking - Strength: {strength:.1f}")
        if self.selected_image_index is not None:
            image = self.get_current_image()
            gaussian = cv2.GaussianBlur(image, (9, 9), 10)
            sharpened = cv2.addWeighted(image, 1 + strength, gaussian, -strength, 0)
            self.update_image_display(sharpened)
            
        self.update()
        self.update_histogram()

    def apply_laplacian_filter(self, kernel_size, label):
        if kernel_size % 2 == 0:
            kernel_size += 1
        label.setText(f"Laplacian Filter - Kernel Size: {kernel_size}")
        if self.selected_image_index is not None:
            image = self.get_current_image()
            laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
            result = cv2.convertScaleAbs(laplacian)
            self.update_image_display(result)
        self.update()
        self.update_histogram()

    def apply_sobel_filter(self, kernel_size, label):
        if kernel_size % 2 == 0:
            kernel_size += 1
        label.setText(f"Sobel Filter - Kernel Size: {kernel_size}")
        if self.selected_image_index is not None:
            image = self.get_current_image()
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
            sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
            self.update_image_display(sobel_combined)
        self.update()
        self.update_histogram()
        
    def apply_glitch_effect(self):
        """Apply glitch art effects to the image."""
        if self.selected_image_index is None or not self.images:
            QMessageBox.warning(self, "No Image", "Please load and select an image first.")
            return

        self.save_state()

        # Get the selected image
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert QPixmap to OpenCV format
        q_image = original_pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        stride = q_image.bytesPerLine()
        img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, stride))[:, :width * 3].reshape((height, width, 3))

        # --- Glitch Effect 1: Random Distortions ---
        for _ in range(10):  # Apply 10 random distortions
            x_start = np.random.randint(0, img_array.shape[1] - 20)
            x_end = x_start + np.random.randint(10, 50)
            y = np.random.randint(0, img_array.shape[0])
            img_array[y:y + 2, x_start:x_end] = np.roll(img_array[y:y + 2, x_start:x_end], shift=np.random.randint(-10, 10), axis=1)

        # --- Glitch Effect 2: Scanlines ---
        for y in range(0, img_array.shape[0], 3):
            img_array[y] = img_array[y] // 2

        # --- Glitch Effect 3: Chromatic Aberration ---
        b, g, r = cv2.split(img_array)
        b = np.roll(b, shift=5, axis=1)  # Shift blue channel
        g = np.roll(g, shift=-5, axis=0)  # Shift green channel
        img_array = cv2.merge((r, g, b))  # Convert back to RGB (fix color order)

        # Convert back to QPixmap
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        glitch_pixmap = QPixmap.fromImage(qt_image)

        # Resize to match the current cropped and scaled size
        resized_pixmap = self.resize_pixmap(glitch_pixmap, current_size)

        # Update the displayed pixmap
        self.images[self.selected_image_index]['pixmap'] = resized_pixmap

        self.update_histogram()
        self.update()
        
    def apply_dehaze(self):
        """Apply dehazing to the currently selected image."""
        if self.selected_image_index is None or not self.images:
            QMessageBox.warning(self, "No Image", "Please load and select an image first.")
            return

        self.save_state()

        # Get the original image and current size
        selected_image = self.images[self.selected_image_index]
        original_pixmap = selected_image['original_pixmap']
        current_size = selected_image['pixmap'].size()

        # Convert QPixmap to OpenCV format
        q_image = original_pixmap.toImage()
        q_image = q_image.convertToFormat(QImage.Format_RGB32)  # Ensure consistency with working examples
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        img_array = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # Extract RGB channels only

        # Apply dehazing
        try:
            # Assume `dehaze()` outputs a BGR image; adjust channel order
            dehazed_image = dehaze(img_array)  # `dehaze()` processes RGB or BGR; check its documentation

            # Convert to RGB for QImage/QPixmap compatibility
            if dehazed_image.shape[2] == 3:  # Ensure it's a 3-channel image
                dehazed_image = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB)

            # Convert back to QPixmap
            h, w, ch = dehazed_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(dehazed_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            dehazed_pixmap = QPixmap.fromImage(qt_image)

            # Resize to match the current size
            resized_pixmap = self.resize_pixmap(dehazed_pixmap, current_size)

            # Update the displayed pixmap
            selected_image['pixmap'] = resized_pixmap
            self.update()
            QMessageBox.information(self, "Dehazing Applied", "Dehazing has been successfully applied.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while applying dehazing: {e}")

        self.update()
        self.update_histogram()

    def qimage_to_cv2(self, qimage):
        """Convert QImage to OpenCV format."""
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width, height = qimage.width(), qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        stride = qimage.bytesPerLine()
        img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, stride))[:, :width * 3]
        return img_array.reshape((height, width, 3))

    def get_current_image(self):
        """Retrieve the original image from the images list."""
        original_pixmap = self.images[self.selected_image_index]['original_pixmap']
        image = original_pixmap.toImage()
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        return cv2.cvtColor(np.array(ptr).reshape(height, width, 4), cv2.COLOR_RGBA2BGR)

    def update_image_display(self, image):
        """Update the QLabel or canvas with the processed image."""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.images[self.selected_image_index]['pixmap'] = QPixmap.fromImage(q_image)
        self.update()
        self.update_histogram()
        
    def launch_new_window_sequence(self):
        # Show the splash screen first
        splash = SplashScreen("roboti.gif", width=800, height=500, duration=3000)
        splash.show()

        # Wait for the splash screen to finish and then open the canvas selector
        QTimer.singleShot(3000, self.open_new_canvas_selector)
        
    def open_new_canvas_selector(self):
        self.canvas_selector = CanvasApp()
        self.canvas_selector.show()


    def mousePressEvent(self, event):
        if self.drag_mode and event.button() == Qt.LeftButton:
            for i, img_data in enumerate(self.images):
                if img_data['rect'].contains(event.pos()):
                    self.selected_image_index = i
                    self.last_mouse_position = event.pos()
                    self.save_state()  # Save state before drag starts
                    break
            else:
                self.selected_image_index = None
        elif self.drawing_mode and event.button() == Qt.LeftButton:
            self.drawing_tool.start_drawing(event)
        elif self.eraser_mode and event.button() == Qt.LeftButton:
            self.erase_tool.start_erasing(event)
        elif self.shape_mode and event.button() == Qt.LeftButton:
            self.shape_tool.start_drawing(event)
        elif self.text_mode and event.button() == Qt.LeftButton:
            if self.text_tool.is_active:
                # Finalize the previous text before starting a new one
                self.text_tool.finalize_text()
            self.text_tool.start_typing(event)
            self.update()
        elif self.crop_mode and event.button() == Qt.LeftButton:
            # Start defining the crop area
            self.crop_start_point = event.pos()
            self.crop_end_point = None
            self.crop_rect = None
        elif self.scale_mode and event.button() == Qt.LeftButton:
            for i, img_data in enumerate(self.images):
                if img_data['rect'].contains(event.pos()):
                    self.selected_image_index = i
                    self.last_mouse_position = event.pos()
                    self.anchor_clicked = self.get_clicked_anchor(event.pos(), img_data['rect'])
                    if self.anchor_clicked:
                        self.save_state()  # Save state before scaling
                    break
            else:
                self.selected_image_index = None
        elif event.button() == Qt.LeftButton:
            for i, img_data in enumerate(self.images):
                if img_data['rect'].contains(event.pos()):
                    self.selected_image_index = i
                    self.last_mouse_position = event.pos()
                    self.anchor_clicked = self.get_clicked_anchor(event.pos(), img_data['rect'])
                    if self.anchor_clicked:
                        self.save_state()  # Save state before scaling
                    break
            else:
                self.selected_image_index = None
        elif event.button() == Qt.RightButton:
            for i, img_data in enumerate(self.images):
                if img_data['rect'].contains(event.pos()):
                    self.selected_image_index = i
                    break
            else:
                self.selected_image_index = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.selected_image_index is not None:
            img_data = self.images[self.selected_image_index]
            if self.anchor_clicked:
                # Scale the image
                self.scale_image(event.pos(), img_data)
            elif self.crop_mode and self.crop_start_point:
                self.crop_end_point = event.pos()
                self.crop_rect = QRect(self.crop_start_point, self.crop_end_point).normalized()
                self.update()  # Redraw to show the updated crop rectangle
            elif self.drawing_mode and event.buttons() == Qt.LeftButton:
                self.drawing_tool.continue_drawing(event)
            elif self.eraser_mode and event.buttons() == Qt.LeftButton:
                self.erase_tool.continue_erasing(event)
            elif self.shape_mode and event.buttons() == Qt.LeftButton:
                self.shape_tool.continue_drawing(event)

            elif self.drag_mode and event.buttons() == Qt.LeftButton:
                # Drag the image
                img_data = self.images[self.selected_image_index]
                delta = event.pos() - self.last_mouse_position
                self.last_mouse_position = event.pos()
                img_data['x'] += delta.x()
                img_data['y'] += delta.y()
                img_data['rect'].moveTo(img_data['x'], img_data['y'])
                self.update()
            elif self.drag_mode and event.buttons() == Qt.LeftButton:
                # Drag the image
                delta = event.pos() - self.last_mouse_position
                self.last_mouse_position = event.pos()
                img_data['x'] += delta.x()
                img_data['y'] += delta.y()
                img_data['rect'].moveTo(img_data['x'], img_data['y'])
            elif self.scale_mode and self.anchor_clicked:
                # Perform scaling
                self.scale_image(event.pos(), img_data)
            self.update()
            
        elif self.drawing_mode and event.buttons() == Qt.LeftButton:
                self.drawing_tool.continue_drawing(event)
                
        elif self.eraser_mode and event.buttons() == Qt.LeftButton:
                self.erase_tool.continue_erasing(event)
        elif self.shape_mode and event.buttons() == Qt.LeftButton:
            self.shape_tool.continue_drawing(event)

    def mouseReleaseEvent(self, event):
        if self.selected_image_index is not None and self.scale_mode:
            if self.anchor_clicked:  # Finalize scaling
                self.save_state()
        elif self.drawing_mode:
            self.drawing_tool.end_drawing()
            self.save_state()
        elif self.eraser_mode:
            self.erase_tool.end_erasing()
            self.save_state()
        elif self.shape_mode:
            self.shape_tool.end_drawing(event)
            self.save_state()
        elif self.text_mode and self.text_tool.is_active:
            # Finalize the text when the mouse is released
            self.text_tool.finalize_text()
            self.update()
        elif self.crop_mode and self.crop_start_point and event.button() == Qt.LeftButton:
            self.crop_end_point = event.pos()
            self.crop_rect = QRect(self.crop_start_point, self.crop_end_point).normalized()
            self.perform_crop()  # Perform the cropping action
            self.crop_start_point = None
            self.crop_end_point = None
            self.crop_rect = None
            self.update()  # Redraw after cropping
        self.anchor_clicked = None
        self.update()
        
    def keyPressEvent(self, event):
        if self.text_tool.is_active:
            self.text_tool.handle_key_press(event)
            self.update()
        else:
            super().keyPressEvent(event)
            
class MyApplication:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.canvas_app = None

        # Show the splash screen
        self.splash = SplashScreen("roboti.gif", duration=5000)
        self.splash.show()

        # Wait for 5 seconds before showing the main app
        QTimer.singleShot(5000, self.show_main_app)

    def show_main_app(self):
        self.canvas_app = CanvasApp()
        self.canvas_app.show()

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    app = MyApplication()
    app.run()

