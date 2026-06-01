import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QLabel, QLineEdit, QSpinBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

"""
ecog_gui_v3

Update Log:
01/05/2025 (XW):    Added ESC key to clear unsaved selection
                    - Method 'on_key_press': "ESC" logic added
                    - Method 'clear_selection': defined
                    - Method 'keyPressEvent': "ESC" logic added
"""


class EcogGUI(QMainWindow):
    def __init__(self, ecog_data):
        super().__init__()
        self.ecog = ecog_data
        self.sr = ecog_data['srate']
        self.nr_sample = len(ecog_data['data'][0])
        self.selected_channels = np.array(self.ecog['selectedChannels'])

        # Initialize parameters
        self.n_displayed = 40           # Default number of channels to display
        self.start_time = 0             # Start time in seconds
        self.interval = 5               # Default window width in seconds
        self.vertical_scale = 1         # Default vertical scale factor
        self.current_channel_start = 0  # Starting index of displayed channels

        # Interaction flags
        self.shift_pressed = False
        self.x1 = None
        self.x2 = None
        self.rect_patch = None
        self.marked_intervals = []  # Store x1, x2

        # Add 'marked_intervals' field to ecog dict if not present
        if 'marked_intervals' not in self.ecog:
            self.ecog['marked_intervals'] = []

        self.initUI()

    def initUI(self):
        self.setWindowTitle("ECoG Time Series GUI")
        self.setGeometry(0, 0, 1200, 800)

        widget = QWidget()
        layout = QGridLayout()

        # Matplotlib figure (signal display area)
        self.figure, self.ax = plt.subplots(figsize=(16, 9))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 0, 0, 2, 10)

        # =================== Control Panel ===================

        # Channel selection
        layout.addWidget(QLabel("Ch"), 2, 0, alignment=Qt.AlignRight)
        self.channel_input = QLineEdit(" ".join(map(str, self.selected_channels.tolist())))
        self.channel_input.textChanged.connect(self.update_plot)  # Listen for input changes
        layout.addWidget(self.channel_input, 2, 1)

        # Number of channels to display
        layout.addWidget(QLabel("#"), 2, 2, alignment=Qt.AlignRight)
        self.nChannelsDisplayed = QSpinBox()
        self.nChannelsDisplayed.setValue(self.n_displayed)
        self.nChannelsDisplayed.setMinimum(1)
        self.nChannelsDisplayed.setMaximum(len(self.selected_channels))
        self.nChannelsDisplayed.valueChanged.connect(self.update_plot)  # Listen for input changes
        layout.addWidget(self.nChannelsDisplayed, 2, 3)

        # Up/Down buttons
        self.up_btn = QPushButton("↑")
        self.down_btn = QPushButton("↓")
        self.up_btn.clicked.connect(self.scroll_up_channels)
        self.down_btn.clicked.connect(self.scroll_down_channels)
        layout.addWidget(self.up_btn, 2, 4)
        layout.addWidget(self.down_btn, 3, 4)

        # Page left/right buttons
        self.page_left_btn = QPushButton("<<")
        self.page_right_btn = QPushButton(">>")
        self.page_left_btn.clicked.connect(lambda: self.page_time(-1))
        self.page_right_btn.clicked.connect(lambda: self.page_time(1))
        layout.addWidget(self.page_left_btn, 2, 5)
        layout.addWidget(self.page_right_btn, 2, 6)

        # Step left/right buttons
        self.step_left_btn = QPushButton("<")
        self.step_right_btn = QPushButton(">")
        self.step_left_btn.clicked.connect(lambda: self.page_time(-0.33))
        self.step_right_btn.clicked.connect(lambda: self.page_time(0.33))
        layout.addWidget(self.step_left_btn, 3, 5)
        layout.addWidget(self.step_right_btn, 3, 6)

        # Start time input
        layout.addWidget(QLabel("Start:"), 3, 0, alignment=Qt.AlignRight)
        self.start_input = QLineEdit(str(self.start_time))
        self.start_input.textChanged.connect(self.update_plot)  # Listen for input changes
        layout.addWidget(self.start_input, 3, 1)

        # Interval input
        layout.addWidget(QLabel("Interval:"), 3, 2, alignment=Qt.AlignRight)
        self.interval_input = QLineEdit(str(self.interval))
        self.interval_input.textChanged.connect(self.update_plot)  # Listen for input changes
        layout.addWidget(self.interval_input, 3, 3)

        # Vertical Scale input
        layout.addWidget(QLabel("Vertical Scale:"), 2, 8, alignment=Qt.AlignRight)
        self.vertical_scale_input = QLineEdit(str(self.vertical_scale))
        self.vertical_scale_input.textChanged.connect(self.update_plot)  # Listen for input changes
        layout.addWidget(self.vertical_scale_input, 3, 8)

        # Vertical Scale adjustment buttons
        self.scale_up_btn = QPushButton("*2")
        self.scale_down_btn = QPushButton("/2")
        self.scale_up_btn.clicked.connect(self.scale_up)
        self.scale_down_btn.clicked.connect(self.scale_down)
        layout.addWidget(self.scale_up_btn, 2, 9)
        layout.addWidget(self.scale_down_btn, 3, 9)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Bind events
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        # Force canvas to take focus
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        # Initialize plot
        self.update_plot()

    def closeEvent(self, event):
        """Override close event to save marked_intervals to file"""
        # Save marked_intervals to ecogStruct1_badEpochs.pkl
        with open('data/raw/ecogStruct1_badEpochs.pkl', 'wb') as f:
            pickle.dump(self.ecog, f)
        print("✅ Marked intervals saved to data/raw/ecogStruct1_badEpochs.pkl")
        super().closeEvent(event)

    def on_mouse_click(self, event):
        """Mouse click event (SHIFT + left click)"""
        if event.xdata is None:  # Check if event.xdata is valid
            return

        if event.button == 1 and self.shift_pressed:  # Left click + SHIFT
            # Clear old blue region
            if self.rect_patch:
                self.rect_patch.remove()
                self.rect_patch = None

            if self.x1 is None:
                self.x1 = event.xdata  # Record first point
            else:
                self.x2 = event.xdata  # Record second point
                if self.x1 > self.x2:
                    self.x1, self.x2 = self.x2, self.x1  # Ensure x1 < x2
                print(f'X1: {self.x1}, X2: {self.x2}')

                # Draw cyan rectangle
                self.rect_patch = self.ax.axvspan(self.x1, self.x2, color='cyan', alpha=0.3)
                self.canvas.draw_idle()  # Refresh plot

    def on_key_press(self, event):
        """Handle key press events"""
        if event.key == "shift":
            self.shift_pressed = True  # Record SHIFT pressed
        elif event.key == "b":
            if self.x1 is not None and self.x2 is not None:
                # Save x1, x2 to ecog dict
                self.ecog['marked_intervals'].append((self.x1, self.x2))
                print(f"✅ Saved interval: {self.x1:.2f} - {self.x2:.2f}")

                # Clear current rectangle
                if self.rect_patch:
                    self.rect_patch.remove()
                    self.rect_patch = None

                # Reset x1 and x2
                self.x1, self.x2 = None, None

                # Refresh plot
                self.canvas.draw_idle()
            else:
                print(f'X1 is None')
        elif event.key == "escape":  # Escape button function: clear selection of unsaved points
            self.clear_selection()

    def clear_selection(self):
        """Clear the current selection (x1, x2 and rectangle)"""
        if self.rect_patch:
            self.rect_patch.remove()
            self.rect_patch = None
        self.x1, self.x2 = None, None
        self.canvas.draw_idle()
        print("Cleared current selection")

    def on_key_release(self, event):
        """Detect SHIFT release"""
        if event.key == "shift":
            self.shift_pressed = False  # Release SHIFT
            # Only reset x1 if no interval is selected
            if self.x2 is None:
                self.x1 = None  # Cancel x1, restart selection

    def on_scroll(self, event):
        """Mouse scroll event"""
        if event.button == 'up':
            self.scroll_up_channels()
        elif event.button == 'down':
            self.scroll_down_channels()

    def keyPressEvent(self, event):
        """Global key press event for PyQt5"""
        if event.key() == Qt.Key_Shift:
            self.shift_pressed = True   # SHIFT pressed
        elif event.key() == Qt.Key_Escape:
            self.clear_selection()      # ESCAPE pressed
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Global key release event for PyQt5"""
        if event.key() == Qt.Key_Shift:
            self.shift_pressed = False  # SHIFT released
            self.x1 = None  # Cancel x1
        super().keyReleaseEvent(event)

    def update_plot(self):
        """Update ECoG signal display"""
        try:
            # Parse inputs
            ch_str = self.channel_input.text().strip().split()
            channels = [int(ch) - 1 for ch in ch_str]  # Convert 1-based to 0-based
            self.n_displayed = int(self.nChannelsDisplayed.text())  # Ensure `#` input takes effect
            self.start_time = float(self.start_input.text())
            self.interval = float(self.interval_input.text())
            self.vertical_scale = float(self.vertical_scale_input.text())
        except ValueError:
            print("Input format error, please check!")
            return

        self.ax.clear()

        # Calculate time index range
        start_idx = int(self.start_time * self.sr)
        end_idx = int((self.start_time + self.interval) * self.sr)

        max_channels = len(self.ecog['data'])  # Maximum number of channels
        channels = [ch for ch in channels if 0 <= ch < max_channels]  # Filter invalid channels

        # Ensure displayed channels do not exceed n_displayed
        if len(channels) > self.n_displayed:
            channels = channels[self.current_channel_start:self.current_channel_start + self.n_displayed]

        # Plot signals
        y_ticks = []
        for i, ch in enumerate(channels):
            if 0 <= ch < max_channels:
                data = self.ecog['data'][ch, start_idx:end_idx] * self.vertical_scale

                # Color: odd channels blue, even channels orange
                color = 'blue' if (i + 1) % 2 == 1 else 'red'

                # Plot signal and adjust baseline to avoid overlap
                time_axis = np.arange(start_idx, end_idx) / self.sr  # Convert sample_index to seconds
                self.ax.plot(time_axis, data + i * 100, label=f"Ch {ch + 1}", color=color)
                data_avg = np.mean(data + i * 100)
                self.ax.axhline(data_avg, linestyle='--', color='grey', alpha=0.6)  # Grey baseline
                y_ticks.append(data_avg)

        # Set Y-axis to channel numbers
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels([f"Ch {ch + 1}" for ch in channels])
        self.ax.set_ylabel("Channel")

        # Set X-axis
        self.ax.set_title("ECoG Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_xlim(left=start_idx / self.sr, right=end_idx / self.sr)  # Convert sample_index to seconds
        self.ax.margins(x=0, y=0)
        self.figure.subplots_adjust(left=0.05, right=0.95)

        # Refresh plot
        self.canvas.draw_idle()

    def scroll_up_channels(self):
        """Scroll channels up (move down)"""
        ch_str = self.channel_input.text().strip().split()
        channels = [int(ch) - 1 for ch in ch_str]  # Convert 1-based to 0-based
        if len(channels) > self.n_displayed:
            self.current_channel_start = min(self.current_channel_start + 1, len(channels) - self.n_displayed)
            self.update_plot()

    def scroll_down_channels(self):
        """Scroll channels down (move up)"""
        self.current_channel_start = max(self.current_channel_start - 1, 0)
        self.update_plot()

    def page_time(self, direction):
        """Page left/right (direction: -1 for left, 1 for right)"""
        max_time = len(self.ecog['data'][0]) / self.sr  # Maximum time
        step = self.interval * direction
        new_start = self.start_time + step

        # Ensure no out-of-bounds
        if new_start < 0:
            new_start = 0
        elif new_start + self.interval > max_time:
            new_start = max_time - self.interval

        if new_start != self.start_time:
            self.start_time = new_start
            self.start_input.setText(str(self.start_time))
            self.update_plot()

    def scale_up(self):
        """Zoom in signal"""
        self.vertical_scale *= 2
        self.vertical_scale_input.setText(str(self.vertical_scale))
        self.update_plot()

    def scale_down(self):
        """Zoom out signal"""
        self.vertical_scale /= 2
        self.vertical_scale_input.setText(str(self.vertical_scale))
        self.update_plot()

if __name__ == "__main__":
    # Load ecog data from ecogStruct1_periodogram.pkl
    with open('data/raw/ecogStruct1_periodogram.pkl', 'rb') as f:
        ecog_data = pickle.load(f)

    # # Add 'marked_intervals' field if not present
    # if 'marked_intervals' not in ecog_data:
    #     ecog_data['marked_intervals'] = []

    # Every time run this script, will clean up the marked_intervals
    ecog_data['marked_intervals'] = []

    app = QApplication(sys.argv)
    gui = EcogGUI(ecog_data)
    gui.show()
    sys.exit(app.exec_())