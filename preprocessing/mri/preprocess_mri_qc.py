"""
preprocess_mri_qc.py

Quality control for MRI preprocessing results.

This script:
- Uses tkinter to display QC images one by one
- Saves failed and successful subjects to text files
- Usage:
-   S / <Enter>: Mark as successful
-   F: Mark as failed
-       1: Mark as failed w/ Missing GT
-       2: Mark as failed w/ Bad registration
-       3: Mark as failed w/ Bad data quality
-   Mouse wheel / +/-: Zoom in/out
-   Click and drag: Pan image
-   R: Reset zoom/pan
- ESC: Quit early and save progress

Author: Sjors Verschuren
Date: November 2025
"""

import os
import shutil
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from glob import glob

from util.config import get_data_root

_data_root = get_data_root()
MRI_PREPROCESS_DIR: Path = (
    _data_root / "preprocessing" / "mri"
    if _data_root
    else Path("preprocessing/mri")
)
OVERWRITE_QC_IMGS = True

class MRIQualityControl:
    def __init__(self, root, image_dir, output_dir):
        self.root = root
        self.root.title("MRI Quality Control")
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)

        # Make window fullscreen
        self.root.attributes("-fullscreen", True)
        
        # Find all QC images
        self.images = sorted([Path(p) for p in glob(str(self.image_dir / "*.png"))]) + \
                 sorted([Path(p) for p in glob(str(self.image_dir / "*.jpg"))])
        
        if not self.images:
            messagebox.showerror("Error", f"No images found in {image_dir}")
            root.destroy()
            return

        # Load existing success/fail lists if they exist
        success_file = self.output_dir / "success_subjects.txt"
        failed_file = self.output_dir / "failed_subjects.txt"
        
        existing_subjects = set()
        if success_file.exists():
            with open(success_file, 'r') as f:
                existing_subjects.update(line.strip() for line in f if line.strip())
        if failed_file.exists():
            with open(failed_file, 'r') as f:
                existing_subjects.update(line.strip().split(",")[0] for line in f if line.strip())
        
        # Filter out already reviewed images
        if existing_subjects:
            self.images = [img for img in self.images if img.stem.replace('_qc', '') not in existing_subjects]
            print(f"Skipping {len(existing_subjects)} already reviewed subjects")
            print(f"Remaining images to review: {len(self.images)}")
        
        if not self.images:
            messagebox.showinfo("Complete", "All images have been reviewed!")
            root.destroy()
            return

        self.current_index = 0
        self.failed_subjects = []
        self.success_subjects = []
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.original_image = None
        
        # Setup UI
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        self.label = tk.Label(self.canvas)
        self.image_item = self.canvas.create_window(0, 0, anchor=tk.NW, window=self.label)
        
        self.info_label = tk.Label(root, text="", font=("Arial", 12))
        self.info_label.pack(pady=10)
        
        self.instruction_label = tk.Label(
            root, 
            text="Press ENTER or S for Success | Press F for Fail (Or 1: Missing GT, 2: Bad registration, 3: Bad data quality) | ESC to quit",
            font=("Arial", 10)
        )
        self.instruction_label.pack(pady=5)
        
        # Bind keys
        self.root.bind("<Return>", lambda e: self.mark_success())
        self.root.bind("s", lambda e: self.mark_success())
        self.root.bind("S", lambda e: self.mark_success())
        self.root.bind("f", lambda e: self.mark_fail())
        self.root.bind("F", lambda e: self.mark_fail())
        self.root.bind("1", lambda e: self.mark_fail(reason="Missing GT"))
        self.root.bind("2", lambda e: self.mark_fail(reason="Bad registration"))
        self.root.bind("3", lambda e: self.mark_fail(reason="Bad data quality"))
        self.root.bind("<Escape>", lambda e: self.quit_early())
        self.root.bind("<F11>", lambda e: self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen")))
        self.root.bind("r", lambda e: self.reset_zoom_pan())
        self.root.bind("R", lambda e: self.reset_zoom_pan())
        self.root.bind("+", lambda e: self.zoom(e, direction=1))
        self.root.bind("-", lambda e: self.zoom(e, direction=-1))
        
        # Bind mouse events for zoom and pan
        self.label.bind("<MouseWheel>", self.on_mousewheel)
        self.label.bind("<ButtonPress-1>", self.on_drag_start)
        self.label.bind("<B1-Motion>", self.on_drag_motion)

        self.show_image()
    
    def show_image(self):
        if self.current_index >= len(self.images):
            self.finish()
            return
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Load image
        image_path = self.images[self.current_index]
        self.original_image = Image.open(image_path).convert("RGB")

        # Compute initial zoom to fit screen
        img_w, img_h = self.original_image.size
        zoom_w = screen_width / img_w
        zoom_h = screen_height / img_h
        self.zoom_level = min(zoom_w, zoom_h)   # Fit-to-screen zoom

        # Pan so image is centered
        self.pan_x = (screen_width  - img_w * self.zoom_level) / 2
        self.pan_y = (screen_height - img_h * self.zoom_level) / 2

        # Display original image
        photo = ImageTk.PhotoImage(self.original_image)
        self.label.config(image=photo)
        self.label.image = photo
        
        # Update info
        subject_name = image_path.stem.replace('_qc', '')
        progress = f"Image {self.current_index + 1}/{len(self.images)}: {subject_name}"
        self.info_label.config(text=progress)

        # Update display
        self.update_image_display()
        
    def on_mousewheel(self, event):
        # Zoom in/out with mouse wheel
        direction = +1 if event.delta > 0 else -1
        self.zoom(event, direction)
        
    def zoom(self, event, direction):
        old_zoom = self.zoom_level
        factor = 1.2 if direction > 0 else 1/1.2
        self.zoom_level = max(0.1, min(10, self.zoom_level * factor))

        # Zoom relative to cursor position:
        cursor_x = event.x
        cursor_y = event.y

        self.pan_x = cursor_x - (cursor_x - self.pan_x) * (self.zoom_level / old_zoom)
        self.pan_y = cursor_y - (cursor_y - self.pan_y) * (self.zoom_level / old_zoom)

        self.update_image_display()
        
    def on_drag_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
    def on_drag_motion(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.update_image_display()
        
    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_image_display()
        
    def update_image_display(self):
        if self.original_image is None:
            return
        
        # Calculate visible region to crop before resizing (viewport optimization)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate which part of the image is visible
            x1 = max(0, int(-self.pan_x / self.zoom_level))
            y1 = max(0, int(-self.pan_y / self.zoom_level))
            x2 = min(self.original_image.width, int((canvas_width - self.pan_x) / self.zoom_level))
            y2 = min(self.original_image.height, int((canvas_height - self.pan_y) / self.zoom_level))
            
            # Crop to visible region first, then resize (much faster)
            img_cropped = self.original_image.crop((x1, y1, x2, y2))
            new_w = int((x2 - x1) * self.zoom_level)
            new_h = int((y2 - y1) * self.zoom_level)
        else:
            # Fallback if canvas size not available yet
            img_cropped = self.original_image
            new_w = int(self.original_image.width * self.zoom_level)
            new_h = int(self.original_image.height * self.zoom_level)
            x1, y1 = 0, 0
        
        # Use NEAREST for faster rendering during interaction, LANCZOS for final
        resample = Image.Resampling.NEAREST if abs(self.zoom_level - 1.0) > 0.01 else Image.Resampling.LANCZOS
        img = img_cropped.resize((new_w, new_h), resample)
        photo = ImageTk.PhotoImage(img)
        
        self.label.config(image=photo)
        self.label.image = photo
        
        # Adjust pan to account for crop offset
        self.canvas.coords(self.image_item, self.pan_x + x1 * self.zoom_level, self.pan_y + y1 * self.zoom_level)

    def mark_success(self):
        subject = Path(self.images[self.current_index]).stem.replace('_qc', '')
        self.success_subjects.append(subject)
        print(f"Success: {subject}")
        self.current_index += 1
        self.show_image()
    
    def mark_fail(self, reason=None):
        subject = Path(self.images[self.current_index]).stem.replace('_qc', '')
        self.failed_subjects.append((subject, reason))
        print(f"Failed: {subject}")
        self.current_index += 1
        self.show_image()
    
    def quit_early(self):
        if messagebox.askyesno("Quit", "Save progress and quit?"):
            self.finish()
    
    def finish(self):
        # Write failed subjects to file
        failed_file = self.output_dir / "failed_subjects.txt"
        with open(failed_file, 'a') as f:
            for subject, reason in self.failed_subjects:
                f.write(f"{subject}, {reason}\n")
        
        # Write successful subjects to file
        success_file = self.output_dir / "success_subjects.txt"
        with open(success_file, 'a') as f:
            for subject in self.success_subjects:
                f.write(f"{subject}\n")
        
        print(f"\nQuality control complete!")
        print(f"Total images reviewed: {self.current_index}")
        print(f"Successful subjects: {len(self.success_subjects)}")
        print(f"Failed subjects: {len(self.failed_subjects)}")
        print(f"Success subjects saved to: {success_file}")
        print(f"Failed subjects saved to: {failed_file}")
        
        self.root.destroy()

def main():
    # Setup image directory
    image_dir = MRI_PREPROCESS_DIR / "QC_images"
    image_dir.mkdir(exist_ok=True)

    # Copy QC images to image_dir
    for pattern in ["RESP*", "Bonn*"]:
        for subject_folder in glob(str(MRI_PREPROCESS_DIR / pattern)):
            subject_folder = Path(subject_folder)
            if subject_folder.is_dir():
                subject_id = os.path.basename(subject_folder)
                qc_image = subject_folder / f"{subject_id}_qc.png"
                dest_image = image_dir / f"{subject_id}_qc.png"
                if qc_image.exists() and (OVERWRITE_QC_IMGS or not dest_image.exists()):
                    shutil.copy2(qc_image, dest_image)

    # Setup output directory
    output_dir = image_dir
    
    root = tk.Tk()
    app = MRIQualityControl(root, image_dir, output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
