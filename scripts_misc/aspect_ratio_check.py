import os
from pathlib import Path
from PIL import Image
from collections import defaultdict

def get_image_size(image_path):
    """Get the size (width, height) of an image."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return (width, height)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None


def check_image_sizes(data_dir):
    """Check if all images in RESP folders have the same size."""
    data_path = Path(data_dir)
    
    # Find all RESP**** folders
    resp_folders = [f for f in data_path.iterdir() 
                    if f.is_dir() and f.name.startswith("RESP")]
    
    if not resp_folders:
        print(f"No RESP**** folders found in {data_dir}")
        return
    
    print(f"Found {len(resp_folders)} RESP folders")
    
    for resp_folder in sorted(resp_folders):
        print(f"\nChecking {resp_folder.name}...")
        
        image_sizes = []
        image_files = []
        
        # Check images in main folder
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_file in resp_folder.glob(ext):
                size = get_image_size(img_file)
                if size is not None:
                    image_sizes.append(size)
                    image_files.append(img_file)
        
        # Check images in coregistered subfolder
        coreg_folder = resp_folder / "Coregistered"
        if coreg_folder.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_file in coreg_folder.glob(ext):
                    size = get_image_size(img_file)
                    if size is not None:
                        image_sizes.append(size)
                        image_files.append(img_file)

        if not image_sizes:
            print(f"  No images found")
            continue

        # Check if all sizes are within 1% tolerance
        if not image_sizes:
            continue
            
        # Use first image as reference
        ref_width, ref_height = image_sizes[0]
        tolerance = 0.01  # 1% tolerance
        
        all_within_tolerance = True
        outliers = []
        
        for img_file, (width, height) in zip(image_files, image_sizes):
            width_diff = abs(width - ref_width) / ref_width
            height_diff = abs(height - ref_height) / ref_height
            
            if width_diff > tolerance or height_diff > tolerance:
                all_within_tolerance = False
                outliers.append((img_file, (width, height)))
        
        if all_within_tolerance:
            print(f"  ✓ All {len(image_sizes)} images have similar sizes (within 1% of {ref_width}x{ref_height})")
        else:
            print(f"  ✗ Some images differ by more than 1% from reference size {ref_width}x{ref_height}:")
            for img_file, (width, height) in outliers:
                width_diff_pct = abs(width - ref_width) / ref_width * 100
                height_diff_pct = abs(height - ref_height) / ref_height * 100
                print(f"      - {img_file.name}: {width}x{height} (Δ {width_diff_pct:.1f}% × {height_diff_pct:.1f}%)")


if __name__ == "__main__":
    # Set your data directory here
    data_directory = r"L:\her_knf_golf\Wetenschap\newtransport\Wouter\Projects\04 - Grid Localization\PhotoMatchSjors\Photos"
    
    if not os.path.exists(data_directory):
        print(f"Directory {data_directory} does not exist!")
    else:
        check_image_sizes(data_directory)