!pip install autodistill autodistill_grounding_dino autodistill_yolov8 roboflow
!pip install opencv-python matplotlib tqdm scikit-learn requests pillow









import os
import time
import hashlib
import pathlib
import shutil
import subprocess
import cv2
import matplotlib.pyplot as plt
import math
import requests
import zipfile
from io import BytesIO
from urllib.parse import urlencode
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# --- Image Gathering Tunables ---
PER_PAGE = 10               # API max for image search
MAX_RESULTS_PER_TERM = 100  # Hard cap per term
TEST_SIZE = 0.30
RANDOM_STATE = 42
TIMEOUT = 20
MAX_RETRIES = 3
COOLDOWN_SEC = 1.0
DOWNLOAD_PAUSE = 0.02
DEFAULT_TERMS = [
    "Indian cow", "Indian cattle", "desi cow", "indigenous cow", "zebu cow",
    "Gir cow", "Sahiwal cow", "Red Sindhi cow", "Rathi cow", "Tharparkar cow"
]

# --- Annotation Tunables ---
CLASS_NAME = "cow"
PROMPT = "cow"
NUM_IMAGES_TO_DISPLAY = 16

def google_cse_image_search(api_key, cx, query, start, num):
    """Makes a request to the Google CSE API for images."""
    params = {
        "key": api_key, "cx": cx, "q": query, "searchType": "image", "num": num, "start": start
    }
    url = "https://www.googleapis.com/customsearch/v1?" + urlencode(params)
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def sanitize_ext_from_content_type(content_type, fallback=".jpg"):
    """Determines file extension from content type."""
    if not content_type: return fallback
    mapping = {
        "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
        "image/gif": ".gif", "image/bmp": ".bmp", "image/tiff": ".tiff",
    }
    return mapping.get(content_type.lower(), fallback)

def fetch_image(url):
    """Fetches image content from a URL with retries."""
    last_exc = None
    for _ in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT, stream=True)
            if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
                return resp.content, resp.headers.get("Content-Type")
        except Exception as e:
            last_exc = e
        time.sleep(0.5)
    if last_exc: raise last_exc
    raise RuntimeError("Failed to fetch image")

def save_image_bytes(data: bytes, out_path: str):
    """Saves image bytes to a file, handling potential PIL errors."""
    try:
        Image.open(BytesIO(data)).save(out_path)
    except Exception:
        with open(out_path, "wb") as f:
            f.write(data)

def gather_image_urls_multi_term(api_key, cx, terms, target_count):
    """Collects image URLs for multiple search terms."""
    urls = []
    per_term_target = max(1, target_count // max(1, len(terms)))
    pbar = tqdm(total=target_count, desc="Collecting image URLs across terms")
    for term in terms:
        need = target_count - len(urls)
        if need <= 0: break
        desired_for_term = min(per_term_target, need, MAX_RESULTS_PER_TERM)
        term_urls = []
        for page_idx in range(0, (desired_for_term // PER_PAGE) + 1):
            start = 1 + page_idx * PER_PAGE
            try:
                payload = google_cse_image_search(api_key, cx, term, start, PER_PAGE)
                items = payload.get("items", [])
                if not items: break
                for it in items:
                    link = it.get("link")
                    if link and link not in term_urls:
                        term_urls.append(link)
                        if len(term_urls) >= desired_for_term: break
            except requests.HTTPError: break
            time.sleep(COOLDOWN_SEC)
        urls.extend(term_urls)
        pbar.update(len(term_urls))
        if len(urls) >= target_count: break
    pbar.close()
    return urls

def download_unique_images(urls, out_dir):
    """Downloads images from URLs with deduplication."""
    os.makedirs(out_dir, exist_ok=True)
    seen_hashes = set()
    saved_files = []
    for url in tqdm(urls, desc="Downloading images"):
        try:
            data, ctype = fetch_image(url)
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes: continue
            seen_hashes.add(digest)
            ext = sanitize_ext_from_content_type(ctype)
            fname = f"{digest}{ext}"
            out_path = os.path.join(out_dir, fname)
            if not os.path.exists(out_path):
                save_image_bytes(data, out_path)
                saved_files.append(out_path)
        except Exception:
            pass
        time.sleep(DOWNLOAD_PAUSE)
    return saved_files

def create_zip_archive(source_dir, output_zip):
    """Compresses a directory into a zip file."""
    print(f"\nCreating zip file '{output_zip}' from '{source_dir}'...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.basename(file_path))
    print(f"Zip file '{output_zip}' created successfully.")
    return output_zip

def is_image_file(filename):
    """Checks if a file is an image based on its extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    return os.path.splitext(filename)[1].lower() in image_extensions

def load_image(image_path):
    """Loads an image from a given path and converts it to RGB."""
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def draw_bounding_boxes(image, boxes):
    """Draws bounding boxes on an image based on YOLO format labels."""
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        img_height, img_width = image.shape[:2]
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        left = int(x_center - (width / 2))
        top = int(y_center - (height / 2))
        right = int(x_center + (width / 2))
        bottom = int(y_center + (height / 2))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return image

def parse_yolo_labels(label_path):
    """Parses a YOLO-format label file and returns a list of bounding boxes."""
    boxes = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes

def plot_images_with_boxes(image_paths, label_paths, num_images):
    """Plots a grid of images with their corresponding bounding boxes."""
    grid_size = int(math.ceil(num_images / 4.0))
    fig, axes = plt.subplots(grid_size, 4, figsize=(20, grid_size * 5))
    axes = axes.flatten()
    for i in range(num_images):
        img_path = image_paths[i]
        label_path = label_paths[i]
        img = load_image(img_path)
        boxes = parse_yolo_labels(label_path)
        img = draw_bounding_boxes(img.copy(), boxes)
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Image {i + 1}")
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig('plot.png')
    print("Saved the plot as plot.png")
    plt.show()

def main_pipeline():
    """
    This is the main function that runs the entire pipeline:
    1. Downloads images using Google CSE.
    2. Zips the downloaded images into `cows.zip`.
    3. Annotates the images from the zip file using GroundingDINO.
    4. Displays a plot of the annotated images.
    """
    # --- Part 1: Image Gathering ---
    print("--- Part 1: Downloading Images ---")
    api_key = input("Enter API_KEY: ").strip()
    cx = input("Enter CX (Search engine ID): ").strip()
    target_count_input = input("Enter target_count (default 1000): ").strip()
    target_count = int(target_count_input) if target_count_input.isdigit() else 1000
    terms = DEFAULT_TERMS
    out_all_dir = "downloaded_images/all"

    urls = gather_image_urls_multi_term(api_key, cx, terms, target_count)
    print(f"Collected {len(urls)} URLs")

    files = download_unique_images(urls, out_all_dir)
    print(f"Downloaded {len(files)} unique images to {out_all_dir}")

    # --- Part 2: Create Zip File ---
    print("\n--- Part 2: Creating Zip Archive ---")
    zip_file = "cows.zip"
    create_zip_archive(out_all_dir, zip_file)

    # --- Part 3: Annotation and Visualization ---
    print("\n--- Part 3: Annotating Images ---")
    extract_dir = "cows_extracted"

    if not os.path.exists(zip_file):
        print(f"Error: The file '{zip_file}' was not found.")
        return

    print(f"Unpacking {zip_file}...")
    shutil.unpack_archive(zip_file, extract_dir)
    print("Dataset extracted successfully.")

    only_images_dir = './only_images_dir/'
    os.makedirs(only_images_dir, exist_ok=True)

    src_folder = extract_dir
    dest_folder = only_images_dir

    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if is_image_file(file):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_folder, file)
                shutil.copyfile(src_file_path, dest_file_path)

    print(f"Total number of images found: {len(os.listdir(dest_folder))}")

    base_model = GroundingDINO(ontology=CaptionOntology({CLASS_NAME: PROMPT}))
    print(f"Labeling images for '{CLASS_NAME}' with prompt '{PROMPT}'...")
    base_model.label(only_images_dir)
    print("Labeling complete.")

    image_folder = f'./{only_images_dir}_labeled/train/images'
    label_folder = f'./{only_images_dir}_labeled/train/labels'

    if os.path.exists(image_folder) and os.path.exists(label_folder):
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
        image_files.sort()
        label_files.sort()
        image_files = image_files[:NUM_IMAGES_TO_DISPLAY]
        label_files = label_files[:NUM_IMAGES_TO_DISPLAY]
        image_paths = [os.path.join(image_folder, f) for f in image_files]
        label_paths = [os.path.join(label_folder, f) for f in label_files]

        if image_paths and label_paths:
            plot_images_with_boxes(image_paths, label_paths, len(image_paths))
        else:
            print("No labeled images or labels found to display.")
    else:
        print("Labeled directories not found. Check if the labeling process was successful.")

if __name__ == '__main__':
    main_pipeline()