import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
import io
import re

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append(base_image["image"])
    return images

def extract_images_from_url(url):
    url = url.strip()
    images = []

    # Direct image link detection (jpg, png, tiff, webp, etc.)
    if url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
        print(f"Direct image URL detected: {url}")
        img_bytes = download_and_convert(url)
        if img_bytes:
            return [img_bytes]
        else:
            return []

    # Check if URL contains common image patterns
    if any(pattern in url.lower() for pattern in ['/image/', '/photo/', '/img/', '.png', '.jpg', '.jpeg']):
        print(f"Image-like URL detected: {url}")
        img_bytes = download_and_convert(url)
        if img_bytes:
            return [img_bytes]
        else:
            print("Failed to download as direct image, trying as webpage...")

    # Otherwise, treat it as a webpage
    print(f"Fetching HTML from: {url}")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract <img> tags
        img_tags = soup.find_all("img")
        for tag in img_tags:
            src = tag.get("src")
            if not src:
                continue
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                base = "/".join(url.split("/")[:3])
                src = base + src

            try:
                img_bytes = download_and_convert(src)
                if img_bytes:
                    images.append(img_bytes)
            except Exception as e:
                print(f"Error processing image from {src}: {e}")
    except Exception as e:
        print(f"Error fetching webpage: {e}")
        return []

    print(f"Found {len(images)} images.")
    return images


def download_and_convert(img_url):
    """
    Downloads an image from a URL, opens it with Pillow regardless of format,
    converts to RGB JPEG, and returns bytes.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(img_url, timeout=10, headers=headers)
    resp.raise_for_status()

    try:
        with Image.open(io.BytesIO(resp.content)) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()
    except Exception as e:
        print(f"Could not open/convert {img_url}: {e}")
        return None