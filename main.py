import argparse
from utils.extract_images import extract_images_from_pdf, extract_images_from_url
from utils.classify import classify_images_clip
from PIL import Image
import io

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str, help='Path to PDF file')
    parser.add_argument('--url', type=str, help='Website URL')
    args = parser.parse_args()

    if args.pdf:
        images = extract_images_from_pdf(args.pdf)
    elif args.url:
        print(f"Fetching from: {args.url}")  # Debug
        try:
            images = extract_images_from_url(args.url)
        except Exception as e:
            print(f"Error inside extract_images_from_url: {e}")
            return
    else:
        print("Please provide either a PDF or URL input.")
        return

    print(f"Found {len(images)} images.")

    if not images:
        print("No images were extracted. Exiting.")
        return

    for i, img_bytes in enumerate(images):
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            label = classify_images_clip(image)
            print(f"Image {i+1}: {label}")
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")

if __name__ == '__main__':
    main()
