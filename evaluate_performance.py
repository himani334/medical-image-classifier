#!/usr/bin/env python3
"""
Medical Image Classifier Performance Evaluation Script

This script evaluates the performance of the medical image classifier across multiple metrics:
- Classification accuracy
- Processing speed and throughput
- Memory usage and model efficiency
- Scalability testing
"""

import time
import psutil
import os
import sys
from pathlib import Path
import statistics
from utils.classify import classify_images_clip
from utils.extract_images import extract_images_from_pdf
from PIL import Image
import io
import torch

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def evaluate_single_image_classification():
    """Evaluate classification accuracy on single images"""
    print("🔍 Evaluating Single Image Classification...")
    
    # Test with known medical and non-medical images
    test_cases = [
        ("test_images/test image 1.jpg", "medical"),
    ]
    
    correct = 0
    total = 0
    times = []
    
    for image_path, expected_label in test_cases:
        if not os.path.exists(image_path):
            print(f"⚠️  Skipping {image_path} - file not found")
            continue
            
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Measure classification time
            start_time = time.time()
            predicted_label = classify_images_clip(image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            
            # Check accuracy
            is_correct = predicted_label == expected_label
            correct += int(is_correct)
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {image_path}: Expected '{expected_label}', Got '{predicted_label}' ({processing_time:.3f}s)")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = statistics.mean(times) if times else 0
    
    print(f"\n📊 Single Image Results:")
    print(f"   Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"   Average Time: {avg_time:.3f} seconds")
    print(f"   Throughput: {1/avg_time:.2f} images/second")
    
    return accuracy, avg_time

def evaluate_pdf_classification():
    """Evaluate classification accuracy on PDF documents"""
    print("\n📄 Evaluating PDF Classification...")
    
    pdf_tests = [
        ("test_pdfs/medical_images.pdf", "medical"),
        ("test_pdfs/non_medical_images.pdf", "non-medical"),
        ("test_pdfs/combined_test.pdf", "mixed"),
    ]
    
    total_images = 0
    total_time = 0
    
    for pdf_path, expected_type in pdf_tests:
        if not os.path.exists(pdf_path):
            print(f"⚠️  Skipping {pdf_path} - file not found")
            continue
            
        try:
            print(f"\n📖 Processing {pdf_path} (Expected: {expected_type})")
            
            # Measure extraction and classification time
            start_time = time.time()
            images = extract_images_from_pdf(pdf_path)
            extraction_time = time.time() - start_time
            
            print(f"   Extracted {len(images)} images in {extraction_time:.3f}s")
            
            # Classify each image
            classification_times = []
            for i, img_bytes in enumerate(images):
                try:
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    
                    start_time = time.time()
                    label = classify_images_clip(image)
                    end_time = time.time()
                    
                    classification_time = end_time - start_time
                    classification_times.append(classification_time)
                    
                    print(f"   Image {i+1}: {label} ({classification_time:.3f}s)")
                    
                except Exception as e:
                    print(f"   ❌ Error processing image {i+1}: {e}")
            
            if classification_times:
                avg_class_time = statistics.mean(classification_times)
                total_time += sum(classification_times)
                total_images += len(images)
                
                print(f"   Average classification time: {avg_class_time:.3f}s")
                
        except Exception as e:
            print(f"❌ Error processing PDF {pdf_path}: {e}")
    
    if total_images > 0:
        avg_total_time = total_time / total_images
        throughput = total_images / total_time if total_time > 0 else 0
        
        print(f"\n📊 PDF Processing Results:")
        print(f"   Total Images: {total_images}")
        print(f"   Average Time per Image: {avg_total_time:.3f} seconds")
        print(f"   Overall Throughput: {throughput:.2f} images/second")
        
        return total_images, avg_total_time
    else:
        return 0, 0

def evaluate_memory_efficiency():
    """Evaluate memory usage and model efficiency"""
    print("\n💾 Evaluating Memory Efficiency...")
    
    # Get initial memory
    initial_memory = get_memory_usage()
    print(f"   Initial Memory: {initial_memory:.1f} MB")
    
    # Load model (this happens on first classification)
    print("   Loading CLIP model...")
    start_time = time.time()
    
    # Create a dummy image to trigger model loading
    dummy_image = Image.new('RGB', (100, 100), color='white')
    _ = classify_images_clip(dummy_image)
    
    model_load_time = time.time() - start_time
    memory_after_load = get_memory_usage()
    
    print(f"   Model Load Time: {model_load_time:.3f} seconds")
    print(f"   Memory After Load: {memory_after_load:.1f} MB")
    print(f"   Memory Increase: {memory_after_load - initial_memory:.1f} MB")
    
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"   GPU Available: Yes ({gpu_memory:.0f} MB total)")
    else:
        print(f"   GPU Available: No (using CPU)")
    
    return memory_after_load - initial_memory, model_load_time

def evaluate_scalability():
    """Evaluate how the system scales with multiple images"""
    print("\n📈 Evaluating Scalability...")
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n   Testing batch size: {batch_size}")
        
        # Create test images
        test_images = [Image.new('RGB', (224, 224), color='white') for _ in range(batch_size)]
        
        # Measure processing time
        start_time = time.time()
        for image in test_images:
            _ = classify_images_clip(image)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_image = total_time / batch_size
        throughput = batch_size / total_time
        
        results.append({
            'batch_size': batch_size,
            'total_time': total_time,
            'avg_time_per_image': avg_time_per_image,
            'throughput': throughput
        })
        
        print(f"     Total Time: {total_time:.3f}s")
        print(f"     Avg Time per Image: {avg_time_per_image:.3f}s")
        print(f"     Throughput: {throughput:.2f} images/second")
    
    # Analyze scalability
    print(f"\n📊 Scalability Analysis:")
    for i, result in enumerate(results):
        if i > 0:
            efficiency = results[0]['throughput'] / result['throughput'] * result['batch_size']
            print(f"   Batch {result['batch_size']}: {efficiency:.2f}x efficiency vs single image")
    
    return results

def generate_performance_report():
    """Generate comprehensive performance report"""
    print("🚀 Medical Image Classifier Performance Evaluation")
    print("=" * 60)
    
    # Run all evaluations
    single_acc, single_time = evaluate_single_image_classification()
    pdf_images, pdf_time = evaluate_pdf_classification()
    memory_usage, model_load_time = evaluate_memory_efficiency()
    scalability_results = evaluate_scalability()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"🎯 Classification Accuracy:")
    print(f"   Single Images: {single_acc:.1f}%")
    print(f"   PDF Documents: Tested with {pdf_images} images")
    
    print(f"\n⚡ Processing Speed:")
    print(f"   Single Image: {single_time:.3f}s ({1/single_time:.2f} img/sec)")
    print(f"   PDF Processing: {pdf_time:.3f}s per image")
    print(f"   Model Loading: {model_load_time:.3f}s (first run only)")
    
    print(f"\n💾 Resource Usage:")
    print(f"   Memory Usage: {memory_usage:.1f} MB")
    print(f"   GPU Support: {'Yes' if torch.cuda.is_available() else 'No'}")
    
    print(f"\n📈 Scalability:")
    for result in scalability_results:
        print(f"   Batch {result['batch_size']}: {result['throughput']:.2f} img/sec")
    
    print(f"\n🏆 Overall Assessment:")
    if single_acc >= 95:
        print("   ✅ Excellent classification accuracy")
    elif single_acc >= 90:
        print("   ✅ Good classification accuracy")
    else:
        print("   ⚠️  Classification accuracy needs improvement")
    
    if single_time < 1.0:
        print("   ✅ Fast processing speed")
    elif single_time < 3.0:
        print("   ✅ Acceptable processing speed")
    else:
        print("   ⚠️  Processing speed could be improved")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        generate_performance_report()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1) 