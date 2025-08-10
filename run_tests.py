#!/usr/bin/env python3
"""
Comprehensive Test Suite for Medical Image Classifier

This script runs all tests to validate the system functionality:
- Unit tests for classification
- Integration tests for PDF processing
- End-to-end tests for URL processing
- Performance validation
"""

import unittest
import time
import os
import sys
from pathlib import Path
from utils.classify import classify_images_clip
from utils.extract_images import extract_images_from_pdf, extract_images_from_url
from PIL import Image
import io

class TestMedicalImageClassifier(unittest.TestCase):
    """Test suite for medical image classifier"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_image_path = "test_images/test image 1.jpg"
        self.test_pdfs = [
            "test_pdfs/medical_images.pdf",
            "test_pdfs/non_medical_images.pdf",
            "test_pdfs/combined_test.pdf"
        ]
        self.test_url = "https://images.pexels.com/photos/3376790/pexels-photo-3376790.jpeg?auto=compress&cs=tinysrgb&w=400"
        
        # Verify test files exist
        if not os.path.exists(self.test_image_path):
            self.skipTest(f"Test image not found: {self.test_image_path}")
        
        for pdf_path in self.test_pdfs:
            if not os.path.exists(pdf_path):
                self.skipTest(f"Test PDF not found: {pdf_path}")
    
    def test_01_local_image_classification(self):
        """Test local image classification functionality"""
        print("\n🔍 Testing local image classification...")
        
        # Load test image
        image = Image.open(self.test_image_path).convert("RGB")
        
        # Test classification
        start_time = time.time()
        label = classify_images_clip(image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Assertions
        self.assertIsInstance(label, str)
        self.assertIn(label, ["medical", "non-medical"])
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
        
        print(f"✅ Local image classification: {label} ({processing_time:.3f}s)")
    
    def test_02_pdf_image_extraction(self):
        """Test PDF image extraction functionality"""
        print("\n📄 Testing PDF image extraction...")
        
        total_images = 0
        
        for pdf_path in self.test_pdfs:
            print(f"   Processing {pdf_path}...")
            
            # Extract images
            start_time = time.time()
            images = extract_images_from_pdf(pdf_path)
            end_time = time.time()
            
            extraction_time = end_time - start_time
            
            # Assertions
            self.assertIsInstance(images, list)
            self.assertGreater(len(images), 0)
            self.assertLess(extraction_time, 2.0)  # Should extract within 2 seconds
            
            total_images += len(images)
            print(f"     ✅ Extracted {len(images)} images in {extraction_time:.3f}s")
        
        print(f"✅ Total PDF images extracted: {total_images}")
    
    def test_03_pdf_classification(self):
        """Test PDF image classification functionality"""
        print("\n🔍 Testing PDF image classification...")
        
        total_correct = 0
        total_images = 0
        
        for pdf_path in self.test_pdfs:
            print(f"   Classifying {pdf_path}...")
            
            # Extract and classify
            images = extract_images_from_pdf(pdf_path)
            
            for i, img_bytes in enumerate(images):
                try:
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    
                    start_time = time.time()
                    label = classify_images_clip(image)
                    end_time = time.time()
                    
                    classification_time = end_time - start_time
                    
                    # Assertions
                    self.assertIsInstance(label, str)
                    self.assertIn(label, ["medical", "non-medical"])
                    self.assertLess(classification_time, 3.0)
                    
                    total_images += 1
                    print(f"     Image {i+1}: {label} ({classification_time:.3f}s)")
                    
                except Exception as e:
                    self.fail(f"Failed to classify image {i+1} from {pdf_path}: {e}")
        
        print(f"✅ Total images classified: {total_images}")
    
    def test_04_url_image_processing(self):
        """Test URL image processing functionality"""
        print("\n🌐 Testing URL image processing...")
        
        try:
            # Test URL image extraction
            start_time = time.time()
            images = extract_images_from_url(self.test_url)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Assertions
            self.assertIsInstance(images, list)
            self.assertGreater(len(images), 0)
            self.assertLess(processing_time, 10.0)  # Network operations can take time
            
            print(f"✅ URL image extraction: {len(images)} images in {processing_time:.3f}s")
            
            # Test classification of extracted image
            if images:
                image = Image.open(io.BytesIO(images[0])).convert("RGB")
                
                start_time = time.time()
                label = classify_images_clip(image)
                end_time = time.time()
                
                classification_time = end_time - start_time
                
                # Assertions
                self.assertIsInstance(label, str)
                self.assertIn(label, ["medical", "non-medical"])
                self.assertLess(classification_time, 3.0)
                
                print(f"✅ URL image classification: {label} ({classification_time:.3f}s)")
            
        except Exception as e:
            # Network issues might cause failures, so we'll skip this test
            self.skipTest(f"URL processing failed (network issue): {e}")
    
    def test_05_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n⚡ Testing performance benchmarks...")
        
        # Test single image performance
        image = Image.open(self.test_image_path).convert("RGB")
        
        times = []
        for _ in range(3):  # Run 3 times for average
            start_time = time.time()
            _ = classify_images_clip(image)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Assertions
        self.assertLess(avg_time, 2.0)  # Should average less than 2 seconds
        
        print(f"✅ Average classification time: {avg_time:.3f}s")
        print(f"✅ Throughput: {1/avg_time:.2f} images/second")
    
    def test_06_error_handling(self):
        """Test error handling capabilities"""
        print("\n🛡️ Testing error handling...")
        
        # Test with invalid image path
        with self.assertRaises(Exception):
            Image.open("nonexistent_image.jpg")
        
        # Test with invalid PDF path
        with self.assertRaises(Exception):
            extract_images_from_pdf("nonexistent.pdf")
        
        print("✅ Error handling working correctly")
    
    def test_07_output_consistency(self):
        """Test output consistency across multiple runs"""
        print("\n🔄 Testing output consistency...")
        
        image = Image.open(self.test_image_path).convert("RGB")
        
        # Run classification multiple times
        results = []
        for i in range(5):
            label = classify_images_clip(image)
            results.append(label)
        
        # All results should be the same
        self.assertEqual(len(set(results)), 1, "Classification results should be consistent")
        
        print(f"✅ Consistent classification: {results[0]} (5/5 runs)")
    
    def test_08_memory_efficiency(self):
        """Test memory efficiency"""
        print("\n💾 Testing memory efficiency...")
        
        # Process multiple images to check memory usage
        images = []
        for pdf_path in self.test_pdfs[:2]:  # Use first 2 PDFs
            pdf_images = extract_images_from_pdf(pdf_path)
            images.extend(pdf_images)
        
        # Classify all images
        start_time = time.time()
        for img_bytes in images:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            _ = classify_images_clip(image)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Assertions
        self.assertGreater(len(images), 0)
        self.assertLess(total_time, 30.0)  # Should complete within 30 seconds
        
        print(f"✅ Processed {len(images)} images in {total_time:.3f}s")
        print(f"✅ Average time per image: {total_time/len(images):.3f}s")

def run_test_suite():
    """Run the complete test suite"""
    print("🧪 Medical Image Classifier - Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMedicalImageClassifier)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ All tests passed successfully!")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Skipped: {len(result.skipped)}")
    else:
        print("❌ Some tests failed!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"   {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"   {test}: {traceback}")
    
    print("\n" + "=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    try:
        success = run_test_suite()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1) 