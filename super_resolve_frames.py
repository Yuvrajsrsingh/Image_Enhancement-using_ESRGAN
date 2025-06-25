
import os
import cv2
import torch
import numpy as np
import gc
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

from scipy import ndimage
import json



if torch.cuda.is_available():
    torch.cuda.empty_cache()

input_dir = 'extracted_frames'
output_dir = 'esrgan_output/enhanced_frames'
ocr_output_dir = 'esrgan_output/ocr_results'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ocr_output_dir, exist_ok=True)

model_path = 'C:/Users/Asus/Desktop/esrgan/model/RealESRGAN_x4plus.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def advanced_image_preprocessing(image):
    """Advanced preprocessing for better text recognition"""
    
 
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  
    
   
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    
    image_np = np.array(image)
    
    
    image_np = cv2.bilateralFilter(image_np, 9, 75, 75)
    
    
    gamma = 0.7
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image_np = cv2.LUT(image_np, table)
    
    return image_np

def text_specific_enhancement(image):
    """Specific enhancements for text readability"""
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    

    kernel = np.ones((2,2), np.uint8)
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    
   
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb

def extract_text_with_confidence(image, min_confidence=30):
    """Extract text using Tesseract with confidence scoring"""
    
   
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    
    
    ocr_configs = [
        '--oem 3 --psm 6',  
        '--oem 3 --psm 8',  
        '--oem 3 --psm 7',  
        '--oem 3 --psm 11', 
        '--oem 3 --psm 13'  
    ]
    
    best_result = {"text": "", "confidence": 0, "config": ""}
    all_results = []
    
    for config in ocr_configs:
        try:
           
            data = pytesseract.image_to_data(image_pil, config=config, output_type=pytesseract.Output.DICT)
            
            confident_text = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > min_confidence:
                    text = data['text'][i].strip()
                    if text:
                        confident_text.append(text)
                        confidences.append(int(data['conf'][i]))
            
            if confident_text:
                full_text = ' '.join(confident_text)
                avg_confidence = np.mean(confidences) if confidences else 0
                
                result = {
                    "text": full_text,
                    "confidence": avg_confidence,
                    "config": config,
                    "word_count": len(confident_text)
                }
                
                all_results.append(result)
                
                if avg_confidence > best_result["confidence"]:
                    best_result = result
                    
        except Exception as e:
            print(f"Error with OCR config {config}: {e}")
            continue
    
    return best_result, all_results


print("Loading Real-ESRGAN model...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

try:
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=512,  
        tile_pad=10,
        pre_pad=0,
        half=True if device.type == 'cuda' else False,
        device=device
    )
    print("Real-ESRGAN loaded successfully")
except Exception as e:
    print(f"Error loading Real-ESRGAN: {e}")
    exit(1)


image_files = [f for f in sorted(os.listdir(input_dir)) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

total_files = len(image_files)
print(f"Found {total_files} image files to process")

all_ocr_results = {}


for i, fname in enumerate(image_files):
    print(f"\nProcessing {i+1}/{total_files}: {fname}")
    
    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, f"enhanced_{fname}")
    
    try:
    
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f" Could not load {fname}")
            continue
        
        print(f"  Original size: {img.shape[1]}x{img.shape[0]}")
        
       
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        print(" Applying Real-ESRGAN...")
        enhanced, _ = upsampler.enhance(img_rgb, outscale=4)
        print(f"  Enhanced size: {enhanced.shape[1]}x{enhanced.shape[0]}")
        
        
        print("Advanced preprocessing...")
        enhanced = advanced_image_preprocessing(enhanced)
        
        
        print("Text-specific enhancement...")
        text_enhanced = text_specific_enhancement(enhanced)
        
       
        enhanced_bgr = cv2.cvtColor(text_enhanced, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, enhanced_bgr)
        print(f"Saved enhanced image")
        
        
        print("  🔍 Extracting text...")
        best_ocr, all_ocr = extract_text_with_confidence(text_enhanced)
        
        if best_ocr["text"]:
            print(f" Text found (confidence: {best_ocr['confidence']:.1f}%)")
            print(f"     Text: '{best_ocr['text'][:100]}{'...' if len(best_ocr['text']) > 100 else ''}'")
        else:
            print("No confident text found")
        
        
        all_ocr_results[fname] = {
            "best_result": best_ocr,
            "all_results": all_ocr,
            "original_size": f"{img.shape[1]}x{img.shape[0]}",
            "enhanced_size": f"{enhanced.shape[1]}x{enhanced.shape[0]}"
        }
        
        
        del img, img_rgb, enhanced, text_enhanced, enhanced_bgr
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        continue


ocr_results_path = os.path.join(ocr_output_dir, 'ocr_results.json')
with open(ocr_results_path, 'w', encoding='utf-8') as f:
    json.dump(all_ocr_results, f, indent=2, ensure_ascii=False)


print(f"\n📊 Processing Summary:")
print(f"Total files processed: {len(all_ocr_results)}")

files_with_text = sum(1 for result in all_ocr_results.values() 
                     if result['best_result']['text'])
print(f"Files with detected text: {files_with_text}")

if files_with_text > 0:
    avg_confidence = np.mean([result['best_result']['confidence'] 
                             for result in all_ocr_results.values() 
                             if result['best_result']['text']])
    print(f"Average confidence: {avg_confidence:.1f}%")

summary_path = os.path.join(ocr_output_dir, 'text_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("EXTRACTED TEXT SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    
    for fname, result in all_ocr_results.items():
        if result['best_result']['text']:
            f.write(f"File: {fname}\n")
            f.write(f"Confidence: {result['best_result']['confidence']:.1f}%\n")
            f.write(f"Text: {result['best_result']['text']}\n")
            f.write("-" * 30 + "\n\n")

print(f"\n✅ Complete! Check results in:")
print(f"   Enhanced images: {output_dir}")
print(f"   OCR results: {ocr_results_path}")
print(f"   Text summary: {summary_path}")


if device.type == 'cuda':
    torch.cuda.empty_cache()
gc.collect()