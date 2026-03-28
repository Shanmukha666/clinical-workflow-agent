"""
Production OCR Service for Clinical Document Processing
Supports multiple OCR engines with fallback, preprocessing, and confidence scoring
"""

from __future__ import annotations

import os
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

# OCR Libraries
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Optional: Advanced OCR engines
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

class OCRConfig:
    """OCR configuration settings"""
    
    # Tesseract path (platform specific)
    if os.name == 'nt':  # Windows
        TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    else:  # Linux/Mac
        TESSERACT_PATH = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")
    
    # Set Tesseract path if exists
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    
    # OCR Engine preference (tesseract, easyocr, paddle, hybrid)
    PREFERRED_ENGINE = os.getenv("OCR_ENGINE", "hybrid")
    
    # Image preprocessing settings
    PREPROCESSING = {
        "resize_factor": 2,  # Upscale for better OCR
        "denoise": True,
        "threshold": True,
        "deskew": True
    }
    
    # Confidence thresholds
    MIN_CONFIDENCE = 0.6
    FALLBACK_ENGINE = "tesseract"
    
    # Cache settings
    CACHE_OCR_RESULTS = os.getenv("CACHE_OCR", "True").lower() == "true"
    CACHE_DIR = Path("/tmp/ocr_cache")
    
    # Language
    LANGUAGES = os.getenv("OCR_LANGUAGES", "eng").split(",")
    
    # Timeout (seconds)
    TIMEOUT = 60


class OCRResult:
    """OCR result with metadata"""
    
    def __init__(self, text: str, confidence: float, engine: str, 
                 processing_time: float, preprocessing_applied: List[str]):
        self.text = text
        self.confidence = confidence
        self.engine = engine
        self.processing_time = processing_time
        self.preprocessing_applied = preprocessing_applied
        self.timestamp = datetime.now().isoformat()
        self.word_count = len(text.split())
        self.char_count = len(text)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "engine": self.engine,
            "processing_time_ms": self.processing_time,
            "preprocessing_applied": self.preprocessing_applied,
            "timestamp": self.timestamp,
            "word_count": self.word_count,
            "char_count": self.char_count
        }
    
    def is_reliable(self) -> bool:
        return self.confidence >= OCRConfig.MIN_CONFIDENCE and self.word_count > 10


# ==================== Image Preprocessing ====================

class ImagePreprocessor:
    """Advanced image preprocessing for better OCR"""
    
    @staticmethod
    def preprocess(image_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess image for optimal OCR
        Returns (processed_image, applied_steps)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        applied_steps = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        applied_steps.append("grayscale")
        
        # 1. Denoise
        if OCRConfig.PREPROCESSING["denoise"]:
            gray = cv2.fastNlMeansDenoising(gray, h=30)
            applied_steps.append("denoise")
        
        # 2. Deskew (fix rotation)
        if OCRConfig.PREPROCESSING["deskew"]:
            gray = ImagePreprocessor._deskew(gray)
            applied_steps.append("deskew")
        
        # 3. Resize (upscale for better recognition)
        if OCRConfig.PREPROCESSING["resize_factor"] > 1:
            height, width = gray.shape
            new_size = (width * OCRConfig.PREPROCESSING["resize_factor"], 
                       height * OCRConfig.PREPROCESSING["resize_factor"])
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)
            applied_steps.append(f"resize_{OCRConfig.PREPROCESSING['resize_factor']}x")
        
        # 4. Adaptive thresholding
        if OCRConfig.PREPROCESSING["threshold"]:
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            applied_steps.append("adaptive_threshold")
        
        # 5. Morphological operations to clean text
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        applied_steps.append("morphological_cleanup")
        
        return gray, applied_steps
    
    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """Deskew image to fix rotation"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 2:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), 
                                   flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
        
        return image


# ==================== OCR Engines ====================

class TesseractEngine:
    """Tesseract OCR engine"""
    
    @staticmethod
    def extract(image: np.ndarray) -> Tuple[str, float]:
        """Extract text using Tesseract"""
        try:
            # Configure Tesseract
            config = f'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.:/-% '
            
            text = pytesseract.image_to_string(image, config=config, lang='+'.join(OCRConfig.LANGUAGES))
            
            # Get confidence data
            confidence_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in confidence_data['conf'] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.5
            
            return text, avg_confidence
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return "", 0.0


class EasyOCREngine:
    """EasyOCR engine (supports multiple languages)"""
    
    _reader = None
    
    @classmethod
    def get_reader(cls):
        if cls._reader is None and EASYOCR_AVAILABLE:
            cls._reader = easyocr.Reader(OCRConfig.LANGUAGES, gpu=False)
        return cls._reader
    
    @classmethod
    def extract(cls, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using EasyOCR"""
        if not EASYOCR_AVAILABLE:
            return "", 0.0
        
        try:
            reader = cls.get_reader()
            if reader is None:
                return "", 0.0
            
            # Convert to RGB for EasyOCR
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            results = reader.readtext(image)
            
            text = " ".join([result[1] for result in results])
            confidence = sum([result[2] for result in results]) / len(results) if results else 0.0
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return "", 0.0


class PaddleOCREngine:
    """PaddleOCR engine"""
    
    _ocr = None
    
    @classmethod
    def get_ocr(cls):
        if cls._ocr is None and PADDLE_AVAILABLE:
            cls._ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return cls._ocr
    
    @classmethod
    def extract(cls, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using PaddleOCR"""
        if not PADDLE_AVAILABLE:
            return "", 0.0
        
        try:
            ocr = cls.get_ocr()
            if ocr is None:
                return "", 0.0
            
            # Save temp image for PaddleOCR
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                result = ocr.ocr(tmp.name, cls=True)
                os.unlink(tmp.name)
            
            if not result or not result[0]:
                return "", 0.0
            
            text = " ".join([line[1][0] for line in result[0]])
            confidence = sum([line[1][1] for line in result[0]]) / len(result[0]) if result[0] else 0.0
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return "", 0.0


# ==================== Main OCR Service ====================

class OCRService:
    """
    Main OCR service with multiple engine support, caching, and fallback
    """
    
    def __init__(self):
        self.cache_dir = OCRConfig.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize engines
        self.engines = {
            "tesseract": TesseractEngine,
            "easyocr": EasyOCREngine if EASYOCR_AVAILABLE else None,
            "paddle": PaddleOCREngine if PADDLE_AVAILABLE else None
        }
        
        logger.info(f"OCR Service initialized. Engines: {[k for k, v in self.engines.items() if v]}")
    
    async def extract_text(self, image_path: str) -> OCRResult:
        """
        Extract text from image with automatic engine selection
        """
        start_time = datetime.now()
        
        # Check cache
        cache_key = self._get_cache_key(image_path)
        if OCRConfig.CACHE_OCR_RESULTS:
            cached = self._get_cached(cache_key)
            if cached:
                logger.info(f"OCR cache hit for {image_path}")
                return cached
        
        # Preprocess image
        try:
            processed_image, preprocessing_steps = ImagePreprocessor.preprocess(image_path)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            processed_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            preprocessing_steps = ["none"]
        
        # Try engines in order of preference
        result = None
        
        if OCRConfig.PREFERRED_ENGINE == "hybrid":
            # Try multiple engines and pick best
            results = await self._try_multiple_engines(processed_image)
            if results:
                result = max(results, key=lambda x: x.confidence)
        else:
            # Try preferred engine, fallback to others
            engine = self.engines.get(OCRConfig.PREFERRED_ENGINE)
            if engine:
                text, confidence = engine.extract(processed_image)
                result = OCRResult(
                    text=text,
                    confidence=confidence,
                    engine=OCRConfig.PREFERRED_ENGINE,
                    processing_time=(datetime.now() - start_time).total_seconds() * 1000,
                    preprocessing_applied=preprocessing_steps
                )
                
                if not result.is_reliable():
                    # Fallback to Tesseract
                    text, confidence = TesseractEngine.extract(processed_image)
                    fallback_result = OCRResult(
                        text=text,
                        confidence=confidence,
                        engine="tesseract_fallback",
                        processing_time=(datetime.now() - start_time).total_seconds() * 1000,
                        preprocessing_applied=preprocessing_steps
                    )
                    if fallback_result.is_reliable():
                        result = fallback_result
        
        if result is None or not result.text:
            # Last resort: raw Tesseract with no preprocessing
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            text, confidence = TesseractEngine.extract(img)
            result = OCRResult(
                text=text or "",
                confidence=confidence,
                engine="tesseract_raw",
                processing_time=(datetime.now() - start_time).total_seconds() * 1000,
                preprocessing_applied=["none"]
            )
        
        # Cache result
        if OCRConfig.CACHE_OCR_RESULTS and result.text:
            self._cache_result(cache_key, result)
        
        logger.info(f"OCR completed: engine={result.engine}, confidence={result.confidence:.2%}, words={result.word_count}")
        
        return result
    
    async def _try_multiple_engines(self, image: np.ndarray) -> List[OCRResult]:
        """Try multiple OCR engines in parallel"""
        results = []
        
        tasks = []
        for name, engine in self.engines.items():
            if engine:
                tasks.append(self._run_engine(engine, image, name))
        
        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for item in completed:
                if isinstance(item, OCRResult) and item.text:
                    results.append(item)
        
        return results
    
    async def _run_engine(self, engine_class, image: np.ndarray, name: str) -> OCRResult:
        """Run single OCR engine"""
        loop = asyncio.get_event_loop()
        start = datetime.now()
        
        try:
            text, confidence = await loop.run_in_executor(
                self.executor, engine_class.extract, image
            )
            
            return OCRResult(
                text=text,
                confidence=confidence,
                engine=name,
                processing_time=(datetime.now() - start).total_seconds() * 1000,
                preprocessing_applied=[]
            )
        except Exception as e:
            logger.error(f"Engine {name} failed: {e}")
            return None
    
    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key from image content"""
        with open(image_path, 'rb') as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[OCRResult]:
        """Get cached OCR result"""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return OCRResult(
                    text=data['text'],
                    confidence=data['confidence'],
                    engine=data['engine'],
                    processing_time=data['processing_time_ms'],
                    preprocessing_applied=data['preprocessing_applied']
                )
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        return None
    
    def _cache_result(self, key: str, result: OCRResult):
        """Cache OCR result"""
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result.to_dict(), f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


# ==================== Singleton Instance ====================

_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get singleton OCR service instance"""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service


async def extract_text_from_image(image_path: str) -> str:
    """Async wrapper for backward compatibility"""
    service = get_ocr_service()
    result = await service.extract_text(image_path)
    return result.text


# ==================== Test Block ====================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("🚀 Testing OCR Service...\n")
        
        service = get_ocr_service()
        
        test_image = "sample.png"  # Change this
        if os.path.exists(test_image):
            result = await service.extract_text(test_image)
            
            print(f"✅ OCR Result:")
            print(f"   Engine: {result.engine}")
            print(f"   Confidence: {result.confidence:.2%}")
            print(f"   Words: {result.word_count}")
            print(f"   Time: {result.processing_time:.2f}ms")
            print(f"   Preprocessing: {result.preprocessing_applied}")
            print(f"\n📝 Extracted Text:\n{result.text[:500]}")
        else:
            print(f"❌ Test image not found: {test_image}")
    
    asyncio.run(test())