import io
import os
import threading
import time
from dataclasses import dataclass
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch
import transformers
from PIL import Image, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor
import metrics as m

MODEL_NAME = os.getenv('CAPTION_MODEL', 'Salesforce/blip-image-captioning-base')
MAX_NEW_TOKENS = int(os.getenv('CAPTION_MAX_TOKENS', '30'))
GEN_KWARGS = {'max_new_tokens': MAX_NEW_TOKENS, 'num_beams': int(os.getenv('CAPTION_BEAMS', '3')), 'no_repeat_ngram_size': 3}
MAX_SIDE = 1024
Image.MAX_IMAGE_PIXELS = 60000000


@dataclass
class CaptionResult:
    caption: str
    inference_s: float
    preprocess_s: float
    width: int
    image_format: str


class Captioner:
    def __init__(self) -> None:
        threads = int(os.getenv('TORCH_THREADS', max(1, (os.cpu_count() or 2) // 2)))
        torch.set_num_threads(threads)
        m.TORCH_THREADS.set(threads)
        t0 = time.perf_counter()
        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME, use_safetensors=True).eval()
        m.MODEL_LOAD_SECONDS.set(time.perf_counter() - t0)
        n_params = sum((p.numel() for p in self.model.parameters()))
        m.MODEL_PARAMETERS.set(n_params)
        m.MODEL_MEMORY_BYTES.set(sum((p.numel() * p.element_size() for p in self.model.parameters())))
        m.MODEL_INFO.info({'model_name': MODEL_NAME, 'device': 'cpu', 'dtype': str(next(self.model.parameters()).dtype), 'torch_version': torch.__version__, 'transformers_version': transformers.__version__, 'num_beams': str(GEN_KWARGS['num_beams'])})
        m.MODEL_LOADED.set(1)
        self._lock = threading.Lock()

    def caption(self, data: bytes, mode: str) -> CaptionResult:
        t0 = time.perf_counter()

        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or 'unknown').lower()
            width, height = img.size
            m.IMAGE_MEGAPIXELS.labels(mode=mode).observe(width * height / 1000000.0)
            img = img.convert('RGB')
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            inputs = self.processor(images=img, return_tensors='pt')

        preprocess_s = time.perf_counter() - t0
        m.PREPROCESS_SECONDS.labels(mode=mode).observe(preprocess_s)
        m.IMAGE_FILE_BYTES.labels(mode=mode).observe(len(data))
        m.INFERENCE_QUEUE.inc()
        q0 = time.perf_counter()

        with self._lock:
            m.INFERENCE_QUEUE.dec()
            m.QUEUE_WAIT_SECONDS.labels(mode=mode).observe(time.perf_counter() - q0)
            t1 = time.perf_counter()

            with torch.inference_mode():
                out = self.model.generate(**inputs, **GEN_KWARGS)

            inference_s = time.perf_counter() - t1

        m.INFERENCE_SECONDS.labels(mode=mode).observe(inference_s)
        caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
        m.TOKENS_GENERATED.labels(mode=mode).inc(out.shape[-1])
        m.CAPTION_WORDS.labels(mode=mode).observe(len(caption.split()))
        m.IMAGES_PROCESSED.labels(mode=mode, status='success', image_format=fmt).inc()
        m.LAST_SUCCESS_TS.labels(mode=mode).set_to_current_time()

        return CaptionResult(caption, inference_s, preprocess_s, width, height, fmt)
    
BAD_IMAGE_ERRORS = (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError)
_instance: Captioner | None = None
_instance_lock = threading.Lock()

def get_captioner() -> Captioner:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Captioner()
        return _instance
