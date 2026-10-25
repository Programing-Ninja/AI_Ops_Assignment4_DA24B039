import threading
import time
from prometheus_client import Counter, Gauge, Histogram, Info, Summary, start_http_server

NS = 'captioner'

REQUESTS = Counter('requests', 'Caption requests (one single-image upload or one ZIP job = one request)', ['mode', 'status', 'client_ip', 'session_id', 'browser'], namespace=NS)

IMAGES_PROCESSED = Counter('images_processed', 'Images that went through the captioning pipeline', ['mode', 'status', 'image_format'], namespace=NS)

ERRORS = Counter('errors', 'Errors raised while handling uploads, by stage and exception type', ['mode', 'stage', 'error_type'], namespace=NS)

BULK_JOBS = Counter('bulk_jobs', 'ZIP bulk jobs by final outcome', ['status'], namespace=NS)

UPLOAD_BYTES = Counter('upload_bytes', 'Raw bytes received from clients', ['mode'], namespace=NS)

ZIP_ENTRIES_SKIPPED = Counter('zip_entries_skipped', 'ZIP members ignored before inference', ['reason'], namespace=NS)

TOKENS_GENERATED = Counter('generated_tokens', 'Decoder tokens produced by the captioning model', ['mode'], namespace=NS)

CLIENT_IMAGES = Counter('client_images', 'Images captioned per client (client tracking)', ['client_ip', 'session_id'], namespace=NS)

SESSIONS_STARTED = Counter('sessions_started', 'Browser sessions that executed the app at least once', ['browser'], namespace=NS)

PAGE_VIEWS = Counter('script_runs', 'Streamlit script reruns (every widget interaction triggers one)', namespace=NS)

ACTIVE_REQUESTS = Gauge('active_requests', 'Requests currently being processed', ['mode'], namespace=NS)

INFERENCE_QUEUE = Gauge('inference_queue_depth', 'Images waiting for the (serialised) model lock', namespace=NS)

BULK_IMAGES_REMAINING = Gauge('bulk_images_remaining', 'Images left across all running bulk jobs', namespace=NS)

LAST_BULK_JOB_SIZE = Gauge('last_bulk_job_images', 'Number of valid images in the most recent bulk job', namespace=NS)

MODEL_LOADED = Gauge('model_loaded', '1 if the captioning model is loaded and ready, else 0', namespace=NS)

MODEL_LOAD_SECONDS = Gauge('model_load_seconds', 'Wall time spent loading the model at startup', namespace=NS)

MODEL_PARAMETERS = Gauge('model_parameters', 'Number of parameters in the loaded model', namespace=NS)

MODEL_MEMORY_BYTES = Gauge('model_weights_bytes', 'Bytes occupied by model weights (params x dtype size)', namespace=NS)

TORCH_THREADS = Gauge('torch_threads', 'Intra-op CPU threads used by torch', namespace=NS)

LAST_SUCCESS_TS = Gauge('last_success_timestamp_seconds', 'Unix time of the last successfully captioned image', ['mode'], namespace=NS)

ACTIVE_SESSIONS = Gauge('active_sessions', 'Distinct sessions seen in the last 5 minutes', namespace=NS)

MODEL_INFO = Info('model', 'Static information about the deployed model', namespace=NS)

INFERENCE_SECONDS = Histogram('inference_seconds', 'Model generate() time per image', ['mode'], buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0), namespace=NS)

PREPROCESS_SECONDS = Histogram('preprocess_seconds', 'Decode + resize + processor() time per image', ['mode'], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0), namespace=NS)

QUEUE_WAIT_SECONDS = Histogram('queue_wait_seconds', 'Time an image waited for the model lock', ['mode'], buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0), namespace=NS)

REQUEST_SECONDS = Histogram('request_duration_seconds', 'End-to-end time per request (single image, or whole ZIP job)', ['mode'], buckets=(0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0), namespace=NS)

IMAGE_MEGAPIXELS = Histogram('image_megapixels', 'Resolution of uploaded images before resizing', ['mode'], buckets=(0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 50.0), namespace=NS)

UPLOAD_SIZE_BYTES = Histogram('upload_size_bytes', 'Size of each uploaded file (image or ZIP)', ['mode'], buckets=(10000.0, 50000.0, 100000.0, 500000.0, 1000000.0, 5000000.0, 10000000.0, 50000000.0, 100000000.0, 200000000.0), namespace=NS)

CAPTION_WORDS = Summary('caption_length_words', 'Words per generated caption', ['mode'], namespace=NS)

BULK_JOB_IMAGES = Summary('bulk_job_images', 'Valid images per bulk job', namespace=NS)

IMAGE_FILE_BYTES = Summary('image_file_bytes', 'Encoded size of each image fed to the model', ['mode'], namespace=NS)

BULK_JOB_SECONDS = Summary('bulk_job_seconds', 'Wall time per bulk job', namespace=NS)

_SESSION_WINDOW_S = 300

_sessions: dict[str, float] = {}
_sessions_lock = threading.Lock()

def touch_session(session_id: str, browser: str) -> None:
    now = time.time()
    with _sessions_lock:
        if session_id not in _sessions:
            SESSIONS_STARTED.labels(browser=browser).inc()
        _sessions[session_id] = now

def _count_active_sessions() -> int:
    cutoff = time.time() - _SESSION_WINDOW_S
    with _sessions_lock:
        for sid in [s for s, t in _sessions.items() if t < cutoff]:
            del _sessions[sid]
        return len(_sessions)

ACTIVE_SESSIONS.set_function(_count_active_sessions)
_server_started = False
_server_lock = threading.Lock()

def start_metrics_server(port: int) -> None:
    global _server_started
    with _server_lock:
        if not _server_started:
            start_http_server(port)
            _server_started = True
