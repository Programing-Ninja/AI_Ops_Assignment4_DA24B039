import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import metrics
from captioner import get_captioner
from streamlit.web import cli as stcli

if __name__ == '__main__':
    
    port = int(os.getenv('METRICS_PORT', '8000'))
    metrics.start_metrics_server(port)
    print(f'metrics: http://localhost:{port}/metrics', flush=True)
    get_captioner()
    print('model loaded', flush=True)
    sys.argv = ['streamlit', 'run', str(APP_DIR / 'app.py'), *sys.argv[1:]]
    sys.exit(stcli.main())
