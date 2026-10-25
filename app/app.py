import io
import os
import time
import zipfile
from pathlib import PurePosixPath
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import metrics as m
from captioner import BAD_IMAGE_ERRORS, get_captioner

METRICS_PORT = int(os.getenv('METRICS_PORT', '8000'))
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tif', '.tiff'}
MAX_ZIP_IMAGES = int(os.getenv('MAX_ZIP_IMAGES', '200'))
MAX_ZIP_UNCOMPRESSED = 500 * 1024 ** 2
MAX_MEMBER_BYTES = 25 * 1024 ** 2

st.set_page_config(page_title='CPU Image Captioner', page_icon='Page', layout='wide')

@st.cache_resource
def _metrics_server() -> int:
    m.start_metrics_server(METRICS_PORT)
    return METRICS_PORT

def _client_context() -> tuple[str, str, str]:
    headers = st.context.headers
    ip = (headers.get('X-Forwarded-For', '').split(',')[0].strip() or headers.get('X-Real-Ip', '') or st.context.ip_address or '127.0.0.1').removeprefix('::ffff:')
    ctx = get_script_run_ctx()
    session_id = ctx.session_id[:8] if ctx else 'unknown'
    ua = headers.get('User-Agent', '').lower()
    if 'headless' in ua:
        browser = 'headless'
    elif 'edg/' in ua:
        browser = 'edge'
    elif 'firefox' in ua:
        browser = 'firefox'
    elif 'chrome' in ua:
        browser = 'chrome'
    elif 'safari' in ua:
        browser = 'safari'
    else:
        browser = 'other'
    return (ip, session_id, browser)

_metrics_server()

with st.spinner('Loading BLIP model on CPU…'):
    captioner = get_captioner()

client_ip, session_id, browser = _client_context()
m.touch_session(session_id, browser)
m.PAGE_VIEWS.inc()

def _count_request(mode: str, status: str) -> None:
    m.REQUESTS.labels(mode=mode, status=status, client_ip=client_ip, session_id=session_id, browser=browser).inc()

def run_single(upload) -> dict:
    mode = 'single'
    data = upload.getvalue()
    m.UPLOAD_BYTES.labels(mode=mode).inc(len(data))
    m.UPLOAD_SIZE_BYTES.labels(mode=mode).observe(len(data))

    with m.ACTIVE_REQUESTS.labels(mode=mode).track_inprogress(), m.REQUEST_SECONDS.labels(mode=mode).time():
        try:
            res = captioner.caption(data, mode)
        except BAD_IMAGE_ERRORS as e:
            m.ERRORS.labels(mode=mode, stage='decode', error_type=type(e).__name__).inc()
            m.IMAGES_PROCESSED.labels(mode=mode, status='error', image_format='unknown').inc()
            _count_request(mode, 'error')
            return {'error': f'Could not read image: {e}'}
        except Exception as e:
            m.ERRORS.labels(mode=mode, stage='inference', error_type=type(e).__name__).inc()
            _count_request(mode, 'error')
            return {'error': f'Inference failed: {e}'}
        
    m.CLIENT_IMAGES.labels(client_ip=client_ip, session_id=session_id).inc()
    _count_request(mode, 'success')
    return {'result': res}

def _select_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    chosen, total = ([], 0)

    for info in zf.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.parts[0] == '__MACOSX' or path.name.startswith('.'):
            m.ZIP_ENTRIES_SKIPPED.labels(reason='hidden').inc()
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            m.ZIP_ENTRIES_SKIPPED.labels(reason='not_image').inc()
            continue
        if info.file_size > MAX_MEMBER_BYTES:
            m.ZIP_ENTRIES_SKIPPED.labels(reason='too_large').inc()
            continue
        if len(chosen) >= MAX_ZIP_IMAGES or total + info.file_size > MAX_ZIP_UNCOMPRESSED:
            m.ZIP_ENTRIES_SKIPPED.labels(reason='over_limit').inc()
            continue
        total += info.file_size
        chosen.append(info)

    return chosen

def run_bulk(upload, progress) -> dict:
    mode = 'bulk'
    data = upload.getvalue()
    m.UPLOAD_BYTES.labels(mode=mode).inc(len(data))
    m.UPLOAD_SIZE_BYTES.labels(mode=mode).observe(len(data))
    rows, n_ok, n_err, n_done = ([], 0, 0, 0)
    t0 = time.perf_counter()

    with m.ACTIVE_REQUESTS.labels(mode=mode).track_inprogress(), m.REQUEST_SECONDS.labels(mode=mode).time():
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            m.ERRORS.labels(mode=mode, stage='unzip', error_type=type(e).__name__).inc()
            m.BULK_JOBS.labels(status='rejected').inc()
            _count_request(mode, 'rejected')
            return {'error': 'Not a valid ZIP archive.'}
        
        with zf:
            members = _select_members(zf)
            if not members:
                m.BULK_JOBS.labels(status='rejected').inc()
                _count_request(mode, 'rejected')
                return {'error': 'ZIP contains no supported images.'}
            m.LAST_BULK_JOB_SIZE.set(len(members))
            m.BULK_JOB_IMAGES.observe(len(members))
            m.BULK_IMAGES_REMAINING.inc(len(members))

            try:
                for i, info in enumerate(members, 1):
                    name = info.filename
                    try:
                        res = captioner.caption(zf.read(info), mode)
                        rows.append({'file': name, 'caption': res.caption, 'resolution': f'{res.width}x{res.height}', 'inference_s': round(res.inference_s, 3)})
                        n_ok += 1
                        m.CLIENT_IMAGES.labels(client_ip=client_ip, session_id=session_id).inc()
                    except BAD_IMAGE_ERRORS as e:
                        n_err += 1
                        m.ERRORS.labels(mode=mode, stage='decode', error_type=type(e).__name__).inc()
                        m.IMAGES_PROCESSED.labels(mode=mode, status='error', image_format='unknown').inc()
                        rows.append({'file': name, 'caption': f'unreadable: {e}'})
                    except Exception as e:
                        n_err += 1
                        m.ERRORS.labels(mode=mode, stage='inference', error_type=type(e).__name__).inc()
                        rows.append({'file': name, 'caption': f'failed: {e}'})
                    finally:
                        n_done += 1
                        m.BULK_IMAGES_REMAINING.dec()

                    progress.progress(i / len(members), text=f'{i}/{len(members)}: {name}')

            except BaseException:
                m.BULK_IMAGES_REMAINING.dec(len(members) - n_done)
                m.BULK_JOBS.labels(status='aborted').inc()
                _count_request(mode, 'error')
                raise
            
    elapsed = time.perf_counter() - t0
    m.BULK_JOB_SECONDS.observe(elapsed)
    status = 'success' if n_err == 0 else 'partial' if n_ok else 'failed'
    m.BULK_JOBS.labels(status=status).inc()
    _count_request(mode, 'success' if n_ok else 'error')
    return {'rows': rows, 'ok': n_ok, 'err': n_err, 'elapsed': elapsed}

st.title('CPU Image Captioner')
st.caption(f"Model: `{os.getenv('CAPTION_MODEL', 'Salesforce/blip-image-captioning-base')}` · metrics on `:{METRICS_PORT}/metrics` · session `{session_id}` · client `{client_ip}`")
single_tab, bulk_tab = st.tabs(['Single image', 'Bulk (ZIP)'])

with single_tab:
    up = st.file_uploader('Upload an image', type=[e[1:] for e in IMAGE_EXTS], key='single_upload')
    if up is not None:
        cache = st.session_state.setdefault('single_cache', {})
        if up.file_id not in cache:
            with st.spinner('Captioning…'):
                cache[up.file_id] = run_single(up)
        out = cache[up.file_id]
        left, right = st.columns([2, 3])
        with left:
            if 'error' not in out:
                st.image(up.getvalue(), width='stretch')
        with right:
            if 'error' in out:
                st.error(out['error'])
            else:
                r = out['result']
                st.markdown(f'### “{r.caption}”')
                st.write(f'{r.width}×{r.height} {r.image_format.upper()} · inference {r.inference_s:.2f}s · preprocess {r.preprocess_s * 1000:.0f} ms')
                st.success('done', icon='✅')
with bulk_tab:
    st.write(f'Upload a ZIP of images (max {MAX_ZIP_IMAGES} images, 25 MB each). Non-image and unreadable files are skipped, not fatal.')
    zup = st.file_uploader('Upload a ZIP', type=['zip'], key='bulk_upload')
    if zup is not None:
        cache = st.session_state.setdefault('bulk_cache', {})
        if zup.file_id not in cache:
            bar = st.progress(0.0, text='Starting…')
            cache[zup.file_id] = run_bulk(zup, bar)
            bar.empty()
        
        out = cache[zup.file_id]
        if 'error' in out:
            st.error(out['error'])
        else:
            st.success(f"Captioned {out['ok']} images ({out['err']} failed) in {out['elapsed']:.1f}s", icon='✅')
            df = pd.DataFrame(out['rows'])
            st.dataframe(df, width='stretch', hide_index=True)
            st.download_button('Download captions.csv', df.to_csv(index=False).encode(), file_name='captions.csv', mime='text/csv')
