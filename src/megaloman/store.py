"""Content-addressed blob store, step cache, model-call transcripts. Everything lives under one root dir."""
import hashlib, json, os

ROOT = os.environ.get('MEGALOMAN_HOME', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
CACHE = os.path.join(ROOT, 'cache')


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)


def key(obj) -> str:
    return sha(canon(obj).encode())


def put_blob(data: bytes, ext='png') -> str:
    h = sha(data)
    p = blob_path(h, ext)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, p)
    return h


def blob_path(h: str, ext='png') -> str:
    return os.path.join(CACHE, 'blobs', h[:2], f'{h}.{ext}')


def get_blob(h: str, ext='png') -> bytes:
    with open(blob_path(h, ext), 'rb') as f:
        return f.read()


def _json_path(kind, k):
    return os.path.join(CACHE, kind, k[:2], f'{k}.json')


def get_json(kind, k):
    p = _json_path(kind, k)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def put_json(kind, k, obj):
    p = _json_path(kind, k)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


def file_hash(path) -> str:
    with open(path, 'rb') as f:
        return sha(f.read())
