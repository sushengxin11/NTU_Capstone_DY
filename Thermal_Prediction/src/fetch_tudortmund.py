"""Fetch publisher ZIP members with range requests; verify CRC and provenance."""
import concurrent.futures, hashlib, io, json, pathlib, struct, urllib.request, zipfile, zlib
import ssl, certifi
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))))

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/tudortmund'
URL = 'https://zenodo.org/api/records/14065331/files/dataset_merged.zip/content'
SIZE = 207735184

def get_range(start, end):
    req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
            raise RuntimeError('Unexpected range response')
        b = r.read()
    if len(b) != end-start+1:
        raise RuntimeError('Incomplete range')
    return b

class Remote(io.RawIOBase):
    def __init__(self): self.pos = 0
    def seekable(self): return True
    def seek(self, n, w=0):
        self.pos = n if w == 0 else self.pos+n if w == 1 else SIZE+n
        return self.pos
    def tell(self): return self.pos
    def read(self, n=-1):
        if n < 0: n = SIZE-self.pos
        if not n: return b''
        b = get_range(self.pos, min(self.pos+n, SIZE)-1)
        self.pos += len(b)
        return b

def fetch(info):
    dest = RAW / ('selected_raw_monitor' if 'dataset_raw' in URL else 'selected_merged') / info.filename
    if dest.exists():
        b = dest.read_bytes()
        if len(b) == info.file_size and zlib.crc32(b) == info.CRC:
            return {'member': info.filename, 'bytes':len(b), 'sha256':hashlib.sha256(b).hexdigest(), 'crc32_verified':True}
    header = get_range(info.header_offset, info.header_offset+29)
    signature, = struct.unpack_from('<I', header)
    if signature != 0x04034b50: raise RuntimeError('ZIP header mismatch')
    name_len, extra_len = struct.unpack_from('<HH', header, 26)
    start = info.header_offset+30+name_len+extra_len
    packed = get_range(start, start+info.compress_size-1)
    b = zlib.decompress(packed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else packed
    if len(b) != info.file_size or zlib.crc32(b) != info.CRC: raise RuntimeError('Member CRC mismatch')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b)
    return {'member':info.filename, 'bytes':len(b), 'sha256':hashlib.sha256(b).hexdigest(), 'crc32_verified':True}

def main():
    global URL, SIZE
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-monitor', action='store_true')
    args=parser.parse_args()
    kind='raw_monitor' if args.raw_monitor else 'merged'
    if args.raw_monitor:
        URL='https://zenodo.org/api/records/14065331/files/dataset_raw.zip/content'
        SIZE=389787805
    z = zipfile.ZipFile(Remote())
    infos = z.infolist()
    (RAW/f'{kind}_archive_index.json').write_text(json.dumps([{'member':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size} for i in infos],indent=2))
    # CCS cars and unidirectional DC tests: no discharge or CHAdeMO samples.
    selected = [i for i in infos if i.filename.endswith('.csv') and '_DC_' in i.filename
                and ('_charging_' in i.filename or '_ts_unidir_' in i.filename)
                and not i.filename.startswith(('Nissan Leaf/', 'Mitsubishi Eclipse Cross/'))]
    if args.raw_monitor:
        selected=[i for i in infos if i.filename.endswith('.csv') and 'EV-Monitor' in i.filename
                  ]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch, selected):
            rows.append(row)
            print('verified',row['member'],flush=True)
    (RAW/f'{kind}_selected_manifest.json').write_text(json.dumps({'source':URL,'record':'https://zenodo.org/records/14065331','license':'CC-BY-4.0','scope':kind + ' candidate subset','whole_archive_downloaded':False,'files':rows},indent=2))
    print('verified members',len(rows))
if __name__ == '__main__':main()
