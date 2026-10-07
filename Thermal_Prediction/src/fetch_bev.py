"""Download the public KU Leuven BEV archive; verify publisher checksum."""
from pathlib import Path
import hashlib,json,urllib.request,shutil
import ssl, certifi
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))))

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/kuleuven_bev'
RAW.mkdir(parents=True,exist_ok=True)
META='https://rdr.kuleuven.be/api/datasets/:persistentId/?persistentId=doi:10.48804/8KPDTW'
j=json.load(urllib.request.urlopen(META,timeout=30))
(RAW/'record_metadata.json').write_text(json.dumps(j,ensure_ascii=False,indent=2))
for x in j['data']['latestVersion']['files']:
    d=x['dataFile']
    if not d['filename'].endswith(('.zip','.txt')):continue
    name='BEV_energy_dynamic_data_V2.zip' if d['filename'].endswith('.zip') else d['filename']
    dest=RAW/name
    kind=d['checksum']['type'].lower();expected=d['checksum']['value']
    def checksum(p):
        h=hashlib.new(kind)
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()
    if dest.exists() and checksum(dest)==expected:
        print('verified cached',name);continue
    tmp=dest.with_suffix(dest.suffix+'.part')
    with urllib.request.urlopen(f"https://rdr.kuleuven.be/api/access/datafile/{d['id']}",timeout=60) as r,tmp.open('wb') as f:
        shutil.copyfileobj(r,f)
    if tmp.stat().st_size!=d['filesize'] or checksum(tmp)!=expected:raise RuntimeError('Download checksum/size mismatch')
    tmp.replace(dest);print('downloaded and verified',name)
