"""Feed saved validation audio into native Rust and compare raw Python outputs.

No audio devices. Align using declared model delay, flush tail with zero hops,
and compare at original gain (no fitted gain, shift search or clipping).
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf


def read_exact(stream,size):
    data=bytearray()
    while len(data)<size:
        chunk=stream.read(size-len(data))
        if not chunk:raise RuntimeError('processor ended before complete output frame')
        data.extend(chunk)
    return bytes(data)


def wsl_path(path):
    p=Path(path).resolve()
    assert p.drive and len(p.drive)==2
    return '/mnt/'+p.drive[0].lower()+p.as_posix()[2:]


def process(binary,bundle,x,parity_mode=True,wsl=False):
    command=(["wsl.exe","-d","Ubuntu","--",wsl_path(binary),'--model-bundle',wsl_path(bundle)]
             if wsl else [str(binary),'--model-bundle',str(bundle)])
    if parity_mode:command.append('--parity-mode')
    p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    try:
        ready=p.stderr.readline().decode(errors='replace').strip()
        if not ready.startswith('ready sample_rate=48000 hop_samples=480 '):
            raise RuntimeError('not ready: '+ready+' '+p.stderr.read().decode(errors='replace'))
        delay=int(re.search(r'model_delay_samples=(\d+)',ready).group(1))
        expected=hashlib.file_digest(Path(bundle).open('rb'),'sha256').hexdigest()
        assert 'model_sha256='+expected in ready and 'model_mode=custom' in ready
        hop=480
        frames=int(np.ceil((len(x)+delay)/hop))
        padded=np.pad(x,(0,frames*hop-len(x)))
        output=np.empty(frames*hop,dtype='float32')
        for i in range(frames):
            p.stdin.write(padded[i*hop:(i+1)*hop].astype('<f4').tobytes())
            output[i*hop:(i+1)*hop]=np.frombuffer(read_exact(p.stdout,hop*4),dtype='<f4')
        p.stdin.close()
        log=p.stderr.read().decode(errors='replace')
        assert p.wait(timeout=30)==0,log
        result=output[delay:delay+len(x)]
        assert len(result)==len(x) and np.isfinite(result).all()
        return result,ready
    finally:
        if p.poll() is None:p.kill();p.wait()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processor',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--validation',type=Path,required=True)
    ap.add_argument('--save-dir',type=Path,required=True)
    ap.add_argument('--system',default='no_oversampling')
    ap.add_argument('--count',type=int,default=10)
    ap.add_argument('--wsl',action='store_true',help='launch Linux build through WSL on Windows')
    a=ap.parse_args()
    if a.count <= 0:ap.error('--count must be positive')
    a.save_dir.mkdir(parents=True,exist_ok=True)
    reports=[]
    for idx in range(a.count):
        x,sr=sf.read(a.validation/f'{idx:04d}_noisy.wav',dtype='float32')
        ref,rr=sf.read(a.validation/a.system/f'{idx:04d}_output.wav',dtype='float64')
        assert sr==rr==48000 and x.ndim==ref.ndim==1 and len(x)==len(ref)
        y,ready=process(a.processor,a.bundle,x,wsl=a.wsl)
        diff=y-ref
        rms=float(np.sqrt(np.mean(diff*diff)))
        relative=rms/(float(np.sqrt(np.mean(ref*ref)))+1e-15)
        maxdiff=float(np.max(np.abs(diff)))
        # Preset gate: no data-fitted alignment/gain; no tolerance relaxation.
        passed=relative<=0.01 and maxdiff<=0.01
        sf.write(a.save_dir/f'{idx:04d}_rust.wav',y,48000,subtype='FLOAT')
        reports.append({'id':idx,'samples':len(x),'ready':ready,'rms_difference':rms,
            'relative_rms_difference':relative,'max_absolute_difference':maxdiff,
            'correlation':float(np.corrcoef(y,ref)[0,1]),'passed':passed})
        print(idx,'PASS' if passed else 'FAIL','relative_rms',relative,'maxdiff',maxdiff,flush=True)
    report={'workflow':'file-fed native Rust; not live or acoustic latency',
        'bundle_sha256':hashlib.file_digest(a.bundle.open('rb'),'sha256').hexdigest(),
        'processor_sha256':hashlib.file_digest(a.processor.open('rb'),'sha256').hexdigest(),
        'reference_scope':'ten saved validation cases, two rain recordings; not full defence coverage',
        'runtime':'parity mode; full processing; no postfilter/EQ/gain',
        'gate':{'relative_rms_difference_max':.01,'max_absolute_difference_max':.01},
        'passed':all(r['passed'] for r in reports),'cases':reports}
    (a.save_dir/'parity_report.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit('Parity gate failed: do not deploy')
    print('RAW WAVEFORM PARITY PASS',flush=True)

if __name__=='__main__':main()
