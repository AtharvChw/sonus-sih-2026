"""Export and score validation-only candidates. No optimizer or training calls.

Metrics use delay-compensated raw model outputs: no EQ, gain or postfilter.
Validation estimates are developmental; they are not sealed final-test claims.
"""
import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_model(run, epoch):
    from df.enhance import init_df
    from df.checkpoint import read_cp
    model, state, _ = init_df(str(run), epoch=None, post_filter=False, config_allow_defaults=True)
    actual = read_cp(model, 'model', str(run / 'checkpoints'), epoch=epoch)
    assert actual is not None and (epoch != 0 or actual == 0)
    model.eval()
    return model, state, actual


def export_model(run):
    import onnx
    import onnxruntime as ort
    from df.scripts.export import export
    model, state, epoch = load_model(run, 'best')
    torch.manual_seed(26052)
    out = run / 'export'
    out.mkdir(exist_ok=True)
    export(model, str(out), state, check=True, simplify=False, opset=14)
    checks = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    for part in ['enc', 'erb_dec', 'df_dec']:
        onnx.checker.check_model(str(out / (part + '.onnx')), full_check=True)
        inputs = np.load(out / (part + '_input.npz'))
        expected = np.load(out / (part + '_output.npz'))
        session = ort.InferenceSession(str(out / (part + '.onnx')), sess_options=options, providers=['CPUExecutionProvider'])
        outputs = session.run(None, {x.name: inputs[x.name] for x in session.get_inputs()})
        checks[part] = {}
        for node, actual in zip(session.get_outputs(), outputs):
            ref = expected[node.name]
            assert actual.shape == ref.shape and np.isfinite(actual).all()
            # Upstream logs mismatches as warnings. This gate raises on mismatch.
            np.testing.assert_allclose(actual, ref, rtol=1e-4, atol=1e-5)
            checks[part][node.name] = float(np.max(np.abs(actual - ref)))
    shutil.copy2(run / 'config.ini', out / 'config.ini')
    bundle = out / (run.name + '_onnx.tar.gz')
    names = ['enc.onnx', 'erb_dec.onnx', 'df_dec.onnx', 'config.ini']
    with tarfile.open(bundle, 'w:gz') as t:
        for name in names:
            t.add(out / name, arcname=name)
    with tarfile.open(bundle) as t:
        assert sorted(t.getnames()) == sorted(names)
    report = {'ok': True, 'epoch': epoch, 'tensor_parity': checks, 'rtol': 1e-4, 'atol': 1e-5,
              'bundle_sha256': sha(bundle), 'bundle_bytes': bundle.stat().st_size,
              'waveform_runtime_parity_verified': False}
    (out / 'export_report.json').write_text(json.dumps(report, indent=2))
    print('STRICT ONNX TENSOR PARITY PASS', run.name, flush=True)


def read_audio(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True)
    assert np.isfinite(x).all()
    x = x.mean(axis=1)
    if sr != 48000:
        d = gcd(sr, 48000)
        x = resample_poly(x, 48000 // d, sr // d)
    assert len(x) >= 2400 and np.mean(x*x) > 1e-12
    return x


def make_cases(root, out):
    manifest = list(csv.DictReader((root / 'lists/split_manifest.csv').open()))
    # Validation speakers and noise recordings only; deterministic balanced subset.
    groups = defaultdict(list)
    for row in manifest:
        if row['split'] != 'valid':
            continue
        if row['kind'] == 'speech':
            key = ('speech', row['source_id'])
        elif row['kind'] == 'sesa_noise':
            key = ('noise', Path(row['absolute_path']).stem.rsplit('_', 1)[0])
        else:
            key = ('noise', 'esc_category_' + Path(row['absolute_path']).stem.rsplit('-', 1)[-1])
        groups[key].append(row)
    speech, noise = [], []
    for key, rows in sorted(groups.items()):
        # Sort by pre-existing content hashes, not model scores.
        selected_count = 0
        for row in sorted(rows, key=lambda r: r['sha256']):
            p = Path(row['absolute_path'])
            if not p.exists():
                matches = list(Path('/kaggle/input').rglob(p.name))
                assert len(matches) == 1, ('ambiguous/missing input', p.name)
                p = matches[0]
            assert sha(p) == row['sha256'], ('validation source drift', p)
            if key[0] == 'speech' and len(read_audio(p)) < 2 * 48000:
                continue
            item = (key[1], p, row['sha256'])
            (speech if key[0] == 'speech' else noise).append(item)
            selected_count += 1
            if selected_count == 2:
                break
        assert selected_count >= (2 if key[0] == 'speech' else 1), ('insufficient validation files', key)
    assert len(speech) == 6 and len(noise) >= 6
    cases = []
    rng = np.random.default_rng(26052)
    out.mkdir(exist_ok=True)
    for speaker, sp, sh in speech:
        clean = read_audio(sp)[:48000*5]
        clean = clean / np.sqrt(np.mean(clean*clean)) * 0.08
        for category, npth, nh in noise:
            ns = read_audio(npth)
            ns = np.tile(ns, int(np.ceil(len(clean)/len(ns))) + 1)
            start = int(rng.integers(0, len(ns) - len(clean) + 1))
            ns = ns[start:start+len(clean)]
            assert np.mean(ns*ns) > 1e-12
            ns = ns / np.sqrt(np.mean(ns*ns))
            for snr in [-5, 0, 5, 10, 15]:
                n = ns * np.sqrt(np.mean(clean*clean)) / 10**(snr/20)
                scale = min(1., 0.95 / max(np.max(np.abs(clean+n)), np.max(np.abs(clean))))
                s, mix = clean*scale, (clean+n)*scale
                actual = 10*np.log10(np.sum(s*s)/np.sum((mix-s)**2))
                assert abs(actual-snr) < 1e-8
                idx = len(cases)
                sf.write(out / f'{idx:04d}_clean.wav', s, 48000, subtype='FLOAT')
                sf.write(out / f'{idx:04d}_noisy.wav', mix, 48000, subtype='FLOAT')
                cases.append({'id':idx,'speaker':speaker,'category':category,'input_snr_db':snr,
                              'speech_sha256':sh,'noise_sha256':nh,'samples':len(s),
                              'clean_sha256':sha(out/f'{idx:04d}_clean.wav'),
                              'noisy_sha256':sha(out/f'{idx:04d}_noisy.wav')})
    (out / 'cases.json').write_text(json.dumps(cases, indent=2))
    print('VALIDATION MIXTURES VERIFIED', len(cases), flush=True)


def metrics(s, y):
    from pystoi import stoi
    from pesq import pesq
    assert y.shape == s.shape and np.isfinite(y).all()
    snr = 10*np.log10(np.sum(s*s)/(np.sum((y-s)**2)+1e-15))
    sc, yc = s-s.mean(), y-y.mean()
    target = sc*np.dot(yc,sc)/(np.dot(sc,sc)+1e-15)
    sisnr = 10*np.log10((np.dot(target,target)+1e-15)/(np.sum((yc-target)**2)+1e-15))
    s16, y16 = resample_poly(s,1,3), resample_poly(y,1,3)
    return {'snr_db':float(snr), 'si_snr_db':float(sisnr),
            'stoi':float(stoi(s16,y16,16000,extended=False)),
            'pesq_wb':float(pesq(16000,s16,y16,'wb'))}


def score(root, name):
    from df.enhance import enhance
    out = root / 'validation'
    cases = json.loads((out/'cases.json').read_text())
    if name != 'bypass':
        run = root / 'runs' / ('impulse_weighted' if name=='pretrained' else name)
        model,state,epoch = load_model(run, 0 if name=='pretrained' else 'best')
    else:
        epoch = None
    destination = out/name
    destination.mkdir(exist_ok=True)
    rows = []
    for case in cases:
        idx = case['id']
        s,_ = sf.read(out/f'{idx:04d}_clean.wav')
        x,_ = sf.read(out/f'{idx:04d}_noisy.wav')
        if name=='bypass':
            y=x
        else:
            with torch.inference_mode():
                y=enhance(model,state,torch.from_numpy(x.astype('float32')).unsqueeze(0),pad=True).cpu().numpy().reshape(-1)
        row = dict(case, system=name, loaded_epoch=epoch, **metrics(s,y))
        row['clipped_samples'] = int(np.sum(np.abs(y)>=1))
        row['output_rms_dbfs'] = float(10*np.log10(np.mean(y*y)+1e-15))
        assert all(np.isfinite(row[k]) for k in ['snr_db','si_snr_db','stoi','pesq_wb'])
        rows.append(row)
        if idx < 10:
            sf.write(destination/f'{idx:04d}_output.wav',y,48000,subtype='FLOAT')
        if (idx+1)%25==0:
            print(name, idx+1, '/', len(cases), flush=True)
    with (destination/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (destination/'report.json').write_text(json.dumps({'ok':True,'system':name,'pairs':len(rows),'epoch':epoch,
        'means':{k:float(np.mean([r[k] for r in rows])) for k in ['snr_db','si_snr_db','stoi','pesq_wb']},
        'selection_scope':'validation only; final acceptance unmeasured'},indent=2))
    print('VALIDATION SCORING PASS',name,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--phase',choices=['export','mix','score'],required=True)
    p.add_argument('--system',choices=['bypass','pretrained','impulse_weighted','no_oversampling'])
    a=p.parse_args(); root=Path(a.root)
    torch.set_num_threads(2)
    if a.phase=='export':
        assert a.system in ['impulse_weighted','no_oversampling']
        export_model(root/'runs'/a.system)
    elif a.phase=='mix':
        make_cases(root,root/'validation')
    else:
        assert a.system
        score(root,a.system)

if __name__=='__main__':
    main()
