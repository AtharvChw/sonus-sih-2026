"""Verify/extract v18 package and inspect paired validation gains and losses."""
import csv
import os
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('SONUS_WORKSPACE', Path(__file__).resolve().parents[1]))  # publish repo root by default; point at a full workspace to re-verify the private v18 archive
e=ROOT/'evidence/kaggle_v18'
archive=e/'download/ps26052-v17-export-eval.tar.gz'
expected=Path(str(archive)+'.sha256').read_text().split()[0]
digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
assert digest==expected,'package checksum mismatch'
out=e/'verified';out.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
    members=t.getmembers()
    assert all((out/m.name).resolve().is_relative_to(out.resolve()) and not m.issym() and not m.islnk() for m in members)
    t.extractall(out,filter='data')
cases=json.loads((out/'validation/cases.json').read_text())
assert len(cases)==480 and {c['input_snr_db'] for c in cases}=={-5,0,5,10,15}
rows={}
for name in ['bypass','pretrained','impulse_weighted','no_oversampling']:
    rows[name]=list(csv.DictReader((out/'validation'/name/'metrics.csv').open()))
    assert len(rows[name])==len(cases)
    for row,case in zip(rows[name],cases):
        assert int(row['id'])==case['id']
        for key in ['speech_sha256','noise_sha256','clean_sha256','noisy_sha256']:
            assert row[key]==case[key]
        assert all(np.isfinite(float(row[k])) for k in ['snr_db','si_snr_db','stoi','pesq_wb'])
for name in ['impulse_weighted','no_oversampling']:
    run=out/'runs'/name/'export'
    report=json.loads((run/'export_report.json').read_text())
    bundle=run/(name+'_onnx.tar.gz')
    assert report['ok'] and hashlib.file_digest(bundle.open('rb'),'sha256').hexdigest()==report['bundle_sha256']
    with tarfile.open(bundle) as t:
        assert sorted(t.getnames())==['config.ini','df_dec.onnx','enc.onnx','erb_dec.onnx']
metric_names=['snr_db','si_snr_db','stoi','pesq_wb']
baseline={k:np.array([float(r[k]) for r in rows['pretrained']]) for k in metric_names}
speakers=sorted({c['speaker'] for c in cases});noises=sorted({c['noise_sha256'] for c in cases})
sp=np.array([speakers.index(c['speaker']) for c in cases]);ns=np.array([noises.index(c['noise_sha256']) for c in cases])
rng=np.random.default_rng(26052)
weights=[]
for _ in range(2000):
    sw=np.bincount(rng.integers(0,len(speakers),len(speakers)),minlength=len(speakers))
    nw=np.bincount(rng.integers(0,len(noises),len(noises)),minlength=len(noises))
    w=sw[sp]*nw[ns];assert w.sum()>0;weights.append(w/w.sum())
weights=np.array(weights)
summary={'archive_sha256':digest,'pairs':len(cases),'speakers':len(speakers),'noise_recordings':len(noises),
    'scope':'validation only; not final test or live runtime',
    'ci_method':'exploratory two-way cluster bootstrap over speakers and noise recordings, 2000 replicates; only 3 speakers limits generalization',
    'systems':{},'paired_vs_pretrained':{}}
for name,rs in rows.items():
    summary['systems'][name]={k:float(np.mean([float(r[k]) for r in rs])) for k in metric_names}
for name in ['impulse_weighted','no_oversampling']:
    delta={k:np.array([float(r[k]) for r in rows[name]])-baseline[k] for k in metric_names}
    paired={k:{'mean_delta':float(d.mean()),'exploratory_95pct_ci':np.quantile(weights@d,[.025,.975]).tolist(),
               'improved_pairs':int((d>0).sum()),'regressed_pairs':int((d<0).sum())} for k,d in delta.items()}
    paired['material_loss_counts']={'stoi_drop_over_0.02':int((delta['stoi']<-.02).sum()),
        'pesq_drop_over_0.1':int((delta['pesq_wb']<-.1).sum()),'snr_drop_over_1db':int((delta['snr_db']<-1).sum())}
    groups=defaultdict(list)
    for i,c in enumerate(cases):groups[c['category']].append(i)
    paired['by_noise_category']={g:{k:float(d[ix].mean()) for k,d in delta.items()} for g,ix in groups.items()}
    paired['by_input_snr']={str(s):{k:float(d[[i for i,c in enumerate(cases) if c['input_snr_db']==s]].mean()) for k,d in delta.items()} for s in [-5,0,5,10,15]}
    summary['paired_vs_pretrained'][name]=paired
(e/'paired_review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# v18 paired validation review','','Verified 480 identical mixtures per system; package and export checksums match.','',
       '| System | Output SNR dB | SI-SNR dB | STOI | PESQ-WB |','|---|---:|---:|---:|---:|']
for name,v in summary['systems'].items():lines.append(f"| {name} | {v['snr_db']:.3f} | {v['si_snr_db']:.3f} | {v['stoi']:.4f} | {v['pesq_wb']:.4f} |")
lines+=['','## Paired losses and gains','']
for name,v in summary['paired_vs_pretrained'].items():
    lines += [f'### {name}','',f"Material losses versus pretrained: {v['material_loss_counts']}.",'']
    for k in metric_names:lines.append(f"- {k}: mean delta {v[k]['mean_delta']:.5f}; exploratory cluster CI {v[k]['exploratory_95pct_ci']}.")
lines+=['','These are validation means, not per-condition guarantees. Three speakers are insufficient for final generalization claims.','ONNX tensor parity passed; Rust waveform parity and Pi runtime remain unverified.','No deployment selection made by this script.']
(ROOT/'handoff/V18_VALIDATION_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
