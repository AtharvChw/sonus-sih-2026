"""Inference-only base/best parity probe; never updates checkpoint weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from df.enhance import init_df, enhance
from df.checkpoint import read_cp
from df.io import resample

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--probe', required=True)
    a = p.parse_args()
    run = Path(a.run)
    x, sr = sf.read(a.probe, dtype='float32', always_2d=True)
    assert x.shape[1] == 1 and np.isfinite(x).all()
    report = {'probe_sha256': hashlib.sha256(Path(a.probe).read_bytes()).hexdigest(), 'outputs': {}}
    for tag, epoch in [('base', 0), ('best', 'best')]:
        model, state, _ = init_df(str(run), epoch=None, config_allow_defaults=True)
        loaded = read_cp(model, 'model', str(run / 'checkpoints'), epoch=epoch)
        assert loaded is not None and (epoch != 0 or loaded == 0)
        audio = torch.from_numpy(x.T.copy())
        if sr != state.sr():
            audio = resample(audio, sr, state.sr())
        with torch.inference_mode():
            y = enhance(model, state, audio).detach().cpu().numpy().T
        assert np.isfinite(y).all() and len(y) > 0
        out = run / ('smoke_' + tag) / Path(a.probe).name
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, y, state.sr())
        report['outputs'][tag] = {'epoch': loaded, 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'samples': len(y)}
    (run / 'export_probe_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('BASE/BEST INFERENCE PASS', run.name)

if __name__ == '__main__':
    main()
