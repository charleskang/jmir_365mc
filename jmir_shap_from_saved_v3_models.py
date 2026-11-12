from __future__ import annotations
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def get_args():
    p = argparse.ArgumentParser(description='OOF SHAP for existing JMIR v3 8-motion Combined models; no retraining')
    p.add_argument('--csv', required=True, help='Exact final patient1096_primary_surgeon CSV used in v3')
    p.add_argument('--core', default='classification_sklearn_jmir_r1_progress_no_velocity.py')
    p.add_argument('--root', default='R1_robustness_8_US', help='Existing US run root containing local_models, predictions, manifest')
    p.add_argument('--out', default=None, help='Separate SHAP output folder (default: root/OOF_SHAP_v3)')
    p.add_argument('--outcomes', default='surface,induration,bruising,seroma')
    p.add_argument('--repeat', type=int, default=0, help='Use 0 for prespecified first repeated CV split')
    p.add_argument('--verify-only', action='store_true', help='Verify source/model/prediction prerequisites only; no SHAP')
    p.add_argument('--force', action='store_true', help='Recompute existing fold SHAP outputs')
    return p.parse_args()


def preflight(root: Path, csv: Path, core_path: Path, args):
    manifest_p = root / 'manifest.json'
    conf_p = root / 'config_snapshot.json'
    sets_p = root / 'predictor_sets.json'
    for path in [manifest_p, conf_p, sets_p]:
        if not path.is_file():
            raise FileNotFoundError(f'Expected v3 file missing: {path}')
    manifest = json.loads(manifest_p.read_text(encoding='utf-8-sig'))
    conf = json.loads(conf_p.read_text(encoding='utf-8-sig'))
    sets = json.loads(sets_p.read_text(encoding='utf-8-sig'))
    if digest(csv) != manifest['final_csv_sha256']:
        raise ValueError('Dataset checksum differs from saved model source. Specify the EXACT original CSV.')
    if digest(core_path) != manifest['core_sha256']:
        raise ValueError('Core script checksum differs from fitted model source. Do not modify the original core .py.')
    if manifest.get('pilot', False):
        raise ValueError('Pilot model outputs cannot be used for paper figures.')
    if manifest.get('repeats') != 10 or manifest.get('folds') != 5 or manifest.get('n_estimators') != 400:
        raise ValueError(f'Unexpected CV/model manifest: {manifest}')
    if conf.get('families') != ['Patient', 'Combined'] or conf.get('strategies') != ['none']:
        raise ValueError('Unexpected base-model configuration; no assumption-based substitution.')
    if conf.get('site_col') != 'sugical_site' or conf.get('id_col') != 'PSENTRY':
        raise ValueError('Unexpected final column configuration.')
    expected_motion = ['totalStrokeCount', 'cohesion_average', 'cohesion_max',
                       'Cohesion_out', 'Separation_average', 'Separation_max',
                       'Separation_min', 'Separation_out']
    if sets['full'][8:] != expected_motion or len(sets['patient']) != 8 or len(sets['full']) != 16:
        raise ValueError('Unexpected input variables in saved v3 model; refuse to run.')
    labels = {'surface': 'Rad_2mon_w', 'induration': 'Rad_2mon_m',
              'bruising': 'Rad_4week', 'seroma': 'Rad_2week'}
    chosen = [v.strip() for v in args.outcomes.split(',') if v.strip()]
    if not chosen or any(x not in labels for x in chosen) or len(chosen) != len(set(chosen)):
        raise ValueError(f'Outcomes must be unique and in {list(labels)}')
    if args.repeat != 0:
        raise ValueError('Use --repeat 0 for prespecified one-time first-repeat OOF SHAP.')
    for name in chosen:
        o = labels[name]
        for fold in range(5):
            for f in [root/'local_models'/o/f'r00_f{fold:02d}.joblib',
                      root/'predictions'/o/'full'/f'r00_f{fold:02d}.csv']:
                if not f.is_file():
                    raise FileNotFoundError(f'Missing required saved model or prediction: {f}')
    return conf, sets, labels, chosen, manifest


def make_shap_beeswarm(frame: pd.DataFrame, features: list[str], destination: Path, title: str):
    """Actual SHAP beeswarm (numeric value color; categorical feature values left missing)."""
    import shap
    vals = frame.pivot(index='case', columns='feature', values='shap_value').reindex(columns=features)
    display = frame.pivot(index='case', columns='feature', values='display_value').reindex(index=vals.index, columns=features)
    if vals.isna().any().any() or len(vals) != 1096:
        raise ValueError('Incomplete OOF SHAP matrix: expected 1096 patients x 16 input predictors.')
    numeric_disp = display.apply(pd.to_numeric, errors='coerce')

    plt.close('all')
    shap.summary_plot(vals.to_numpy(dtype=float), numeric_disp.to_numpy(dtype=float),
                      feature_names=features, plot_type='dot', max_display=16,
                      show=False, plot_size=(11, 8))
    fig = plt.gcf()
    fig.suptitle(title + '\nFirst repetition, all 1,096 out-of-fold evaluations',
                 fontsize=11, y=1.02)
    fig.savefig(destination, dpi=240, bbox_inches='tight')
    fig.savefig(destination.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def composite_panels(out: Path, labels: dict, chosen: list[str]):

    from PIL import Image, ImageDraw, ImageFont
    order = [x for x in ['induration', 'surface', 'seroma', 'bruising'] if x in chosen]
    if len(order) == 4:
        files = [out/x/'shap_summary_colored.png' for x in order]
        if all(p.is_file() for p in files):
            images = [Image.open(p).convert('RGB') for p in files]
            width = 1400
            scaled = [im.resize((width, int(im.height * width/im.width))) for im in images]
            height = max(im.height for im in scaled)
            panel = Image.new('RGB', (width*2, height*2+65), 'white')
            d = ImageDraw.Draw(panel)
            for i, (label, im) in enumerate(zip(order, scaled)):
                x=(i%2)*width; y=(i//2)*(height+32)
                panel.paste(im,(x,y+32))
                d.text((x+24,y+6), f'({chr(97+i)}) {label.replace("_"," ").title()}', fill='black')
            panel.save(out/'Figure4_SHAP_4outcome_draft.png', optimize=True)

    req = [
        (x, f) for x in ['induration', 'surface'] for f in
        ['totalStrokeCount', 'Cohesion_out']
    ]
    if all(x in chosen for x in ['induration', 'surface']):
        files = [out/x/f'shap_dependence_{f}.png' for x,f in req]
        if all(p.is_file() for p in files):
            images = [Image.open(p).convert('RGB') for p in files]
            width = 1250
            scaled = [im.resize((width, int(im.height * width/im.width))) for im in images]
            height = max(im.height for im in scaled)
            panel = Image.new('RGB', (width*2, height*2+65), 'white')
            d = ImageDraw.Draw(panel)
            for i, ((outcome, feature), im) in enumerate(zip(req,scaled)):
                x=(i%2)*width; y=(i//2)*(height+32)
                panel.paste(im,(x,y+32))
                d.text((x+24,y+6), f'({chr(97+i)}) {outcome.title()} / {feature}',fill='black')
            panel.save(out/'Figure5_SHAP_dependence_4panel_draft.png', optimize=True)


def main():
    args = get_args()
    root, csv, core_path = (Path(x).expanduser().resolve() for x in (args.root, args.csv, args.core))
    out = Path(args.out).expanduser().resolve() if args.out else root/'OOF_SHAP_v3'
    conf, sets, labels, chosen, manifest = preflight(root, csv, core_path, args)
    print(f'PRECHECK PASS: verified exact data and fitted-core SHA256, 16 inputs, 20 models (if all outcomes).', flush=True)
    if args.verify_only:
        return

    import jmir_tracking_robustness_v3_no_velocity as robustness
    core = robustness.import_core(core_path)
    cfg = core.Config(**{k:v for k,v in conf.items() if k in core.Config.__dataclass_fields__})
    cfg.csv=str(csv)
    cfg.n_jobs=1  
    cfg.shap=False
    cfg.plots=False
    cfg.outcomes=[labels[x] for x in chosen]
    out.mkdir(parents=True, exist_ok=True)
    df, feature_sets, categorical = core.audit_and_load(cfg,out/'input_audit')
    if len(df) != 1096 or len(df['_case'].unique()) != 1096:
        raise ValueError('Unexpected patient sample count or repeated case identifiers.')
    if feature_sets['Combined'] != sets['full']:
        raise ValueError('Feature order differs between reconstructed dataset and stored model.')
    maps=robustness.get_split_map(core, df, cfg.outcomes, cfg)
    report=[]
    for name in chosen:
        outcome = labels[name]
        sub,y,splits = maps[outcome]
        parts=[]
        for rep,fold,tr,te in splits:
            if rep != 0: continue
            fold_dir=out/name/f'fold_{fold}'
            cache=fold_dir/'shap_values.csv.gz'
            Xtest=sub.iloc[te].copy()
            saved=pd.read_csv(root/'predictions'/outcome/'full'/f'r00_f{fold:02d}.csv')
            if len(saved)!=len(te) or set(saved.case.astype(str))!=set(Xtest._case.astype(str)):
                raise ValueError(f'OOF case mismatch for {name} fold {fold}')
            if not np.array_equal(saved.set_index('case').loc[Xtest._case.astype(str),'y_true'].to_numpy(),y[te]):
                raise ValueError(f'OOF label mismatch for {name} fold {fold}')
            print(f'{name}: fold {fold}/4; n={len(te)}; loading saved Combined model',flush=True)
            bundle = robustness.load_full_model(core,root,outcome,rep,fold)
            predicted=bundle.base.predict_proba(Xtest)
            reported=saved.set_index('case').loc[Xtest._case.astype(str),'p_raw'].to_numpy(dtype=float)
            err=float(np.max(np.abs(predicted-reported)))
            if err>1e-7: raise ValueError(f'Predicted scores differ from stored OOF: {name}/{fold} maxerr={err:g}')
            if cache.is_file() and not args.force:
                part=pd.read_csv(cache)
                if len(part)!=len(te)*16 or set(part.case.astype(str))!=set(Xtest._case.astype(str)):
                    raise ValueError(f'Stale or incomplete SHAP cache, use --force: {cache}')
                print('  restored previously checked fold SHAP', flush=True)
            else:
                part=core.compute_shap(bundle,Xtest,rep,fold,fold_dir)
                if part is None: raise RuntimeError('SHAP package unavailable; pip install shap')
            parts.append(part)
            report.append({'outcome':name,'fold':fold,'evaluation_n':len(te),'predictive_score_max_abs_diff':err,'features':16})
            del bundle
        if len(parts)!=5: raise ValueError(f'Five OOF folds not available for {name}')
        joined=pd.concat(parts,ignore_index=True)
        if joined['case'].nunique()!=1096 or len(joined)!=1096*16:
            raise ValueError(f'Incomplete SHAP output {name}: {joined.case.nunique()} cases, {len(joined)} rows')
        d=out/name;d.mkdir(parents=True,exist_ok=True)
        joined.to_csv(d/'all_oof_shap.csv.gz',index=False,compression='gzip',encoding='utf-8-sig')
        core.plot_shap_summary(joined,d,core.LABEL_MAP[outcome] + ' / Combined')
        make_shap_beeswarm(joined,sets['full'],d/'shap_summary_colored.png',core.LABEL_MAP[outcome])
        summary=(joined.assign(abs_shap=joined.shap_value.abs())
                    .groupby('feature',as_index=False).agg(mean_abs_shap=('abs_shap','mean'))
                    .sort_values('mean_abs_shap',ascending=False))
        summary.to_csv(d/'mean_abs_shap_ranked.csv',index=False)
        print(f'{name} DONE; top5: '+', '.join(summary.feature.head(5).tolist()),flush=True)
    pd.DataFrame(report).to_csv(out/'audit_checks.csv',index=False)
    (out/'provenance.json').write_text(json.dumps({
        'method':'uncalibrated RF probability, first-repeat out-of-fold SHAP, no retraining',
        'outcomes':chosen,'patient_n':1096,'total_features':16,'repeat':0,
        'source_csv_sha256':digest(csv),'core_sha256':digest(core_path),
        'saved_v3_manifest_sha256':digest(root/'manifest.json')},indent=2),encoding='utf-8')
    composite_panels(out,labels,chosen)
    print('ALL REQUESTED SHAP RESULTS COMPLETE:',out,flush=True)

if __name__=='__main__':
    main()
