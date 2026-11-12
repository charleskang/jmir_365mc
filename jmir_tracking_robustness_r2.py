

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

LOG = logging.getLogger("jmir_robustness")
OUTCOMES = {
    "seroma": "Rad_2week", "bruising": "Rad_4week",
    "surface": "Rad_2mon_w", "induration": "Rad_2mon_m",
}
COUNTS = ["totalStrokeCount", "Cohesion_out", "Separation_out"]
COHESION = ["cohesion_average", "cohesion_max", "Cohesion_out"]
SEPARATION = ["Separation_average", "Separation_max", "Separation_min", "Separation_out"]
GEOMETRY = ["cohesion_average", "cohesion_max", "Separation_average", "Separation_max", "Separation_min"]
RATIOS = ["Cohesion_out_per_count", "Separation_out_per_count"]
VARIANTS = [
    "patient", "full", "no_outlier", "no_velocity", "clean",
    "no_cohesion", "no_separation", "geometry_only", "stroke_only",
    "ratio", "clip01", "clip05",
]

DEFAULT_LEVELS = [0., .05, .10, .20]
DEFAULT_TIPPING = [0., .05, .10, .20, .30, .40, .50, .75, 1.0]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def resolve_core(path: str | None) -> Path:
    if path:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"재분석 원본 코드 없음: {p}")
        return p
    roots = [Path.cwd(), Path(__file__).resolve().parent]
    for root in roots:
        for name in ["classification_sklearn_jmir_r1.py", "classification_sklearn_jmir_r1(1).py"]:
            p = root/name
            if p.exists():
                return p.resolve()
    raise FileNotFoundError("--core 경로에 classification_sklearn_jmir_r1.py를 지정하세요.")


def import_core(path: Path):

    name = "jmir_robustness_primary_core"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for attr in ["Config", "audit_and_load", "outer_splits", "fit_bundle", "seed_for", "make_derived_frame"]:
        if not hasattr(mod, attr):
            raise RuntimeError(f"기존 재분석 코드에 {attr} 함수가 없습니다. --core 버전 확인 필요")
    return mod


class Clock:
    def __init__(self, jobs: int):
        self.start = time.monotonic()
        self.jobs = max(1,jobs)
        self.finished = 0
    def step(self, msg: str, cached: bool = False):
        self.finished += 1
        elapsed = time.monotonic() - self.start
        left = elapsed / self.finished * (self.jobs-self.finished)
        LOG.info("[%d/%d, %.1f%%] elapsed %.1fm, ETA %.1fm | %s%s",
                 self.finished,self.jobs,100*self.finished/self.jobs,elapsed/60,left/60,
                 "cached " if cached else "",msg)


def read_levels(raw: str) -> list[float]:
    nums = sorted(set(float(x.strip()) for x in raw.split(",") if x.strip()))
    if not nums or nums[0]<0 or nums[-1]>2.0:
        raise ValueError("오차 SD 입력은 0 이상 2 이하의 소수 (0.05 = 5%)")
    return nums


def configure(args, core, csv: Path, pilot: bool):
    cfg = core.Config(
        csv=str(csv), id_col=args.id_col,
        site_col=args.site_col, branch_col=args.branch_col,
        surgeon_col=args.surgeon_col, schemes=["cv"],
        families=["Patient", "Combined"], strategies=["none"],
        outcomes=[OUTCOMES[x] for x in args.outcomes.split(",")],
        folds=(3 if pilot else args.folds), repeats=(1 if pilot else args.repeats),
        inner_folds=(2 if pilot else args.inner_folds),
        seed=args.seed, n_estimators=(8 if pilot else args.n_estimators),
        n_jobs=args.jobs, tune=(not args.no_tune),
        shap=False, plots=False, n_bootstrap=args.bootstrap,
        bootstrap_unit="patient", confirm_reviewed_features=False)
    return cfg


def get_split_map(core, df: pd.DataFrame, outcomes: list[str], cfg):
    maps={}
    for out in outcomes:
        sub=df[df[out].notna()].copy().reset_index(drop=True)
        y=sub[out].astype(int).to_numpy()
        spl=core.outer_splits(sub,y,"cv",cfg)
        maps[out]=(sub,y,spl)
    return maps


def get_sets(features: dict, df: pd.DataFrame):
    p=features["Patient"]
    m=features["Motion"]
    if m!=["totalStrokeCount", "totalStrokeVelocity", "cohesion_average", "cohesion_max",
           "Cohesion_out", "Separation_average", "Separation_max", "Separation_min", "Separation_out"]:
        raise ValueError("Motion feature 목록이 예상과 다릅니다. 스크립트/실험을 점검하세요.")
    if len(p)!=8:
        LOG.warning("Patient 입력 %d개; 재분석 Table의 8개와 다릅니다. --site-col 확인",len(p))
    sets={
        "patient":p,
        "full":p+m,
        "no_outlier":p+[x for x in m if x not in ["Cohesion_out","Separation_out"]],
        "no_velocity":p+[x for x in m if x!="totalStrokeVelocity"],
        "clean":p+[x for x in m if x not in ["Cohesion_out","Separation_out","totalStrokeVelocity"]],
        "no_cohesion":p+[x for x in m if x not in COHESION],
        "no_separation":p+[x for x in m if x not in SEPARATION],
        "geometry_only":p+GEOMETRY,
        "stroke_only":p+["totalStrokeCount", "totalStrokeVelocity"],
        "ratio":p+[x for x in m if x not in ["Cohesion_out", "Separation_out"]]+RATIOS,
        "clip01":p+m,
        "clip05":p+m,
    }
    missing=[c for c in ["cohesion_count","Separation_count"] if c not in df]
    if missing: raise ValueError(f"ratio variant에 필요한 분모 열 없음: {missing}")
    for count,denom in [("Cohesion_out","cohesion_count"),("Separation_out","Separation_count")]:
        num=pd.to_numeric(df[count],errors="raise").to_numpy(float)
        dn=pd.to_numeric(df[denom],errors="raise").to_numpy(float)
        if not np.isfinite(num).all() or not np.isfinite(dn).all():
            raise ValueError(f"{count}/{denom}: NaN/Inf.")
        if np.any(dn<=0) or np.any(num<0) or np.any(num>dn):
            raise ValueError(f"{count}/{denom}: 비율 정의 위반. 실제 count 해석 확인 필요")
        df[RATIOS[["Cohesion_out","Separation_out"].index(count)]]=num/dn
    return sets


def clip_inputs(train: pd.DataFrame, test: pd.DataFrame, fields: list[str], tail: float):
    tr=train.copy(); te=test.copy(); quant={}
    for c in fields:
        if c not in tr or c not in te: continue
        vec=pd.to_numeric(tr[c],errors="raise").to_numpy(float)
        finite=vec[np.isfinite(vec)]
        if len(finite)<5: continue
        lo,hi=np.quantile(finite,[tail,1-tail])
        tr[c]=tr[c].clip(lower=lo,upper=hi)
        te[c]=te[c].clip(lower=lo,upper=hi)
        quant[c]=[float(lo),float(hi)]
    return tr,te,quant


def split_inputs(train,test,variant,features,motion):
    if variant.startswith("clip"):
        tail=.01 if variant=="clip01" else .05

        tr,te,meta=clip_inputs(train,test,motion,tail)
        return tr,te,{"train_fold_quantiles":meta,"clip_tail":tail}
    return train.copy(),test.copy(),{}


def split_path(root:Path,out:str,variant:str,rep:int,fold:int):
    return root/"predictions"/out/variant/f"r{rep:02d}_f{fold:02d}.csv"


def model_path(root,out,rep,fold):
    return root/"local_models"/out/f"r{rep:02d}_f{fold:02d}.joblib"


def ablation(core, cfg, df, sets, maps, root, variants):
    jobs=sum(len(v[2]) for v in maps.values())*len(variants)
    clock=Clock(jobs)
    for out,(sub,y,splits) in maps.items():
        for variant in variants:
            sel=sets[variant]
            cats=[c for c in ["sex",cfg.site_col] if c in sel]
            for rep,fold,tr,te in splits:
                dest=split_path(root,out,variant,rep,fold)
                mpath=model_path(root,out,rep,fold)
                if dest.exists() and (variant!="full" or mpath.exists()):
                    clock.step(f"{out}/{variant}/{rep}/{fold}",True); continue
                trdf,tedf,extra=split_inputs(sub.iloc[tr],sub.iloc[te],variant,sel,
                                               [x for x in sets["full"] if x not in sets["patient"]])
                seed=core.seed_for(cfg.seed,"cv",out,"full",rep,fold)
                bundle=core.fit_bundle(trdf,y[tr],None,sel,cats,"none",cfg,seed)
                raw,cal=bundle.predict(tedf)
                write_csv(dest,{
                    "case":tedf["_case"].to_numpy(), "repeat":rep,"fold":fold,
                    "y_true":y[te],"p_raw":raw,"p_calibrated":cal,
                    "branch":tedf["_branch"].to_numpy(),
                    "surgeon":tedf["_surgeon"].to_numpy(),
                    "train_n":len(tr),"eval_n":len(te),"outcome":out,
                    "variant":variant,
                })
                if variant=="full":
                    mpath.parent.mkdir(parents=True,exist_ok=True)

                    joblib.dump({"bundle":bundle,"core_sha256":digest(core.__file__ and Path(core.__file__)),
                                 "not_a_deployment_model":True},mpath,compress=1)
                if extra:
                    write_json(root/"clipping_thresholds"/out/variant/f"r{rep:02d}_f{fold:02d}.json",extra)
                clock.step(f"{out}/{variant}/{rep}/{fold}")
    LOG.info("Ablation CV fits completed. Now use --stage summarize for paired bootstrap.")


def load_variant(root,out,variant):
    fs=sorted((root/"predictions"/out/variant).glob("r*_f*.csv"))
    if not fs: raise FileNotFoundError(f"No CV predictions: {out}/{variant}; 먼저 --stage ablation")
    frame=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
    if frame.duplicated(["case","repeat"]).any(): raise ValueError(f"Duplicate OOF: {out}/{variant}")
    return frame


def metrics_array(y:np.ndarray, p:np.ndarray):

    if len(np.unique(y))!=2:return np.array([np.nan,np.nan,np.nan])
    return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)])


def aligned_matrices(root,out,variants):
    reference=load_variant(root,out,"patient").sort_values(["repeat","case"])
    counts=reference.groupby("repeat").size()
    if counts.nunique()!=1: raise ValueError("Repeated OOF coverage not constant")
    reps=sorted(reference.repeat.unique()); n=int(counts.iloc[0])
    cases=reference[reference.repeat==reps[0]].case.astype(str).tolist()
    y=reference[reference.repeat==reps[0]].y_true.to_numpy(int)
    all_p={}
    for v in variants:
        a=load_variant(root,out,v).sort_values(["repeat","case"])
        if len(a)!=len(reference) or not a[["case","repeat","y_true"]].reset_index(drop=True).equals(
                reference[["case","repeat","y_true"]].reset_index(drop=True)):
            raise ValueError(f"Patient/repeat alignment failed: {v}")
        if any(a.groupby("repeat").size().to_numpy()!=n):
            raise ValueError("OOF incomplete")
        all_p[v]=np.stack([a[a.repeat==r].p_raw.to_numpy(float) for r in reps])
    if len(set(cases))!=n or not np.isin(y,[0,1]).all(): raise ValueError("Repeated patient coverage corrupted")
    return reps,cases,y,all_p


def bootstrap_summary(core, cfg, root, variants, outcomes, nboot):
    results=[]; clock=Clock(len(outcomes)*(len(variants)-1))
    for out in outcomes:
        reps,cases,y,mat=aligned_matrices(root,out,variants)
        seed=core.seed_for(cfg.seed,"robustness_bootstrap",out)
        samples=core.sample_indices(len(y),None,nboot,seed)
        ref=mat["patient"]
        ref_point=np.nanmean([metrics_array(y,r) for r in ref],axis=0)
        rows=[]
        for var in variants:
            p=mat[var]
            mm=np.mean([metrics_array(y,r) for r in p],axis=0)
            rows.append((var,mm))
        for var,m in rows:
            if var=="patient":
                for j,key in enumerate(["auroc","average_precision","brier"]):
                    results.append({"outcome":out,"variant":var,"metric":key,"estimate":m[j],
                                    "patient_baseline":ref_point[j],"delta_vs_patient":0.,
                                    "ci_low":np.nan,"ci_high":np.nan,"bootstrap_n":nboot})
                continue
            boot=np.empty((nboot,3));boot[:]=np.nan
            for bi,idx in enumerate(samples):
                yy=y[idx]
                if len(np.unique(yy))!=2: continue
                deltas=[metrics_array(yy,pp[idx])-metrics_array(yy,rr[idx]) for pp,rr in zip(mat[var],ref)]
                boot[bi]=np.mean(deltas,axis=0)
            point=m-ref_point
            for j,key in enumerate(["auroc","average_precision","brier"]):
                lo,hi,valid=core.interval(boot[:,j],.95)
                results.append({"outcome":out,"variant":var,"metric":key,"estimate":m[j],
                                "patient_baseline":ref_point[j],"delta_vs_patient":point[j],
                                "ci_low":lo,"ci_high":hi,"valid_bootstraps":valid,"bootstrap_n":nboot})
            write_csv(root/"summaries"/f"partial_{out}.csv",[x for x in results if x["outcome"]==out])
            clock.step(f"{out}/{var} bootstrap {nboot}")
    write_csv(root/"summaries"/"paired_ablation_bootstrap.csv",results)
    LOG.info("Patient-level paired intervals computed for all specified models.")


def input_perturb(X:pd.DataFrame, fields:list[str], sd:float, seed:int, mode:str):
    if sd==0:
        return X.copy()
    rng=np.random.default_rng(seed)
    result=X.copy(); n=len(result)

    common=np.exp(rng.normal(-.5*sd*sd,sd,size=n))
    for col in fields:
        vals=pd.to_numeric(X[col],errors="raise").to_numpy(float)
        if mode=="correlated" and col in (COHESION+SEPARATION):
            fac=common
        else:
            fac=np.exp(rng.normal(-.5*sd*sd,sd,size=n))
        if mode=="count_thinning":
            if col in COUNTS:

                new=rng.binomial(np.maximum(0,np.rint(vals)).astype(int),1-sd)
            else:
                new=vals
        else:
            new=vals*fac
            if col in COUNTS:
                new=np.maximum(0,np.rint(new)).astype(float)
        result[col]=new
    return result


def load_full_model(core,root,out,rep,fold):
    p=model_path(root,out,rep,fold)
    if not p.exists(): raise FileNotFoundError(f"Missing local fitted model: {p}. Run --stage ablation including full first.")
    item=joblib.load(p)  
    if item["core_sha256"]!=digest(Path(core.__file__)):
        raise RuntimeError("Original core source code changed since models were saved")
    return item["bundle"]


def perturb(core,cfg,maps,sets,root,levels,draws,mode,stage_name):
    if mode=="count_thinning" and max(levels)>=1:
        LOG.warning("100%% count-thinning is a hypothetical extreme stress test")
    motion=[c for c in sets["full"] if c not in sets["patient"]]
    jobs=sum(len(v[2]) for v in maps.values())*len(levels)*draws
    clock=Clock(jobs)
    for out,(df,y,splits) in maps.items():

        for rep,fold,tr,te in splits:
            candidates=[(sd,draw,root/stage_name/mode/out/f"sd_{sd:g}"/
                  f"draw_{draw:03d}_r{rep:02d}_f{fold:02d}.csv")
                  for sd in levels for draw in range(draws)]
            if all(dest.exists() for _,_,dest in candidates):
                for sd,draw,_ in candidates:
                    clock.step(f"{out} {mode} {sd:g} draw {draw} r{rep}f{fold}",True)
                continue
            bundle=load_full_model(core,root,out,rep,fold)
            for sd,draw,dest in candidates:
                if dest.exists():
                    clock.step(f"{out} {mode} {sd:g} draw {draw} r{rep}f{fold}",True)
                    continue
                pred_input=input_perturb(df.iloc[te],motion,sd,
                    core.seed_for(cfg.seed,"hypothetical_error",stage_name,mode,out,sd,draw,rep,fold),mode)
                raw,cal=bundle.predict(pred_input)
                write_csv(dest,{"case":df.iloc[te]["_case"].to_numpy(),"y_true":y[te],
                    "repeat":rep,"fold":fold,"draw":draw,"assumed_sd":sd,"mode":mode,
                    "p_raw":raw,"p_calibrated":cal,
                    "not_observed_sensor_error":True})
                clock.step(f"{out} {mode} {sd:g} draw {draw} r{rep}f{fold}")
    summarize_perturb(root,stage_name,mode,maps,levels,draws)

def summarize_perturb(root,stage,mode,maps,levels,draws):
    rows=[]
    for out,(df,y,splits) in maps.items():

        reps,cases,yy,preds=aligned_matrices(root,out,["patient","full"])
        baseline=np.mean([metrics_array(yy,a) for a in preds["patient"]],axis=0)
        clean=np.mean([metrics_array(yy,a) for a in preds["full"]],axis=0)
        for sd in levels:
            draws_metrics=[]
            for draw in range(draws):
                files=sorted((root/stage/mode/out/f"sd_{sd:g}").glob(f"draw_{draw:03d}_r*_f*.csv"))
                if len(files)!=len(splits): raise RuntimeError(f"Perturb outputs incomplete: {out} sd={sd} draw={draw}")
                f=pd.concat([pd.read_csv(x) for x in files],ignore_index=True)
                if f.duplicated(["repeat","case"]).any(): raise ValueError("Repeated evaluation patient duplicate")
                r_metrics=[]
                for rep in reps:
                    a=f[f.repeat==rep].sort_values("case")
                    if a.case.astype(str).tolist()!=cases or not np.array_equal(a.y_true.to_numpy(int),yy):
                        raise ValueError("Perturb predictions not aligned with original patients")
                    r_metrics.append(metrics_array(yy,a.p_raw.to_numpy(float)))
                draws_metrics.append(np.mean(r_metrics,axis=0))
            dm=np.array(draws_metrics)
            for i,metric in enumerate(["auroc","average_precision","brier"]):
                rows.append({"outcome":out,"mode":mode,"assumed_error_sd":sd,
                     "metric":metric,"mean_over_draws":float(dm[:,i].mean()),
                     "sd_between_simulation_draws":float(dm[:,i].std(ddof=1)) if draws>1 else np.nan,
                     "draw_min":float(dm[:,i].min()),"draw_max":float(dm[:,i].max()),
                     "patient_model":baseline[i],"original_combined":clean[i],
                     "delta_vs_patient":float(dm[:,i].mean()-baseline[i]),
                     "n_simulations":draws,
                     "interpretation":"simulation_based_not_observed_tracking_quality"})

        for metric in ["auroc","average_precision"]:
            tr=[r for r in rows if r["outcome"]==out and r["metric"]==metric]
            cross=[r["assumed_error_sd"] for r in tr if r["delta_vs_patient"]<=0]
            write_json(root/stage/mode/out/f"tipping_{metric}.json",
                       {"outcome":out,"metric":metric,"first_grid_sd_mean_delta_nonpositive":min(cross) if cross else None,
                        "searched_grid":levels,"not_a_physical_sensor_error_tolerance":True})
    write_csv(root/stage/mode/"scenario_summary.csv",rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for out in maps:
            fig,axs=plt.subplots(1,2,figsize=(11,4))
            for ax,metric in zip(axs,["auroc","average_precision"]):
                f=pd.DataFrame([r for r in rows if r["outcome"]==out and r["metric"]==metric]).sort_values("assumed_error_sd")
                ax.plot(f.assumed_error_sd*100,f.delta_vs_patient,marker='o')
                ax.axhline(0,ls='--');ax.set_xlabel("Assumed error SD (%)")
                ax.set_ylabel(f"Combined minus Patient {metric}")
                ax.set_title("Simulation only; not observed camera error")
            fig.suptitle(out);fig.tight_layout()
            fig.savefig(root/stage/mode/f"tipping_{out}.png",dpi=200)
            plt.close(fig)
    except ImportError:
        pass
    LOG.info("Simulation summaries saved")


def conditional_permute(core,cfg,maps,sets,root,draws):
    cols=[c for c in sets["full"] if c not in sets["patient"]]
    jobs=sum(len(v[2]) for v in maps.values())*draws
    clk=Clock(jobs)
    for out,(sub,y,splits) in maps.items():
        for rep,fold,tr,te in splits:
            files=[root/"conditional"/out/f"draw_{draw:03d}_r{rep:02d}_f{fold:02d}.csv" for draw in range(draws)]
            if all(p.exists() for p in files):
                for draw in range(draws): clk.step(f"{out}/{draw}/{rep}/{fold}",True)
                continue
            train,eval_df=sub.iloc[tr],sub.iloc[te]
            ids=[]
            for c in ["operation_time","fat_ext_volume"]:
                edges=np.unique(np.quantile(pd.to_numeric(train[c],errors="raise"),[.0,.33,.66,1.0]))
                if len(edges)<3: b=np.zeros(len(eval_df),dtype=int)
                else: b=np.searchsorted(edges[1:-1],pd.to_numeric(eval_df[c],errors="raise"),side="right")
                ids.append(b)
            site=eval_df[cfg.site_col].map(core.category_text).astype(str).to_numpy()
            strata=np.array([f"{s}|{a}|{b}" for s,a,b in zip(site,*ids)])
            bun=load_full_model(core,root,out,rep,fold)
            for draw,dest in enumerate(files):
                if dest.exists():clk.step(f"{out}/{draw}/{rep}/{fold}",True);continue
                test=eval_df.copy()
                rng=np.random.default_rng(core.seed_for(cfg.seed,"conditional",out,draw,rep,fold))
                permutation=np.arange(len(test))
                for key in np.unique(strata):
                    ix=np.flatnonzero(strata==key)
                    if len(ix)>1:permutation[ix]=rng.permutation(ix)
                moved=int(np.count_nonzero(permutation!=np.arange(len(test))))
                test.loc[:,cols]=eval_df.iloc[permutation][cols].to_numpy()
                pr,_=bun.predict(test)
                write_csv(dest,{"repeat":rep,"fold":fold,"case":eval_df["_case"].to_numpy(),
                                "y_true":y[te],"p_raw":pr,"draw":draw,"permuted_n":moved,
                                "n_evaluation":len(test),"not_causal_inference":True})
                clk.step(f"{out} cond-draw{draw} r{rep}f{fold} moved={moved}/{len(test)}")
    rows=[]
    for out,(sub,y,splits) in maps.items():
        reps,cases,yy,preds=aligned_matrices(root,out,["patient","full"])
        orig=np.mean([metrics_array(yy,p) for p in preds["full"]],axis=0)
        for draw in range(draws):
            files=sorted((root/"conditional"/out).glob(f"draw_{draw:03d}_r*_f*.csv"))
            f=pd.concat([pd.read_csv(x) for x in files],ignore_index=True)
            scores=[]
            for rep in reps:
                a=f[f.repeat==rep].sort_values("case")
                if a.case.astype(str).tolist()!=cases: raise ValueError("Conditional fold alignment mismatch")
                scores.append(metrics_array(yy,a.p_raw.to_numpy(float)))
            means=np.mean(scores,axis=0)
            for j,met in enumerate(["auroc","average_precision","brier"]):
                rows.append({"outcome":out,"draw":draw,"metric":met,"original_full":orig[j],
                             "conditional_permuted":means[j],"delta_original_minus_permuted":orig[j]-means[j],
                             "note":"shuffle_within_site_and_train_defined_size_duration_bins_not_causal"})
    write_csv(root/"conditional"/"summary.csv",rows)

def subgroup(core,cfg,maps,root):
    rows=[]
    for out,(sub,y,splits) in maps.items():
        comps={v:load_variant(root,out,v) for v in ["patient","full"]}
        raw=sub.set_index("_case")
        for name,field in [("treated_site",cfg.site_col),("branch","_branch"),("surgeon","_surgeon")]:
            if field not in raw:continue
            labs=raw[field].map(core.category_text).astype(str)
            for group,identifiers in labs.groupby(labs):
                keys=set(identifiers.index)
                for rep in range(cfg.repeats):
                    gg={v:c[(c.repeat==rep)&c.case.isin(keys)].sort_values("case") for v,c in comps.items()}
                    if len(gg["full"])!=len(gg["patient"]):raise ValueError("Mismatched subgroup cases")
                    a=gg["full"];b=gg["patient"];pos=int(a.y_true.sum())
                    if not a.case.astype(str).tolist()==b.case.astype(str).tolist():raise ValueError("Mismatch subgroup")
                    if len(a)<20 or min(pos,len(a)-pos)<5:
                        for metric in ["auroc","average_precision","brier"]:
                            rows.append({"outcome":out,"group_type":name,"group_value":group,"repeat":rep,
                                "n":len(a),"events":pos,"metric":metric,"delta_full_minus_patient":np.nan,"status":"suppressed_n_lt20_or_few_events"})
                        continue
                    m1=metrics_array(a.y_true.to_numpy(int),a.p_raw.to_numpy(float))
                    m0=metrics_array(b.y_true.to_numpy(int),b.p_raw.to_numpy(float))
                    for k,met in enumerate(["auroc","average_precision","brier"]):
                        rows.append({"outcome":out,"group_type":name,"group_value":group,"repeat":rep,
                                    "n":len(a),"events":pos,"metric":met,"delta_full_minus_patient":m1[k]-m0[k],"status":"descriptive"})

        for name,c in [("operative_duration","operation_time"),("extracted_fat_volume","fat_ext_volume")]:
            vals=pd.to_numeric(raw[c],errors="raise")
            cuts=np.unique(np.quantile(vals,[0,.5,1]))
            grp=pd.Series(np.searchsorted(cuts[1:-1],vals,side='right'),index=raw.index)
            for cat in np.unique(grp):
                cases=set(grp.index[grp==cat])
                for rep in range(cfg.repeats):
                    a=comps["full"][(comps["full"].repeat==rep)&(comps["full"].case.isin(cases))].sort_values('case')
                    b=comps["patient"][(comps["patient"].repeat==rep)&(comps["patient"].case.isin(cases))].sort_values('case')
                    pos=int(a.y_true.sum());status="descriptive" if len(a)>=20 and min(pos,len(a)-pos)>=5 else "suppressed_n_lt20_or_few_events"
                    mm=metrics_array(a.y_true.to_numpy(int),a.p_raw.to_numpy(float))-metrics_array(b.y_true.to_numpy(int),b.p_raw.to_numpy(float)) if status=="descriptive" else [np.nan]*3
                    for k,met in enumerate(["auroc","average_precision","brier"]):
                        rows.append({"outcome":out,"group_type":name,"group_value":f"median_bin_{cat}","repeat":rep,
                                     "n":len(a),"events":pos,"metric":met,"delta_full_minus_patient":mm[k],"status":status})
        if "ope_date" in raw:
            dt=pd.to_datetime(raw.ope_date.astype(str).str.replace(r"\.0$","",regex=True),format="%Y%m%d",errors="coerce")
            if dt.notna().all():
                midpoint=dt.median()
                for label, keys in [("earlier",set(dt.index[dt<=midpoint])),("later",set(dt.index[dt>midpoint]))]:
                    for rep in range(cfg.repeats):
                        a=comps["full"][(comps["full"].repeat==rep)&(comps["full"].case.isin(keys))].sort_values('case')
                        b=comps["patient"][(comps["patient"].repeat==rep)&(comps["patient"].case.isin(keys))].sort_values('case')
                        pos=int(a.y_true.sum());status="descriptive" if len(a)>=20 and min(pos,len(a)-pos)>=5 else "suppressed_n_lt20_or_few_events"
                        mm=metrics_array(a.y_true.to_numpy(int),a.p_raw.to_numpy(float))-metrics_array(b.y_true.to_numpy(int),b.p_raw.to_numpy(float)) if status=="descriptive" else [np.nan]*3
                        for k,met in enumerate(["auroc","average_precision","brier"]):
                            rows.append({"outcome":out,"group_type":"operation_date_half","group_value":label,"repeat":rep,
                                         "n":len(a),"events":pos,"metric":met,"delta_full_minus_patient":mm[k],"status":status})
    write_csv(root/"subgroup"/"repeat_level_differences.csv",rows)
    agg=(pd.DataFrame(rows).groupby(["outcome","group_type","group_value","metric","status"],dropna=False)
         .agg(n=('n','max'),events=('events','max'),mean_delta=('delta_full_minus_patient','mean'),repeats=('repeat','nunique')).reset_index())
    write_csv(root/"subgroup"/"summary.csv",agg)


def negative_control(core,cfg,maps,sets,root):

    res=[]
    for out,(sub,y,splits) in maps.items():
        rep,fold,tr,te=splits[0]
        rng=np.random.default_rng(core.seed_for(cfg.seed,"negative_control",out))
        shuffled=rng.permutation(y[tr]); seed=core.seed_for(cfg.seed,"negative_fit",out)
        cfg_local=core.Config(**asdict(cfg)); cfg_local.tune=False
        bun=core.fit_bundle(sub.iloc[tr],shuffled,None,sets["full"],
                            [v for v in ["sex",cfg.site_col] if v in sets["full"]],"none",cfg_local,seed)
        p,_=bun.predict(sub.iloc[te]); metrics=metrics_array(y[te],p)
        for j,name in enumerate(["auroc","average_precision","brier"]):
            res.append({"outcome":out,"metric":name,"value":metrics[j],""})
    write_csv(root/"negative"/"shuffled_label_onefold.csv",res)


def snapshot_integrity(args,cfg,core,source:Path,root:Path):

    manifest=root/"manifest.json"
    obj={"final_csv_path":str(source),"final_csv_sha256":digest(source),
         "core_path":str(Path(core.__file__)),"core_sha256":digest(Path(core.__file__)),
         "script_sha256":digest(Path(__file__)),
         "seed":cfg.seed,"repeats":cfg.repeats,"folds":cfg.folds,
         "n_estimators":cfg.n_estimators,"inner_folds":cfg.inner_folds,
         "tune":cfg.tune,"site_col":cfg.site_col,"id_col":cfg.id_col,
         "outcomes":cfg.outcomes,"pilot":args.pilot,
         "analysis_scope":"feature_predictive_robustness_only_not_tracking_accuracy"}
    if manifest.exists():
        older=json.loads(manifest.read_text(encoding="utf-8"))
        for k in ["final_csv_sha256","core_sha256","script_sha256","seed","repeats","folds","n_estimators","inner_folds","tune","site_col","id_col","outcomes","pilot"]:
            if older.get(k)!=obj.get(k):
                raise RuntimeError(f"기존 --outdir의 manifest와 새 실행 설정이 다릅니다: {k}. 새 폴더를 사용하세요.")
    else:
        write_json(manifest,obj)

        shutil.copy2(Path(__file__),root/"script_snapshot.py")
        shutil.copy2(Path(core.__file__),root/"primary_core_snapshot.py")
        write_json(root/"config_snapshot.json",asdict(cfg))


def close_logging():
    """Release Windows file locks before temporary-directory cleanup."""
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()


@contextmanager
def logging_lifetime():
    try:
        yield
    finally:
        close_logging()


def init_logging(root:Path):
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    close_logging()  
    h1=logging.StreamHandler(sys.stdout)
    h2=logging.FileHandler(root/"robustness.log",encoding="utf-8")
    for h in (h1,h2):
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(h)


def run(args):
    if args.self_test:

        with tempfile.TemporaryDirectory(prefix="jmir_robustness_selftest_") as tmp, logging_lifetime():
            mod=import_core(resolve_core(args.core))
            syn=mod.synthetic_data(90,1234)

            syn["cohesion_count"]=np.maximum(syn["cohesion_count"],syn["Cohesion_out"]+1)
            syn["Separation_count"]=np.maximum(syn["Separation_count"],syn["Separation_out"]+1)
            syn.to_csv(Path(tmp)/"synthetic.csv",index=False)
            args.csv=str(Path(tmp)/"synthetic.csv"); args.outdir=str(Path(tmp)/"run")
            args.id_col="PSENTRY";args.site_col="surgical_site"
            args.branch_col="branch_id";args.surgeon_col="surgeon_id"
            args.outcomes="surface";args.pilot=True;args.no_tune=True
            args.stage="ablation,summarize,perturb,conditional,subgroup,negative"
            args.variants="patient,full,no_outlier,clean,ratio,clip01"
            args.levels="0,0.1";args.draws=1;args.bootstrap=25
            args.jobs=1
            _run(args)
            assert (Path(tmp)/"run"/"summaries"/"paired_ablation_bootstrap.csv").exists()
            assert (Path(tmp)/"run"/"perturb"/"independent"/"scenario_summary.csv").exists()
            assert (Path(tmp)/"run"/"conditional"/"summary.csv").exists()
            assert (Path(tmp)/"run"/"subgroup"/"summary.csv").exists()
            assert (Path(tmp)/"run"/"negative"/"shuffled_label_onefold.csv").exists()

            pr=pd.read_csv(Path(tmp)/"run"/"predictions"/"Rad_2mon_w"/"full"/"r00_f00.csv")
            pp=pd.read_csv(Path(tmp)/"run"/"perturb"/"independent"/"Rad_2mon_w"/
                           "sd_0"/"draw_000_r00_f00.csv")
            assert np.allclose(pr.p_raw.to_numpy(),pp.p_raw.to_numpy(),atol=1e-12)

            assert (Path(tmp)/"run"/"manifest.json").exists()
        print("SELF-TEST PASS")
        return
    with logging_lifetime():
        _run(args)


def _run(args):
    if not args.csv: raise ValueError("--csv에 최종 1환자 1행 CSV를 지정하세요.")
    source=Path(args.csv).expanduser().resolve()
    if not source.exists(): raise FileNotFoundError(source)
    root=Path(args.outdir).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    init_logging(root)
    if args.pilot: LOG.warning("PILOT: output is NOT suitable for paper")
    core=import_core(resolve_core(args.core))
    cfg=configure(args,core,source,args.pilot)
    snapshot_integrity(args,cfg,core,source,root)

    audit_root=root/"input_audit";audit_root.mkdir(exist_ok=True)
    df,features,categorical=core.audit_and_load(cfg,audit_root)
    if len(df)!=len(df._case.unique()): raise ValueError("환자 중복이 존재합니다")
    if not args.pilot and len(df)!=1096:
        raise ValueError(f"최종 분석 대상은 1096명입니다. 현재 {len(df)}행. --pilot만 예외 허용")
    if args.site_col not in df:raise ValueError("Treated site column missing")
    if any(df[o].isna().any() for o in cfg.outcomes):
        raise ValueError("Outcome missing: 전체 환자 OOF coverage가 깨질 수 있습니다")
    sets=get_sets(features,df)
    write_json(root/"predictor_sets.json",sets)
    maps=get_split_map(core,df,cfg.outcomes,cfg)
    for out,(d,y,spls) in maps.items():
        for rep,fold,tr,te in spls:
            if set(d.iloc[tr]._case)&set(d.iloc[te]._case):raise AssertionError("patient leakage")
    st=[s.strip() for s in args.stage.split(",")]
    if "all" in st:st=["ablation","summarize","perturb","tipping","conditional","subgroup","negative"]
    variants=[v.strip() for v in args.variants.split(",") if v.strip()]
    if any(v not in VARIANTS for v in variants):raise ValueError(f"Unknown variant; valid={VARIANTS}")
    if "patient" not in variants or "full" not in variants:
        raise ValueError("--variants에 patient,full은 반드시 포함해야 합니다")
    for stage in st:
        LOG.info("STAGE: %s",stage)
        if stage=="ablation":ablation(core,cfg,df,sets,maps,root,variants)
        elif stage=="summarize":bootstrap_summary(core,cfg,root,variants,cfg.outcomes,args.bootstrap)
        elif stage in {"perturb","tipping"}:
            levels=read_levels(args.levels if stage=="perturb" else args.tipping_levels)
            for mode in [x.strip() for x in args.modes.split(",")]:
                if mode not in ["independent","correlated","count_thinning"]:
                    raise ValueError("--modes must use independent,correlated,count_thinning")
                perturb(core,cfg,maps,sets,root,levels,args.draws,mode,stage)
        elif stage=="conditional":conditional_permute(core,cfg,maps,sets,root,args.draws)
        elif stage=="subgroup":subgroup(core,cfg,maps,root)
        elif stage=="negative":negative_control(core,cfg,maps,sets,root)
        else:raise ValueError(f"Unrecognized stage: {stage}")
    LOG.info("Completed stages %s; output folder %s",st,root)



if __name__=="__main__":
    try:
        run(parse_args())
    except Exception as ex:
        LOG.exception("Robustness experiment failed: %s",ex)
        sys.exit(1)
