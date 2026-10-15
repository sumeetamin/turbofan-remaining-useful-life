"""Train and evaluate a turbofan RUL baseline using engine-level holdout."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
SENSORS=[f"s{i}" for i in range(1,22)]
def load(path):
    names=["unit","cycle","op1","op2","op3",*SENSORS]
    df=pd.read_csv(path,sep=r"\s+",header=None,names=names,engine="python").dropna(axis=1,how="all")
    if "unit" not in df or "cycle" not in df: raise ValueError("Expected C-MAPSS whitespace-delimited run-to-failure file.")
    return df
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--train",required=True); p.add_argument("--out",default="outputs"); p.add_argument("--cap-rul",type=int,default=125); a=p.parse_args()
    df=load(a.train); max_cycle=df.groupby("unit").cycle.transform("max"); df["rul"]=(max_cycle-df.cycle).clip(upper=a.cap_rul)
    sensors=[c for c in SENSORS if c in df]
    for c in sensors:
        df[f"{c}_mean10"]=df.groupby("unit")[c].transform(lambda s:s.rolling(10,min_periods=1).mean())
        df[f"{c}_std10"]=df.groupby("unit")[c].transform(lambda s:s.rolling(10,min_periods=2).std()).fillna(0)
    feats=["cycle","op1","op2","op3",*sensors,*[f"{c}_{s}10" for c in sensors for s in ("mean","std")]]
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42); tr,te=next(split.split(df[feats],df.rul,groups=df.unit))
    model=RandomForestRegressor(n_estimators=250,min_samples_leaf=2,max_features=.8,n_jobs=-1,random_state=42)
    model.fit(df.iloc[tr][feats],df.iloc[tr].rul); pred=model.predict(df.iloc[te][feats])
    metrics={"engines":int(df.unit.nunique()),"train_engines":int(df.iloc[tr].unit.nunique()),"test_engines":int(df.iloc[te].unit.nunique()),
       "row_holdout_mae":float(mean_absolute_error(df.iloc[te].rul,pred)),"row_holdout_rmse":float(mean_squared_error(df.iloc[te].rul,pred)**.5),
       "note":"Rows within each held-out engine are correlated; report engine-level results before operational use."}
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    pd.DataFrame({"unit":df.iloc[te].unit,"cycle":df.iloc[te].cycle,"actual_rul":df.iloc[te].rul,"predicted_rul":pred}).to_csv(out/"holdout_predictions.csv",index=False)
    joblib.dump({"model":model,"features":feats,"cap_rul":a.cap_rul},out/"model.joblib"); print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
