"""FastAPI prediction endpoints; compatible with Vercel Python functions."""
from pathlib import Path
import os
from typing import Literal
import joblib
import pandas as pd
from anfis import AdaptiveSugenoANFIS  # registers the class module before unpickling
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE=Path(__file__).resolve().parent
ROOT=HERE
class Traffic(BaseModel):
    class Config:
        extra="forbid"
        allow_inf_nan=False
    duration: float=Field(ge=0); protocol_type: Literal["tcp","udp","icmp"]
    service: str=Field(min_length=1,max_length=40); flag: str=Field(min_length=1,max_length=10)
    src_bytes: float=Field(ge=0); dst_bytes: float=Field(ge=0)
    land: int=Field(ge=0,le=1); wrong_fragment: float=Field(ge=0); urgent: float=Field(ge=0); hot: float=Field(ge=0)
    num_failed_logins: float=Field(ge=0); logged_in: int=Field(ge=0,le=1); num_compromised: float=Field(ge=0); root_shell: int=Field(ge=0,le=1); su_attempted: float=Field(ge=0); num_root: float=Field(ge=0)
    num_file_creations: float=Field(ge=0); num_shells: float=Field(ge=0); num_access_files: float=Field(ge=0); num_outbound_cmds: float=Field(ge=0); is_host_login: int=Field(ge=0,le=1); is_guest_login: int=Field(ge=0,le=1)
    count: float=Field(ge=0); srv_count: float=Field(ge=0)
    serror_rate: float=Field(ge=0,le=1); srv_serror_rate: float=Field(ge=0,le=1); rerror_rate: float=Field(ge=0,le=1); srv_rerror_rate: float=Field(ge=0,le=1); same_srv_rate: float=Field(ge=0,le=1); diff_srv_rate: float=Field(ge=0,le=1); srv_diff_host_rate: float=Field(ge=0,le=1)
    dst_host_count: float=Field(ge=0); dst_host_srv_count: float=Field(ge=0)
    dst_host_same_srv_rate: float=Field(ge=0,le=1); dst_host_diff_srv_rate: float=Field(ge=0,le=1); dst_host_same_src_port_rate: float=Field(ge=0,le=1); dst_host_srv_diff_host_rate: float=Field(ge=0,le=1); dst_host_serror_rate: float=Field(ge=0,le=1); dst_host_srv_serror_rate: float=Field(ge=0,le=1); dst_host_rerror_rate: float=Field(ge=0,le=1); dst_host_srv_rerror_rate: float=Field(ge=0,le=1)
app=FastAPI(title="FuzzyShield IDS API",version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if x.strip()], allow_credentials=False, allow_methods=["GET","POST"], allow_headers=["Content-Type"])
_cache={}
def get_bundle(name):
    if name not in ("ann","anfis"): raise HTTPException(404,"Model must be ann or anfis")
    if name not in _cache:
        path=ROOT/"models"/f"{name}.joblib"
        if not path.exists(): raise HTTPException(503,f"{name.upper()} model is not deployed")
        try: _cache[name]=joblib.load(path)
        except Exception as exc: raise HTTPException(500,f"Could not load {name} model: {exc}")
    return _cache[name]
@app.get("/api/health")
def health(): return {"status":"ok","models":{n:(ROOT/"models"/f"{n}.joblib").exists() for n in ("ann","anfis")}}
@app.post("/api/predict/{model_name}")
def predict(model_name: str, traffic: Traffic):
    bundle=get_bundle(model_name); payload=traffic.model_dump() if hasattr(traffic,"model_dump") else traffic.dict(); frame=pd.DataFrame([payload])
    try:
        transformed=bundle["preprocessor"].transform(frame)
        model=bundle["model"]; probability=float(model.predict_proba(transformed)[0,1]); pred=int(model.predict(transformed)[0])
        return {"model":model_name.upper(),"label":"Attack" if pred else "Normal","prediction":pred,"attack_probability":probability,"confidence":max(probability,1-probability)}
    except Exception as exc: raise HTTPException(422,f"Prediction failed: {exc}")
