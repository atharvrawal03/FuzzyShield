from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from data import FEATURES
st.set_page_config(page_title="FuzzyShield | IDS",page_icon="🛡️",layout="wide")
st.markdown("<style>.stApp{background:radial-gradient(circle at 10% 0%,#18234a 0,#0b1020 43%,#080b14 100%);color:#ecf2ff}.block-container{padding-top:2rem}.metric-card{padding:18px;border:1px solid #ffffff20;border-radius:18px;background:#ffffff0a}</style>",unsafe_allow_html=True)
st.title("🛡️ FuzzyShield")
st.caption("Adaptive intrusion detection · NSL-KDD benchmark")
model_files={name:ROOT/"models"/f"{name}.joblib" for name in ["ann","anfis"]}
@st.cache_resource
def load_model(path): return joblib.load(path)
metrics_path=ROOT/"results/metrics.json"
metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else None
with st.sidebar:
 st.header("Traffic input")
 preset=st.radio("Preset",["Normal","Attack"])
 def_val={"duration":0,"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":215,"dst_bytes":4500,"land":0,"wrong_fragment":0,"urgent":0,"hot":0,"num_failed_logins":0,"logged_in":1,"num_compromised":0,"root_shell":0,"su_attempted":0,"num_root":0,"num_file_creations":0,"num_shells":0,"num_access_files":0,"num_outbound_cmds":0,"is_host_login":0,"is_guest_login":0,"count":2,"srv_count":2,"serror_rate":0,"srv_serror_rate":0,"rerror_rate":0,"srv_rerror_rate":0,"same_srv_rate":1,"diff_srv_rate":0,"srv_diff_host_rate":0,"dst_host_count":10,"dst_host_srv_count":10,"dst_host_same_srv_rate":1,"dst_host_diff_srv_rate":0,"dst_host_same_src_port_rate":0.1,"dst_host_srv_diff_host_rate":0,"dst_host_serror_rate":0,"dst_host_srv_serror_rate":0,"dst_host_rerror_rate":0,"dst_host_srv_rerror_rate":0}
 if preset=="Attack": def_val.update(protocol_type="tcp",service="private",flag="S0",src_bytes=0,dst_bytes=0,logged_in=0,count=511,srv_count=2,serror_rate=1,srv_serror_rate=1,same_srv_rate=0.01,diff_srv_rate=0.99,dst_host_count=255,dst_host_srv_count=1,dst_host_serror_rate=1,dst_host_srv_serror_rate=1)
 with st.expander("Edit selected features",expanded=True):
  service=st.text_input("Service",def_val["service"]); protocol=st.selectbox("Protocol",["tcp","udp","icmp"],index=["tcp","udp","icmp"].index(def_val["protocol_type"])); flag=st.selectbox("Connection flag",["SF","S0","REJ","RSTR","SH"],index=["SF","S0","REJ","RSTR","SH"].index(def_val["flag"]))
  src_bytes=st.number_input("Source bytes",min_value=0,value=int(def_val["src_bytes"])); dst_bytes=st.number_input("Destination bytes",min_value=0,value=int(def_val["dst_bytes"])); count=st.number_input("Connection count",min_value=0,value=int(def_val["count"]))
  logged=st.selectbox("Successful login",[0,1],index=int(def_val["logged_in"]))
 traffic=def_val.copy(); traffic.update(service=service,protocol_type=protocol,flag=flag,src_bytes=src_bytes,dst_bytes=dst_bytes,count=count,logged_in=logged)
 run=st.button("Analyze traffic",type="primary",use_container_width=True)
if run:
 values=pd.DataFrame([{f:traffic[f] for f in FEATURES}])
 cols=st.columns(2)
 for col,name in zip(cols,["ann","anfis"]):
  if model_files[name].exists():
   bundle=load_model(str(model_files[name])); x=bundle["preprocessor"].transform(values); prediction=int(bundle["model"].predict(x)[0]); proba=float(bundle["model"].predict_proba(x)[0,1]); col.metric(name.upper(),"ATTACK" if prediction else "NORMAL",f"Attack likelihood {proba:.1%}")
  else: col.warning("Model not trained yet. Run `python -m src.train`.")
if metrics:
 st.divider(); st.subheader("Benchmark results")
 cards=st.columns(4); cards[0].metric("Official test records",f"{metrics['test_samples']:,}"); cards[1].metric("Train records",f"{metrics['train_samples']:,}"); cards[2].metric("Feature count",metrics['features']); cards[3].metric("Comparison", "ANN vs ANFIS")
 df=pd.DataFrame(metrics["results"]).set_index("model"); display=df[["accuracy","precision","recall","f1","roc_auc","false_positive_rate","training_seconds","inference_ms_per_sample"]].style.format("{:.4f}").to_html(); st.markdown(display,unsafe_allow_html=True)
 plot=ROOT/"results/plots/model_evaluation.png"
 if plot.exists(): st.image(str(plot),use_container_width=True)
else: st.info("Train the models with the official NSL-KDD split to populate actual benchmark metrics.")
