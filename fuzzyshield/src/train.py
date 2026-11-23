"""Reproducible NSL-KDD training and evaluation entry point."""
import json, time, sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import scipy, sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve)
from threadpoolctl import threadpool_limits
from data import ROOT, download_dataset, load_part, build_preprocessor
from anfis import AdaptiveSugenoANFIS

MODELS=ROOT/"models"; RESULTS=ROOT/"results"; PLOTS=RESULTS/"plots"

def evaluate(name, model, X, y):
    preds=[]; probabilities=[]; t=time.perf_counter()
    with threadpool_limits(limits=1):
        for start in range(0,len(y),2048):
            batch=X[start:start+2048]
            preds.append(model.predict(batch))
            probabilities.append(model.predict_proba(batch)[:,1])
    inference=(time.perf_counter()-t)/len(y)*1000
    pred=np.concatenate(preds); proba=np.concatenate(probabilities)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"model":name,"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"roc_auc":roc_auc_score(y,proba),"false_positive_rate":fp/(fp+tn),"training_seconds":None,"inference_ms_per_sample":inference,"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]},proba

def main():
    MODELS.mkdir(exist_ok=True); PLOTS.mkdir(parents=True,exist_ok=True)
    paths=download_dataset(); Xtr,ytr=load_part(paths["train"]); Xte,yte=load_part(paths["test"])
    if set(ytr.unique())!={0,1} or set(yte.unique())!={0,1}:
        raise ValueError("Both Normal and Attack labels must be present in each official split")
    prep=build_preprocessor(); A=prep.fit_transform(Xtr).astype(np.float32); B=prep.transform(Xte).astype(np.float32)
    # Tune architecture and regularization with stratified two-fold CV on official training rows.
    split=next(StratifiedShuffleSplit(n_splits=1,test_size=min(4000,max(2500,len(ytr)//30)),random_state=42).split(A,ytr))
    tune_x,tune_y=A[split[0]],ytr.iloc[split[0]]
    search=GridSearchCV(MLPClassifier(max_iter=40,early_stopping=True,n_iter_no_change=5,random_state=42),{"hidden_layer_sizes":[(32,),(64,32)],"alpha":[0.0001,0.01]},cv=2,scoring="f1",n_jobs=1,refit=True,verbose=1)
    t=time.perf_counter()
    with threadpool_limits(limits=1): search.fit(tune_x,tune_y)
    ann_train=time.perf_counter()-t
    ann=search.best_estimator_
    # Retrain selected model on the full official training set; report total ANN fitting time.
    t=time.perf_counter()
    with threadpool_limits(limits=1): ann.fit(A,ytr)
    ann_train+=time.perf_counter()-t
    joblib.dump({"model":ann,"preprocessor":prep,"features":__import__('data').FEATURES},MODELS/"ann.joblib",compress=3)
    # ANFIS shares the identical transformed feature space and official test set.
    # Cap fit rows to keep neuro-fuzzy optimization practical on a laptop.
    rng=np.random.default_rng(42); sample=rng.choice(len(A),size=min(len(A),24000),replace=False)
    anfis=AdaptiveSugenoANFIS(n_rules=6,epochs=18,batch_size=256,random_state=42)
    t=time.perf_counter()
    with threadpool_limits(limits=1): anfis.fit(A[sample],ytr.iloc[sample].to_numpy())
    anfis_train=time.perf_counter()-t
    joblib.dump({"model":anfis,"preprocessor":prep,"features":__import__('data').FEATURES},MODELS/"anfis.joblib",compress=3)
    # Keep deployable copies with the Next.js/Vercel application and publish measured metrics.
    deploy_root=ROOT.parent/"fuzzyshield-vercel"
    backend_root=deploy_root/"backend"
    (backend_root/"models").mkdir(parents=True,exist_ok=True)
    shutil.copy2(MODELS/"ann.joblib",backend_root/"models"/"ann.joblib")
    shutil.copy2(MODELS/"anfis.joblib",backend_root/"models"/"anfis.joblib")
    ann_m,ann_p=evaluate("ANN",ann,B,yte.to_numpy()); anf_m,anf_p=evaluate("ANFIS",anfis,B,yte.to_numpy())
    ann_m["training_seconds"]=ann_train; anf_m["training_seconds"]=anfis_train
    metrics={"dataset":"NSL-KDD KDDTrain+ / KDDTest+ official split","train_samples":len(ytr),"test_samples":len(yte),"features":len(__import__('data').FEATURES),"preprocessed_features":int(A.shape[1]),"ann_best_params":search.best_params_,"anfis_rules":anfis.n_rules,"anfis_training_samples":len(sample),"results":[ann_m,anf_m]}
    (RESULTS/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (deploy_root/"public").mkdir(exist_ok=True)
    shutil.copy2(RESULTS/"metrics.json",deploy_root/"public"/"evaluation.json")
    inference_requirements="\n".join([
        "fastapi==0.115.12", "pydantic==2.11.7",
        f"numpy=={np.__version__}", f"scipy=={scipy.__version__}",
        f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
        "joblib==1.5.1", "",
    ])
    (backend_root/"requirements.txt").write_text(inference_requirements)
    (deploy_root/"requirements.txt").write_text(inference_requirements)
    fig,axs=plt.subplots(1,3,figsize=(14,4));
    for ax,entry in zip(axs[:2],metrics["results"]): ConfusionMatrixDisplay(np.array(entry["confusion_matrix"]),display_labels=["Normal","Attack"]).plot(ax=ax,colorbar=False,cmap="Blues"); ax.set_title(entry["model"]+" confusion")
    for name,prob in [("ANN",ann_p),("ANFIS",anf_p)]:
        fpr,tpr,_=roc_curve(yte,prob); axs[2].plot(fpr,tpr,label=f"{name} AUC={roc_auc_score(yte,prob):.3f}")
    axs[2].plot([0,1],[0,1],"--",color="gray"); axs[2].set(xlabel="False positive rate",ylabel="True positive rate",title="ROC curves"); axs[2].legend(); fig.tight_layout(); fig.savefig(PLOTS/"model_evaluation.png",dpi=180); plt.close(fig)
    shutil.copy2(PLOTS/"model_evaluation.png",deploy_root/"public"/"model_evaluation.png")
    pd.DataFrame(metrics["results"]).drop(columns="confusion_matrix").to_csv(RESULTS/"metrics.csv",index=False)
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
