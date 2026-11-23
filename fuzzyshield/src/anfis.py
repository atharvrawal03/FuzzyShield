"""Compact adaptive Sugeno neuro-fuzzy binary classifier.

Each rule has a product of Gaussian membership functions as its antecedent,
normalized firing strengths, and a learned two-class zero-order Sugeno
consequent. Centers, widths, and consequents are optimized by minibatch
cross-entropy gradients. The estimator accepts shared preprocessed features.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.special import softmax

class AdaptiveSugenoANFIS(ClassifierMixin, BaseEstimator):
    def __init__(self, n_rules=6, epochs=18, learning_rate=0.025, batch_size=256, random_state=42, l2=1e-4):
        self.n_rules=n_rules; self.epochs=epochs; self.learning_rate=learning_rate; self.batch_size=batch_size; self.random_state=random_state; self.l2=l2

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float32)
        rng=np.random.default_rng(self.random_state); n,d=X.shape; r=self.n_rules
        # K-means initialization supplies data-driven fuzzy prototypes.
        from sklearn.cluster import MiniBatchKMeans
        km=MiniBatchKMeans(n_clusters=r, random_state=self.random_state, n_init=2, batch_size=max(512,r*32), max_iter=60).fit(X)
        self.centers_=km.cluster_centers_.astype(np.float32)
        self.log_widths_=np.full((r,d), np.log(max(float(np.sqrt(np.mean((X-self.centers_[km.labels_])**2))),0.25)), dtype=np.float32)
        self.consequents_=rng.normal(0,0.03,(r,2)).astype(np.float32)
        self.classes_=np.array([0,1],dtype=np.int8)
        eye=np.eye(2,dtype=np.float32)
        for epoch in range(self.epochs):
            order=rng.permutation(n); lr=self.learning_rate/(1+epoch*0.045)
            for start in range(0,n,self.batch_size):
                ids=order[start:start+self.batch_size]; xb=X[ids]; yb=eye[y[ids].astype(int)]
                diff=xb[:,None,:]-self.centers_[None,:,:]
                invvar=np.exp(-2*self.log_widths_)[None,:,:]
                loga=-0.5*np.mean(diff*diff*invvar,axis=2)
                weights=softmax(loga,axis=1).astype(np.float32)
                logits=weights@self.consequents_; probs=softmax(logits,axis=1).astype(np.float32)
                dlogits=(probs-yb)/len(ids)
                dc=weights.T@dlogits + self.l2*self.consequents_
                dw=dlogits@self.consequents_.T
                dloga=weights*(dw-np.sum(dw*weights,axis=1,keepdims=True))
                # Derivatives of product Gaussian rule firing wrt centers/log widths.
                dcenters=np.sum(dloga[:,:,None]*(diff*invvar/d),axis=0)
                dlogwidth=np.sum(dloga[:,:,None]*(diff*diff*invvar/d),axis=0)
                self.consequents_-=lr*np.clip(dc,-3,3)
                self.centers_-=lr*np.clip(dcenters,-2,2)
                self.log_widths_-=lr*np.clip(dlogwidth,-2,2)
                self.log_widths_=np.clip(self.log_widths_,-2.5,2.5)
        self.n_features_in_=d
        return self

    def _activations(self,X):
        diff=X[:,None,:]-self.centers_[None,:,:]
        return softmax(-0.5*np.mean(diff*diff*np.exp(-2*self.log_widths_)[None,:,:],axis=2),axis=1)
    def decision_function(self,X):
        check_is_fitted(self,"consequents_"); X=check_array(X,dtype=np.float32)
        return self._activations(X)@self.consequents_
    def predict_proba(self,X): return softmax(self.decision_function(X),axis=1)
    def predict(self,X): return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
