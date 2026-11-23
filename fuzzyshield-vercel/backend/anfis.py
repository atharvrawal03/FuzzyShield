"""Pickle-compatible ANFIS class for the serverless inference function."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from scipy.special import softmax
class AdaptiveSugenoANFIS(ClassifierMixin, BaseEstimator):
    def __init__(self,n_rules=6,epochs=18,learning_rate=0.025,batch_size=256,random_state=42,l2=1e-4):
        self.n_rules=n_rules; self.epochs=epochs; self.learning_rate=learning_rate; self.batch_size=batch_size; self.random_state=random_state; self.l2=l2
    def _activations(self,X):
        diff=X[:,None,:]-self.centers_[None,:,:]
        return softmax(-0.5*np.mean(diff*diff*np.exp(-2*self.log_widths_)[None,:,:],axis=2),axis=1)
    def decision_function(self,X):
        if not hasattr(self,"consequents_"): raise ValueError("ANFIS model has not been fitted")
        X=check_array(X,dtype=np.float32); return self._activations(X)@self.consequents_
    def predict_proba(self,X): return softmax(self.decision_function(X),axis=1)
    def predict(self,X): return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
