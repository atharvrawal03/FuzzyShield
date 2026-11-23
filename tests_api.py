"""API validation/inference smoke tests. Run: .venv/bin/python -m unittest tests_api"""
import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"fuzzyshield-vercel"/"backend"))
import main as index

NORMAL={"duration":0,"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":215,"dst_bytes":4500,"land":0,"wrong_fragment":0,"urgent":0,"hot":0,"num_failed_logins":0,"logged_in":1,"num_compromised":0,"root_shell":0,"su_attempted":0,"num_root":0,"num_file_creations":0,"num_shells":0,"num_access_files":0,"num_outbound_cmds":0,"is_host_login":0,"is_guest_login":0,"count":2,"srv_count":2,"serror_rate":0,"srv_serror_rate":0,"rerror_rate":0,"srv_rerror_rate":0,"same_srv_rate":1,"diff_srv_rate":0,"srv_diff_host_rate":0,"dst_host_count":10,"dst_host_srv_count":10,"dst_host_same_srv_rate":1,"dst_host_diff_srv_rate":0,"dst_host_same_src_port_rate":.1,"dst_host_srv_diff_host_rate":0,"dst_host_serror_rate":0,"dst_host_srv_serror_rate":0,"dst_host_rerror_rate":0,"dst_host_srv_rerror_rate":0}
class ApiTests(unittest.TestCase):
 def test_health(self): self.assertEqual(index.health()["status"],"ok")
 def test_rejects_bad_protocol(self):
  x=NORMAL.copy(); x['protocol_type']='bogus'
  with self.assertRaises(Exception): index.Traffic(**x)
 def test_rejects_extra_fields(self):
  x=NORMAL|{'surprise':1}
  with self.assertRaises(Exception): index.Traffic(**x)
 def test_unknown_model(self):
  with self.assertRaises(Exception): index.get_bundle('nope')
 def test_real_prediction_if_artifact_present(self):
  if not (ROOT/'fuzzyshield-vercel/backend/models/ann.joblib').exists(): self.skipTest('Train models first')
  r=index.predict('ann',index.Traffic(**NORMAL)); self.assertIn(r['label'],['Normal','Attack']); self.assertGreaterEqual(r['attack_probability'],0); self.assertLessEqual(r['attack_probability'],1)
 def test_anfis_prediction_if_artifact_present(self):
  if not (ROOT/'fuzzyshield-vercel/backend/models/anfis.joblib').exists(): self.skipTest('Train models first')
  r=index.predict('anfis',index.Traffic(**NORMAL)); self.assertEqual(r['model'],'ANFIS'); self.assertGreaterEqual(r['confidence'],0.5)
if __name__=='__main__': unittest.main(verbosity=2)
