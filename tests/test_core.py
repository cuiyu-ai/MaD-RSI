import copy
import math
import tempfile
import unittest
from pathlib import Path

from mad_rsi import (EvidenceConfig, build_bundle, choose_probe, compare_harnesses,
                     model_coverage, response_tensor, run_round, select_queries, validate_digest)
from mad_rsi.acceptance import paired_group_decision
from mad_rsi.serialization import read_json, write_json

MODELS = [{"id": "probe-a", "family": "group-a"}, {"id": "probe-b", "family": "group-b"}]


def records(task_ids=("q0", "q1", "q2"), temperatures=(0.0, 1.0), harness="version-0"):
    rows = []
    for q in task_ids:
        for m in MODELS:
            for temp in temperatures:
                for trial in range(2):
                    success = q == "q0" or (q == "q2" and m["id"] == "probe-a")
                    rows.append({"task_id": q, "model_id": m["id"], "family": m["family"],
                                 "temperature": temp, "trial": trial, "split": "evolve",
                                 "status": "ok", "success": success, "reward": float(success),
                                 "tokens": 10, "harness_hash": harness, "input_hash": "input-" + q,
                                 "protocol_hash": "protocol", "evidence_text": "pass" if success else "fail",
                                 "evidence_id": f"{q}/{m['id']}/{temp}/{trial}"})
    return rows


def diagnosis(bundle):
    refs = [s["evidence_id"] for e in bundle["evidence"] for s in e["sources"]]
    return {"shared_failures": [{"claim": "Fixture observation", "evidence_ids": refs[:1]}],
            "distinct_failures": [], "success_contrasts": [], "counterevidence": [], "uncertainty": []}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rows = records()
        self.ids = ["q0", "q1", "q2"]

    def matrix(self, rows=None, **kw):
        return response_tensor(self.rows if rows is None else rows, self.ids, MODELS, [0.0, 1.0], **kw)

    def test_stable_disagreement_separated_from_sampling_noise(self):
        matrix = self.matrix()
        self.assertAlmostEqual(matrix['q2']['disagreement'], math.log(2))
        self.assertEqual(matrix['q2']['diagnostics']['within_condition_entropy'], 0)
        for r in self.rows:
            if r['task_id'] == 'q2':r['success'] = bool(r['trial']);r['reward'] = float(r['success'])
        matrix = self.matrix()
        self.assertAlmostEqual(matrix['q2']['disagreement'], 0)
        self.assertAlmostEqual(matrix['q2']['diagnostics']['within_condition_entropy'], math.log(2))

    def test_temperature_effect_is_not_model_identity(self):
        for r in self.rows:r['success'] = bool(r['temperature']);r['reward'] = float(r['success'])
        matrix = self.matrix()
        self.assertAlmostEqual(matrix['q0']['disagreement'], 0)
        self.assertAlmostEqual(matrix['q0']['diagnostics']['temperature_given_model'], math.log(2))

    def test_duplicate_and_missing_records_rejected(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]]]:
            with self.assertRaises(ValueError):self.matrix(rows)

    def test_changed_inputs_and_stale_harness_rejected(self):
        for field in ['input_hash', 'harness_hash']:
            rows=copy.deepcopy(self.rows);rows[0][field]='changed'
            with self.assertRaises(ValueError):self.matrix(rows)

    def test_non_evolve_and_infrastructure_errors_rejected(self):
        for key,value in [('split','heldout'),('status','error'),('reward',float('nan')),('success',1)]:
            rows=copy.deepcopy(self.rows);rows[0][key]=value
            with self.assertRaises(ValueError):self.matrix(rows)

    def test_select_disagreement_and_determinism(self):
        matrix=self.matrix()
        a=select_queries(matrix,1,method='disagreement',reserve=False)
        self.assertEqual(a['query_ids'],['q2'])
        self.assertEqual(a,select_queries(matrix,1,method='disagreement',reserve=False))

    def test_selection_rejects_unknown_method_and_dimensions(self):
        matrix=self.matrix()
        with self.assertRaises(ValueError):select_queries(matrix,1,method='typo')
        matrix['q0']['vector'].pop()
        with self.assertRaises(ValueError):select_queries(matrix,1)

    def test_family_balance(self):
        from mad_rsi.validation import family_weights
        weights=family_weights([{'id':'a','family':'x'},{'id':'b','family':'x'},{'id':'c','family':'y'}])
        self.assertEqual(weights,{'a':.25,'b':.25,'c':.5})

    def test_bundle_merges_traces_keeps_all_denominators(self):
        bundle=build_bundle('q2',self.rows,MODELS)
        self.assertEqual(bundle['n_trials'],8)
        self.assertEqual(len(bundle['evidence']),2)
        self.assertEqual(sum(len(e['sources']) for e in bundle['evidence']),8)
        self.assertEqual(bundle['n_queries'],1)

    def test_bundle_budget_and_fabricated_references(self):
        with self.assertRaises(ValueError):build_bundle('q2',self.rows,MODELS,max_chars=20)
        bundle=build_bundle('q2',self.rows,MODELS);d=diagnosis(bundle)
        validate_digest(d,bundle)
        d['shared_failures'][0]['evidence_ids']=['invented']
        with self.assertRaises(ValueError):validate_digest(d,bundle)

    def test_adaptive_probe_validated_and_cost_sensitive(self):
        action=choose_probe(self.rows,self.ids,MODELS,[0.,1.],'uncertainty_per_cost')
        self.assertEqual(action['trial_start'],2)
        rows=copy.deepcopy(self.rows);rows[0]['tokens']=None
        with self.assertRaises(ValueError):choose_probe(rows,self.ids,MODELS,[0.,1.],'uncertainty_per_cost')

    def test_atomic_json_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.json';write_json(p,{'version':1});write_json(p,{'version':2})
            self.assertEqual(read_json(p),{'version':2})
            with self.assertRaises(ValueError):write_json(p,{'value':float('nan')})
            self.assertEqual(read_json(p),{'version':2})


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.rows=records(temperatures=(0.,))
        self.grid={'model_ids':['probe-a','probe-b'],'task_ids':['q0','q1','q2'],'trials':[0,1]}

    def test_known_union_intersection(self):
        result=model_coverage(self.rows,**self.grid)
        self.assertAlmostEqual(result['pass_at_m'],2/3)
        self.assertAlmostEqual(result['pass_all_m'],1/3)
        self.assertEqual(result['paired_groups'],6)

    def test_repeat_alignment_not_best_of_samples(self):
        rows=[r for r in self.rows if r['task_id']=='q0']
        for r in rows:r['success']=(r['model_id']=='probe-a') == (r['trial']==0)
        x=model_coverage(rows,model_ids=['probe-a','probe-b'],task_ids=['q0'],trials=[0,1])
        self.assertEqual(x['pass_at_m'],1);self.assertEqual(x['pass_all_m'],0)

    def test_missing_duplicate_and_extra_rejected(self):
        for rows in [self.rows[:-1],self.rows+[self.rows[0]]]:
            with self.assertRaises(ValueError):model_coverage(rows,**self.grid)

    def test_mixed_harness_protocol_sampling_and_boolean_rejected(self):
        for key,value in [('harness_hash','other'),('protocol_hash','other'),('temperature',1.),('success',1),('status','error')]:
            rows=copy.deepcopy(self.rows);rows[0][key]=value
            with self.assertRaises(ValueError):model_coverage(rows,**self.grid)

    def test_cross_arm_input_and_sampling_guards(self):
        for key,value in [('input_hash','other'),('temperature',1.),('protocol_hash','other')]:
            other=copy.deepcopy(self.rows)
            for row in other:row[key]=value
            with self.assertRaises(ValueError):compare_harnesses({'first':self.rows,'second':other},**self.grid)


class Backend:
    def collect(self,harness,task_ids,models,config):return records(task_ids,config.temperatures,harness)
    def diagnose(self,bundle,system):return diagnosis(bundle)
    def propose(self,harness,feedback,round_index):return ['version-1','version-2']
    def evaluate(self,harness,round_index):return {'value':int(harness[-1])}
    def accept(self,incumbent,candidate):return {'accepted':candidate['value']>incumbent['value']}
    def select(self,incumbent,evaluations,decisions):return max(evaluations,key=lambda h:evaluations[h]['value'])


class AdapterTests(unittest.TestCase):
    def test_recursive_step_keeps_host_selection(self):
        x=run_round('version-0',['q0','q1','q2'],MODELS,EvidenceConfig(1,reserve_anchors=False),Backend())
        self.assertEqual(x['next_harness'],'version-2')
        self.assertEqual(x['selection']['query_ids'],['q2'])
        self.assertEqual(len(x['candidates']),2)

    def test_unknown_winner_rejected(self):
        backend=Backend();backend.select=lambda *args:'unknown'
        with self.assertRaises(ValueError):run_round('version-0',['q0'],MODELS,EvidenceConfig(1),backend)

    def test_fixed_selection_and_parallel_diagnosis(self):
        x=run_round('version-0',['q0','q1','q2'],MODELS,EvidenceConfig(2,workers=2),Backend(),selected_ids=['q0','q2'])
        self.assertEqual(list(x['feedback']),['q0','q2'])

    def test_failure_not_silently_accepted(self):
        backend=Backend();backend.evaluate=lambda *args:(_ for _ in ()).throw(RuntimeError('transport unavailable'))
        with self.assertRaises(RuntimeError):run_round('version-0',['q0'],MODELS,EvidenceConfig(1),backend)

    def test_confirmation_pairing_guard(self):
        left=records(temperatures=(0.,));right=copy.deepcopy(left)
        for rows in [left,right]:
            for row in rows:row['split']='confirmation'
        right[0]['input_hash']='changed'
        tasks=[{'id':q,'group':q} for q in ['q0','q1','q2']]
        with self.assertRaises(ValueError):paired_group_decision(left,right,MODELS,tasks,2,{})

if __name__ == '__main__':unittest.main()
