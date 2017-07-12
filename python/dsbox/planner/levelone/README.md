# L1 Planner

## Sample usage:

Generate random pipeline
```python
import dsbox.planner.levelone
primitives = dsbox.planner.levelone.Primitives()
planner = dsbox.planner.levelone.LevelOnePlanner(primitives=primitives)
policy = dsbox.planner.levelone.AffinityPolicy(primitives)
pipelines = planner.generate_pipelines(20, ignore_preprocessing=False)
for pipeline in pipelines:
	print(pipeline)
```

Generate random pipeline based on stochastic policy. The policy below
states that the primitive pair, 'Descritization' and 'NaiveBayes', should
occur together frequently, as well as teh primitive pair 'Normalization' and 'SVM'.

```python
import dsbox.planner.levelone
primitives = dsbox.planner.levelone.Primitives()
planner = dsbox.planner.levelone.LevelOnePlanner(primitives=primitives)

policy = dsbox.planner.levelone.AffinityPolicy(primitives)
policy.set_symetric_affinity('Descritization', 'NaiveBayes', 1)
policy.set_symetric_affinity('Normalization', 'SVM', 1)

pipelines = planner.generate_pipelines_with_policy(policy, 20, ignore_preprocessing=False)
for pipeline in pipelines:
	print(pipeline)
```
