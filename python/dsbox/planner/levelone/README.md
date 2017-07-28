# L1 Planner

## Sample usage:

Generate random pipeline
```python
import dsbox.planner.levelone
primitives = dsbox.planner.levelone.DSBoxPrimitives()
planner = dsbox.planner.levelone.LevelOnePlanner(primitives=primitives, ignore_preprocessing=False)
pipelines = planner.generate_pipelines(20)
for pipeline in pipelines:
	print(pipeline)
```

Generate random pipeline based on stochastic policy. The policy below
states that the primitive pair, 'Descritization' and 'NaiveBayes', should
occur together frequently, as well as teh primitive pair 'Normalization' and 'SVM'.

```python
import dsbox.planner.levelone
primitives = dsbox.planner.levelone.DSBoxPrimitives()
planner = dsbox.planner.levelone.LevelOnePlanner(primitives=primitives, ignore_preprocessing=False)

policy = dsbox.planner.levelone.AffinityPolicy(primitives)
policy.set_symetric_affinity('Descritization', 'NaiveBayes', 1)
policy.set_symetric_affinity('Normalization', 'SVM', 1)

pipelines = planner.generate_pipelines_with_policy(policy, 20)
for pipeline in pipelines:
	print(pipeline)
```

Generate random seed pipeline using 'curated' D3M primitives. Randomly
select one primitive from each level 2 subtree of the classification
primitive hierarchy.

```python
import dsbox.planner.levelone
primitives = dsbox.planner.levelone.D3mPrimitives()
planner = dsbox.planner.levelone.LevelOnePlanner(primitives=primitives)

pipelines = planner.generate_pipelines_with_hierarchy(level=2)
for pipeline in pipelines:
	print(pipeline)
```

Print curated primitve hierarchies, and primtive counts
```
python dsbox/planner/levelone/primitive_statistics.py
```
