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
primitives = DSBoxPrimitives()

policy = AffinityPolicy(primitives)
policy.set_symetric_affinity(primitives.get_by_name('Descritization'),
                             primitives.get_by_name('NaiveBayes'), 10)
policy.set_symetric_affinity(primitives.get_by_name('Normalization'),
                             primitives.get_by_name('SVM'), 10)

planner = LevelOnePlanner(primitives=primitives, policy=policy)

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


Given a pipeline, generate new pipelines with similar
classification/regression learners.

```python
primitives = get_d3m_primitives()
policy = AffinityPolicy(primitives)
planner = LevelOnePlanner(primitives=primitives, policy=policy)

pipelines = planner.generate_pipelines_with_hierarchy(level=level)

print(pipelines[4])
refined_pipeline = planner.find_similar_learner(pipelines[4], include_siblings=True)
for pipeline in refined_pipeline:
	print(pipeline)
```

Given a seed pipeline, use the primitive hierarchy to generate new
pipelines with feature extractor primitive filled in.

```python
refined_pipeline = planner.fill_feature_with_hierarchy(pipelines[4])
for pipeline in refined_pipeline:
	print(pipeline)
```

Given a seed pipeline, use the primitive weights to generate new
pipelines with feature extractor primitive filled in.

```python
refined_pipeline = planner.fill_feature_by_weights(pipelines[4], 5)
for pipeline in refined_pipeline:
	print(pipeline)
```

Tell the planner the media type is image. Now image related feature
extractors are weighted more.

```python
planner = LevelOnePlanner(primitives=primitives, policy=policy, media_type=VariableFileType.IMAGE)
refined_pipeline = planner.fill_feature_by_weights(pipelines[4], 5)
for pipeline in refined_pipeline:
	print(pipeline)
```
