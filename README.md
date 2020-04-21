# dsbox-ta2
### FINAL VERSION
This is the final release version of dsbox-ta2 system. The dockerlized version is available at https://hub.docker.com/repository/docker/ckxz105/dsbox-ta2 (public). This docker has all primitives installed and ready to run except some speicial primitives that required static files.

If running with some special primitives(most are images/audio related), you need to download the static files. 
To download them, please add extra `python3 -m d3m index download` inside the docker build file (available at dockerBuildFiles folder of this repo).

### Usage
To use the dsbox-ta2 system only. Run with 
```
docker run --entrypoint /user_opt/client.sh \
       --name $docker_name \
       -m 200G \
       --shm-size=50g \
       --cpus=20 \
       -e D3MRUN=ta2 \
       -e D3MINPUTDIR=/input \
       -e D3MOUTPUTDIR=/output \
       -e D3MLOCALDIR=/tmp \
       -e D3MSTATICDIR=/static \
       -e D3MPROBLEMPATH=/input/TRAIN/problem_TRAIN/problemDoc.json \
       -e CUDA_VISIBLE_DEVICES=$gpu_id \
       -e D3MCPU=$D3MCPU \
       -e D3MRAM=50 \
       -e D3MTIMEOUT=$D3MTIMEOUT \
       -e DSBOX_LOGGING_LEVEL="dsbox=DEBUG:console_logging_level=WARNING:file_logging_level=DEBUG" \
       -v ${dataset_dir}/${problem}:/input \
       -v ${output_dir}:/output \
       -v /data/1/dsbox/static_files/static/:/static \
       $docker_image
```

The cpu / memory size can be modified by the users. According to different problems given. The logging level can also be changed.
The important part is to sent corresponding dataset path {dataset_dir}, problem name {problem} and output path {output_dir}.

### Input and Output format
The input dataset should follow the requirement of D3M projects. 
There is one sample dataset available at `dsbox-ta2/unit_tests/resources`
For detail structures of the dataset, please refer to https://gitlab.com/datadrivendiscovery/data-supply/-/tree/v4.0.0 (public repo)
Existed datasets: https://gitlab.datadrivendiscovery.org/d3m/datasets. (D3M users only)

### Output
The main focus output will be some json pipeline files at `output/pipelines_ranked`. Those pipeline files can then be used to run and make predictions. For detail using of those pipelines, please refer to https://gitlab.com/datadrivendiscovery/d3m/-/tree/v2020.1.9/docs

There is also a `score` folder which has the prediction scores based on the score dataset. If the score file is empty, it means the pipeline failed.
