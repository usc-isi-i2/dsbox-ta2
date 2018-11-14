![travis ci](https://travis-ci.org/usc-isi-i2/dsbox-ta2.svg?branch=master)

# DSBox: Automated Machine Learning System #

## Installation Instructions ##

### Installing DSBox using base D3M Image ###

Get docker image from:

```
docker pull registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.7.10-20180801-215033
```

Start the container: 

```
docker run -it 48a23667534e /bin/bash
```

Within the container do the following:

Create the directories:

```
mkdir /output
mkdir /input
mkdir -p /user_opt/dsbox
```

Installing DSBox software and install SKLearn and D3M common primitves:

```
cd /user_opt/dsbox

git clone https://github.com:usc-isi-i2/dsbox-ta2.git --branch eval-2018-summer
cp dsbox-ta2/d3mStart.sh /user_opt/
chmod a+x /user_opt/d3mStart.sh

pip3 install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap@9346c271559fd221dea4bc99c352ce10e518759c#egg=sklearn-wrap
pip3 install -e git+https://gitlab.com/datadrivendiscovery/common-primitives@fa865a1babc190055cb2a17cbdcd5b37e6f5e774#egg=common-primitives
```

To run the DSBox TA2 search:

```
/user_opt/d3mStart.sh
```

The above d3mStart.sh script assumes several D3M related shell environment variables are set. See this page for definition of the shell environment variables: [2018 Summer Evaluation - Execution Process](https://datadrivendiscovery.org/wiki/display/gov/2018+Summer+Evaluation+-+Execution+Process)

The search and test configuration file formats are defined here: [JSON Configuration File Format](https://datadrivendiscovery.org/wiki/pages/viewpage.actionpageId=11275766)


### Using Exisitng DSBox Docker Image ###

A pre-built DSBox Docker image on Docker Hub is here: [DSBox TA2 Image for 2018 Summer Evaluation](https://hub.docker.com/r/uscisii2/dsbox/)

Notes on how to run DSBox TA2 on non-D3M datasets is here: [Running DSBox TA2 on Other Datasets](https://github.com/usc-isi-i2/dsbox-ta2-system/blob/master/docker/dsbox_train_test/run_dsbox_with_other_dataset.md)
