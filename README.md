# 1. dsbox-ta2
The DSBox TA2 component

# 2. TA2 Evaluation
Each TA2 system will be evaluated by NIST. The vehicle for delivery of the TA2 system to NIST is a Docker Image. This
docker image must strictly adhere to an API that will allow their automated evaluation system to run and extract results
from it. This same evaluation system can be used to verify the correctness of candidate images.

Full details are available here: https://datadrivendiscovery.org/wiki/pages/viewpage.action?spaceKey=gov&title=TA2+Submission+Guide

## 2.1 Creating the ISI Docker Image
This section details the distilled commands that need to be run to create the docker image. The documentation mentioned
above is very good but contains a lot of options and variables that need to be navigated in order to come up with the
correct credentials, urls and commands.

1. In the same directory as this readme file run the following:
    
    ``` docker build -f Dockerfile -t registry.datadrivendiscovery.org/ta2/isi_ta2:latest . ```
    
2. Get the id of the image that was just created for use in the next step:

    ```
    > docker images
    REPOSITORY                   TAG                 IMAGE ID            CREATED             SIZE
    isi-test-0.1                 latest              2df1c97d392b        9 hours ago         2.43GB
    ``` 
    
    
3. To test the docker image before pushing to the nist docker registry, run the following command. Notice the paths to 
   config.json - the local copy of the file will need the absolute path. This also goes for the sample data set that
   you want to run (in the case below, baseball)

    ``` 
    > docker run -i --entrypoint /bin/bash -v /home/config.json:/home/code/dsbox-ta2/config.json -v 
    /home/data/o_185:/baseball/ 2df1c97d392b -c 'ta2_search /home/code/dsbox-ta2/config.json' 
    ```

4. If the docker run completes with pipelines created then you are ready to upload the image to the NIST Docker Registry.
   If you do not have an account for this let Daragh Hartnett (daragh.hartnett@sri.com) know and he will get you the 
   credentials you need. Run the following commands:

   ```  
   > docker login registry.datadrivendiscovery.org
   > docker push registry.datadrivendiscovery.org/ta2/isi_ta2:latest 
   ```
   
5. Run the Docker CI pipeline by opening the following in a browser: 

      https://gitlab.datadrivendiscovery.org/TA2/ISI_ta2
    
   - Click on 'Pipelines' in the top menu and then 'Pipelines' in the sub menu. 
   - Click on Run Pipeline
   - Click on Create Pipeline
   - Click on the Play button in the setup node.
   - In the logs, you should be able to see the latest image id (63e81a8c1fed for example) embedded in the long 
   identifier for the latest docker image being used (Using docker image registry.datadrivendiscovery.org/ta2/isi_ta2:
   latest ID=sha256:**63e81a8c1fed**c2beabb2e8ee3b2cb84b4ed96dc97ebc071d0a2217d984716708...)
   

