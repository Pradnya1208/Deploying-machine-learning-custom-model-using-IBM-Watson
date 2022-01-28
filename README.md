<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Deploying machine learning custom model using IBM Watson</div>
<div align="center"><img src="https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/images/intro.gif?raw=true"></div>


## Overview:
In this project, we will understand step by step how to take a trained Scikit-Learn model and deploy it to the Cloud using Watson Machine Learning. This model can then be used for a whole bunch of applications and can even be used in different languages like Javascript, Scala, Java and Go.

In this project we’ll learn how to: 
- Save the model to Watson Machine Learning
- Creating online deployments with Python
- Scoring your model using the Python API

<br>
In this project, we will mainly focus on model deployment.

For more details on Machine Learning Model that we used for this project are [here](https://github.com/Pradnya1208/Telecom-Customer-Churn-prediction)


## Watson Machine Learning:
IBM Watson Machine Learning is a full-service IBM Cloud offering that makes it easy for developers and data scientists to work together to integrate predictive capabilities with their applications.



## 1) Import and install dependencies:
#### install IBM Watson Machine Learning

```
!pip install ibm_watson_machine_learning
```
#### Import dependencies
```
from ibm_watson_machine_learning import APIClient
import json 
import numpy as np
```
## 2) Create deployment space:
- go to `cloud.ibm.com` and login
- go to `catelog`
<br>

<img src="https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/images/step1.PNG?raw=true" width="100%">

<br>

- search for `machine learning`

<br>

<img src="https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/images/step2.PNG?raw=true" width="100%">

<br>

- select `location` and `lite` plan
- Access in Watson studio
<br>

<img src="https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/images/step3.PNG?raw=true" width="80%">
<br>

- go to `deployments` and select `new deployment space`

<br>

<img src="https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/images/step4.PNG?raw=true" width="80%">

<br>

- `name` the deployment space and select the `machine learning service` that you've created in previous steps.

<br>

You will be able to see your space in `deployments`

<br>

<img src="https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/images/step5.PNG?raw=true" width="60%">


## 3) Authentication:
```
from ibm_watson_machine_learning import APIClient
wml_credentials = {
                    "url":"https://us-south.ml.cloud.ibm.com",
                    "apikey":"<your api key>"
}

client = APIClient(wml_credentials)
```
#### For api key follow the following steps:
- go to `ibm cloud`
- select `manage` -> `Access(IAM)` -> `API keys`
<br>
<img src="https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/images/api.PNG?raw=true" width="100%">


#### Where to get you space ID?
```
# create a new deployment space in services and software
space_uid = guid_from_space_name(client, "DeploymentSpace")
print("space_uid : ", space_uid)
```
Other way to get the space ID:

- go to `deployments` -> `manage`
<br>
<img src="https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/images/spacid.PNG?raw=true" width="80%">

```
client.set.default_space(space_uid)
```
## 4) Check the software specifications:
```
client.software_specifications.list()
```
```
-----------------------------  ------------------------------------  ----
NAME                           ASSET_ID                              TYPE
default_py3.6                  0062b8c9-8b7d-44a0-a9b9-46c416adcbd9  base
pytorch-onnx_1.3-py3.7-edt     069ea134-3346-5748-b513-49120e15d288  base
scikit-learn_0.20-py3.6        09c5a1d0-9c1e-4473-a344-eb7b665ff687  base
spark-mllib_3.0-scala_2.12     09f4cff0-90a7-5899-b9ed-1ef348aebdee  base
ai-function_0.1-py3.6          0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda  base
shiny-r3.6                     0e6e79df-875e-4f24-8ae9-62dcc2148306  base
tensorflow_2.4-py3.7-horovod   1092590a-307d-563d-9b62-4eb7d64b3f22  base
pytorch_1.1-py3.6              10ac12d6-6b30-4ccd-8392-3e922c096a92  base
tensorflow_1.15-py3.6-ddl      111e41b3-de2d-5422-a4d6-bf776828c4b7  base
scikit-learn_0.22-py3.6        154010fa-5b3b-4ac1-82af-4d5ee5abbc85  base
default_r3.6                   1b70aec3-ab34-4b87-8aa0-a4a3c8296a36  base
pytorch-onnx_1.3-py3.6         1bc6029a-cc97-56da-b8e0-39c3880dbbe7  base
tensorflow_2.1-py3.6           1eb25b84-d6ed-5dde-b6a5-3fbdf1665666  base
tensorflow_2.4-py3.8-horovod   217c16f6-178f-56bf-824a-b19f20564c49  base
do_py3.8                       295addb5-9ef9-547e-9bf4-92ae3563e720  base
autoai-ts_3.8-py3.8            2aa0c932-798f-5ae9-abd6-15e0c2402fb5  base
tensorflow_1.15-py3.6          2b73a275-7cbf-420b-a912-eae7f436e0bc  base
pytorch_1.2-py3.6              2c8ef57d-2687-4b7d-acce-01f94976dac1  base
spark-mllib_2.3                2e51f700-bca0-4b0d-88dc-5c6791338875  base
pytorch-onnx_1.1-py3.6-edt     32983cea-3f32-4400-8965-dde874a8d67e  base
spark-mllib_3.0-py37           36507ebe-8770-55ba-ab2a-eafe787600e9  base
spark-mllib_2.4                390d21f8-e58b-4fac-9c55-d7ceda621326  base
xgboost_0.82-py3.6             39e31acd-5f30-41dc-ae44-60233c80306e  base
pytorch-onnx_1.2-py3.6-edt     40589d0e-7019-4e28-8daa-fb03b6f4fe12  base
autoai-obm_3.0                 42b92e18-d9ab-567f-988a-4240ba1ed5f7  base
spark-mllib_2.4-r_3.6          49403dff-92e9-4c87-a3d7-a42d0021c095  base
xgboost_0.90-py3.6             4ff8d6c2-1343-4c18-85e1-689c965304d3  base
pytorch-onnx_1.1-py3.6         50f95b2a-bc16-43bb-bc94-b0bed208c60b  base
autoai-ts_3.9-py3.8            52c57136-80fa-572e-8728-a5e7cbb42cde  base
spark-mllib_2.4-scala_2.11     55a70f99-7320-4be5-9fb9-9edb5a443af5  base
spark-mllib_3.0                5c1b0ca2-4977-5c2e-9439-ffd44ea8ffe9  base
autoai-obm_2.0                 5c2e37fa-80b8-5e77-840f-d912469614ee  base
spss-modeler_18.1              5c3cad7e-507f-4b2a-a9a3-ab53a21dee8b  base
cuda-py3.8                     5d3232bf-c86b-5df4-a2cd-7bb870a1cd4e  base
autoai-kb_3.1-py3.7            632d4b22-10aa-5180-88f0-f52dfb6444d7  base
pytorch-onnx_1.7-py3.8         634d3cdc-b562-5bf9-a2d4-ea90a478456b  base
spark-mllib_2.3-r_3.6          6586b9e3-ccd6-4f92-900f-0f8cb2bd6f0c  base
tensorflow_2.4-py3.7           65e171d7-72d1-55d9-8ebb-f813d620c9bb  base
spss-modeler_18.2              687eddc9-028a-4117-b9dd-e57b36f1efa5  base
pytorch-onnx_1.2-py3.6         692a6a4d-2c4d-45ff-a1ed-b167ee55469a  base
do_12.9                        75a3a4b0-6aa0-41b3-a618-48b1f56332a6  base
spark-mllib_2.3-scala_2.11     7963efe5-bbec-417e-92cf-0574e21b4e8d  base
spark-mllib_2.4-py37           7abc992b-b685-532b-a122-a396a3cdbaab  base
caffe_1.0-py3.6                7bb3dbe2-da6e-4145-918d-b6d84aa93b6b  base
pytorch-onnx_1.7-py3.7         812c6631-42b7-5613-982b-02098e6c909c  base
cuda-py3.6                     82c79ece-4d12-40e6-8787-a7b9e0f62770  base
tensorflow_1.15-py3.6-horovod  8964680e-d5e4-5bb8-919b-8342c6c0dfd8  base
hybrid_0.1                     8c1a58c6-62b5-4dc4-987a-df751c2756b6  base
pytorch-onnx_1.3-py3.7         8d5d8a87-a912-54cf-81ec-3914adaa988d  base
caffe-ibm_1.0-py3.6            8d863266-7927-4d1e-97d7-56a7f4c0a19b  base
-----------------------------  ------------------------------------  ----
Note: Only first 50 records were displayed. To display more use 'limit' parameter.
```
## 5) Setting the python environment:
```
software_spec_uid = client.software_specifications.get_uid_by_name("default_py3.8")
```
## 6) Storing the model in space:
```
model_details = client.repository.store_model(model=bestAdaModFitted2, meta_props={
client.repository.ModelMetaNames.NAME: "Customer churn prediction",
client.repository.ModelMetaNames.TYPE: "scikit-learn_0.23",
client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid})

model_id = client.repository.get_model_uid(model_details)
```
## 7) Model deployment:
The model can be deployed in 2 ways:
### Using `deploy` option:
- go to `deployments` -> `spaces` -> select you space name -> select the `deploy` option infront of model
<br>
<img src="https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/images/deploy.PNG?raw=true" width="80%">


### Online model deployment method:
```
deployment_props = {
    client.deployments.ConfigurationMetaNames.NAME:"Sklearn Deployment", 
    client.deployments.ConfigurationMetaNames.ONLINE: {}
}

# Deploy
deployment = client.deployments.create(
    artifact_uid=model_id, 
    meta_props=deployment_props 
)
# Output result
deployment
```

## 8) Prediction using deployed model:
For prediction also we can use 2 ways:
### Using UID:
```
payload = {"input_data":
           [
               {"fields":X_test[0:1].columns.to_numpy().tolist(), "values":X_test[0:1].to_numpy().tolist()}
           ]
          }
```
```
deployment_uid = client.deployments.get_uid(deployment)
result = client.deployments.score(deployment_uid, payload)
result
```
```
{'predictions': [{'fields': ['prediction', 'probability'],
   'values': [[0, [0.5306418238572117, 0.46935817614278824]]]}]}
```
#### Result:
```
pred_values = np.squeeze(result['predictions'][0]['values']);
pred_values[0]
```
```
result is : 0
Which means customer won't churn.
```

### Using IBM Watson studio API reference:

Check the [Notebook](https://github.com/Pradnya1208/Deploying-machine-learning-custom-model-using-IBM-Watson/blob/master/Telecom%20customer%20churn%20prediction%20(IBM%20Watson%20studio%20API%20reference).ipynb) for implementation.
<br>

After deploying the model IBM watson studio generates an API reference URL for accessing the deployed model.

```
API_KEY = ""
token_response = requests.post('' data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
```
```
payload_scoring = {"input_data": [{'fields': ['gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges'],
    "values": [[0.0,
     0.0,
     0.0,
     0.0,
     -0.629446027814582,
     1.0,
     2.0,
     2.0,
     1.0,
     1.0,
     1.0,
     1.0,
     1.0,
     1.0,
     0.0,
     0.0,
     3.0,
     -1.322051671307397,
     -0.8000098038460207]]}]}
```
```
response_scoring = requests.post('<url>', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())
```
```
Scoring response
{'predictions': [{'fields': ['prediction', 'probability'], 'values': [[0, [0.5306418238572117, 0.46935817614278824]]]}]}
```
```
Customer will likely to continue with her subscription (won't churn)
```






## Licences:
`MIT License`


### Learnings:
`Model Deployment using IBM Watson Studio` 






## References:
[IBM Cloud](https://cloud.ibm.com/login)
<br>
[Watson Machine Learning](https://cloud.ibm.com/catalog/service...)
<br>
[Watson Studio](https://cloud.ibm.com/catalog/service...)
<br>
[Example deploying with Cloud Pak for Data](https://github.com/IBM/watson-machine...)
 


### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner



[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]
