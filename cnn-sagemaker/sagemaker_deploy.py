"""Launch a SageMaker training job and deploy the model to a real-time endpoint."""
import os, sagemaker
from sagemaker.tensorflow import TensorFlow

ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
session = sagemaker.Session()

estimator = TensorFlow(
    entry_point="train.py", source_dir=".", role=ROLE,
    instance_count=1, instance_type="ml.m5.xlarge",
    framework_version="2.13", py_version="py310",
    hyperparameters={"framework": "tf", "epochs": 10},
    sagemaker_session=session)
estimator.fit()

predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m5.large",
                             endpoint_name="cifar10-cnn")
print("Deployed:", predictor.endpoint_name)
# predictor.predict(np.random.rand(1, 32, 32, 3).tolist())
# predictor.delete_endpoint()   # clean up to avoid charges
