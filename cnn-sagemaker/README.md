# Image Classification CNN – SageMaker
CNNs in TensorFlow and PyTorch, evaluation with sklearn metrics, deployment via Amazon SageMaker.
```bash
pip install -r requirements.txt
python train.py --framework tf --epochs 5      # or --framework torch
export SAGEMAKER_ROLE_ARN=...
python sagemaker_deploy.py
```
