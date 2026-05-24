## Create venv
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd LundNet
pip install -e .
```

# check torch and CUDA availability
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## Run (test)

```
lundnet --demo --save test --device cpu --num-epochs 1

```
