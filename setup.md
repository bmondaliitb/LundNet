## Create venv
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd LundNet
pip install -e .
```

# check torch version
```python
python3 -m venv venv
import torch
print(torch.__version__)
```


```python
# install dgl from https://www.dgl.ai/pages/start.html
# select correc configuration
# example:
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html
```

## edit these files here

```text
cd ../test-venv/lib/python3.14/site-packages/dgl/
vim utils.py, frame.py, view.py, batched_graph.py

#change collections to collections.abc in the following imports in those files

from collections.abc import Iterable
from collections.abc import Mapping, Iterable
from collections.abc import MutableMapping
from collections import namedtuple

```

## Run (test)

```
lundnet --demo --save test --device cpu --num-epochs 1

```

