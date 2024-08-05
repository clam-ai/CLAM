# Slimscale compiler

### Installation
Setup your virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
cd compiler
pip install -r requirements.txt
pip install -e .
```

### Testing
EleutherAI's [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness) can be run on models created using the `AbstractTransformer` class by wrapping the model with the `AbstractLMForEval` class as follows:
```python
from sscompiler.compiler.abstract import AbstractTransformer
from sscompiler.compiler.abstract_eval import AbstractLMForEval
import lm_eval

at = AbstractTransformer(...)
eval_at = AbstractLMForEval(at, model)
lm_eval.simple_evaluate(model=eval_at, ...)
```

See [eval_our_ft.py](tests/eval_our_ft.py) for an example.