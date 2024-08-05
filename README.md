# Installation
The CLAM framework can be installed using PyPI:
```
pip install -r requirements.txt
cd compiler/
pip install -e .
```

# Experiments
The files and data used in the experiments are available in the `compiler/tests` directory.

Using the optuna scripts will generate a SQLite database, containing all of the information in the studies.
To get the top configuration within a study, we use the following SQL query:
```sql
WITH
top_trials AS (
    SELECT t.trial_id, t.study_id, tv.value
    FROM (select * from trials WHERE state='COMPLETE' order by trial_id asc limit 100) t
    JOIN trial_values tv ON t.trial_id=tv.trial_id
    ORDER BY tv.value DESC
)
SELECT t.value, t.trial_id, tp.param_name, tp.param_value, tp.distribution_json
FROM top_trials t
JOIN trial_params tp ON t.trial_id=tp.trial_id
ORDER BY t.trial_id ASC;
```

This will return all of the information necessary to recreate the `AbstractTransformer` that was used in the study.

## Data
All of our data can be found in the `compiler/tests/databases` and `compiler/tests/databases-gemma` folders.
The first directory contains all of our databases for experiment 1, while the second contains all of our databases used for experiment 2 and 3.
