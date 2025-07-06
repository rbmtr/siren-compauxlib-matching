# Matching of CompAuxLib and code SIREN

## Goal
The project provide potential SIREN codes to each CompAuxLib in a FEC by a fuzzy match of the CompAuxLib label with the name of the companies.

## Fonctioning

### Mode *batch*
The algorithm evaluates the Jaccard similarity between the CompAuxLib and the entries of the SIREN database. In order to account for potential mistakes in the CompAuxLib, a *shingling* stage is applied. For a fast search of potential candidate companies, we have applied the *Locality Sensitive Hashing*. For more information on the technique
https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/.

#### Execution
The application can be executed from the *src* folder via:
```bash
python main.py <task> [--environment] [--force_training] [--fec]
```
with:

*task* : the task to be executed. The possible values are [train, predict].

*--environment* (ou *-e*) : the environment of execution. The possible values are [local, remote].
When the variable is set to *local*, the execution recover the necessary inputs from a local folder, and the results are as well stored locally. When the variable is set to *remote*, the input and the outputs are recovered and stored in a Blob Storage Azure. The task *prediction* behaves differently, with the SQLite database for the FEC that is expected to be stored locally (the remote execution in this case only recovers the necessary files from the Blob Storage).

*--force_training* (ou *-f*) : perform a new training task, deleting any existing file.

*--fec* : the path to the local SQLite database with the FEC to pass to the prediction phase. It is a mandatory parameter for the task *predict*.

All the necessary configurations are stored in *config/appsetting.json*.

#### Current limits and constraints
- Currently the algorithm only looks for the moral entities in the SIREN database.
- The creation of the LSH takes >30 min and it has to be done at every update of the SIREN database;
- In order to perform a prediction, a training phase has to be completed previously, in the main environment chosen for the prediction. In particular, it is necessary that the following files are availabe either locally or in a Azure Storage :
[*denominationUniteLegale_hashtable_\*.parquet*, *denominationUniteLegale_hashtables_metadata.json*,
*discarded_token.parquet*].