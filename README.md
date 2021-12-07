# Dynamic Human Evaluation for Relative Model Comparisons
The source code for the implementation of a simulation framework of two-choice human evaluation, PyTorch implementation of CGA, mturk pre-processing, mturk human evaluation data, and human evaluation analysis. 

## Simulation Framework
The directory *simulated-evaluation* contains the implementation for the simulated human evaluation. 

#### Environment setup
- Donwload the latest version of miniconda: ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh```
- Run the miniconda3 installer
- Run conda: ```eval "$(/your/direcory/miniconda3/bin/conda shell.bash hook)"```
- Update conda to get the latest dependencies: ```conda update conda```
- Create conda environment: ```conda env create -f environment.yml```
- Change to the newly created environment: ```conda activate msc_env```

#### Experiment instructions
- Create a simulated evaluation data: ```python two_choice_evaluation_model.py --seed 21 --n_iterations 1000 --n_workers 100 --min_worker_capability 0.8 --max_worker_capability 1.0 --mean_request_difficulties 0.25 0.125 --n_requests 3500 5000 --delta 0.001 --create_df 1```
- Data will be stored: simulated-evaluation/dataframes/run_*run_id*_*run_date*.
- Run an analysis on the newly created data: ```python two_choice_evaluation_model.py --seed 21 --n_iterations 1000 --n_workers 100 --min_worker_capability 0.8 --max_worker_capability 1.0 --mean_request_difficulties 0.25 0.125 --n_requests 3500 5000 --delta 0.001 --create_df 0 ---write_plots 1 --simulation_id *run_id* --simulation_date *run_date*```
- csv results files are included in *raw_data_results* and visualisations in *visual_analysis* under the newly created experiment id.

## Human Evaluation

### Control Generate Augment (CGA)
The directory *control-generate-augment* includes the adapted CGA framework to train several model versions to compare on Amazon Mechanical Turk. 

#### Environment setup
- Make sure that correct environment is activated: ```conda activate msc_env```
- Run: ```pip install spacy```
- Run: ```python -m spacy download en_core_web_sm```
- Run: ```pip install nltk```
- Run: ```python``` and from there:
```
>>> import nltk 
>>> nltk.download('punkt') 
```
#### Pre-processing
- Download the yelp restaunrant dataset [here](https://github.com/shentianxiao/language-style-transfer/tree/master/data/yelp) and place it in a data folder in the control-generate-augment directory such that we have the following data directory: *control-generate-augment/data/yelp*.
- Run ```python yelp_pre_processing.py``` inside the *control-generate-augment/pre_processing* directory to configure the data setup for CGA and to create pronoun and tense label.

#### Train CGA
To train CGA with multiple attributes we provide few examples commands below. Run one of the below commands inside the *control-generate-augment/multiple_attribute* directory:

**[L(ADV) + standard WD] - dropout rate = 0.7**

```python analysis.py --gpu 4 --samples -1 --word_dropout 0.7 --latent_size 32 --x0 12000 --word_drop_type static --delta 0.5 --back False --hs_rnn_discr 64 2>&1|tee train.log```

**L[(CTX) + cyclical WD]**

```python analysis.py --gpu 6 --samples -1 --word_dropout 0.7 --latent_size 32 --x0 12000 --word_drop_type cyclical --delta 0.5 --back True --hs_rnn_discr 64 2>&1|tee train.log```

The trained models will be stored in *control-generate-augment/multiple_attribute/bins*.

#### Generate data using CGA model
- Update the date and epoch variables in *generation.py*.
- Generate data: ```python generation.py```

#### Attribute matching for the generated data
- Copy the txt file that contains the generated data into *human-evaluation-preprocessing/attr_generated_output*.
- Update the filename for the models being evaluated in *human-evaluation-preprocessing/automatic_evaluation/attribute_matching.py*.

##### Train sentiment classifier on the Yelp data
- If the pre-trained model does not exists we train a textCNN on the yelp dataset. 
- Preprocessing: Copy the sentiment .csv files for yelp into *human-evaluation-preprocessing/automatic_evaluation/yelp_data*.
- In *yelp_data* run ```python create_json.py``` to generate json files for PyTorch.
- To train the network run: ```python sentiment_classification.py```
- The trained model is stored as *tut4-model.pt* in *human-evaluation-preprocessing/automatic_evaluation*. Rename the selected model as *textcnn-model* so it won't be overwritten.
- Once we have the trained model we can run: ```python attribute_matching.py```

### Data Pre-Processing for AMT
#### MTurk preprocessing
- In *human-evaluation-preprocessing/mturk_preprocessing* run: ```python create_mturk_input_file.py```
- That script, prepares the datafiles to be published on Amazon Mechanical Turk for human evaluation, metadata file with source information and attribute combination overview.
- Base files are also saved in *mturk/results/mturk_source_files/* which are used for post-processing with the batch data retrieved from MTurk.

### MTurk Data
#### MTurk postprocessing
- Download the batchfile from MTurk and depending on MTurk environment (sandbox or production) place the batch files in the corresponding folder in *mturk/results* and congifure parameters for post processing accordingly in *mturk/mturk_post_processing.py*. Note: the batch results for the reported AMT experiments are already available in *mturk/results/production*.
- Run ```python mturk_post_processing.py```

### Human Evaluation Analysis
The *human-evaluation* directory contains all relevant implementations to analyse the collected human judgements. 

Experiments: 
- Batch_4444974: GCA vs V1
- Batch_4447602: GCA vs V2 (R1)
- Batch_4483006: GCA vs V2 (R2)

#### Experiment instructions
- The file ```shared_function.py``` includes variable configuration as global variables modified directly in the file.
- Run analysis for all batches: ```./bash_two_choice_mturk```