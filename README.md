
# This folder includes Pytorch version of the  AttentiveChrome Implementation. 
You can run it via the following command: 

```
python train.py --cell_type Toy
```



## We also provide trained AttentiveChrome models through the Kipoi model zoo  [http://kipoi.org/](http://kipoi.org/)

AttentiveChrome model can be run using Kipoi, which is a repository of predictive models for genomics. All models in the repo can be used through shared API.

- The utility codes to adapt AttentiveChrome to Kipoi are in /kipoiutil

### Installation Requirements
* python>=3.5
* numpy
* pytorch-cpu
* torchvision-cpu

## Quick Start
### Creating new conda environtment using kipoi
`kipoi env create AttentiveChrome`


### Activating environment
`conda activate kipoi-AttentiveChrome`

## Command Line
We can run Attentive Chrome using a terminal.

### Getting example input file
To get an example input file for a specific model, run the following command. Replace {model_name} with the actual name of model (e.g. E003, E005, etc.)

`kipoi get-example AttentiveChrome/{model_name} -o example_file`

example: `kipoi get-example AttentiveChrome/E003 -o example_file`

### Predicting using example file
To make a prediction using an input file, run the following command.

`kipoi predict AttentiveChrome/{model_name} --dataloader_args='{"input_file": "example_file/input_file", "bin_size": 100}' -o example_predict.tsv`

This should produce a tsv file containing the results. To run it using another file, replace "example_file/input+file" with the path of your file.

## Python API
We can also use Attentive Chrome through the Kipoi Python API.
### Fetching the model
First, import kipoi:
`import kipoi`

Next, get the model. Replace {model_name} with the actual name of model (e.g. E003, E005, etc.)

`model = kipoi.get_model("AttentiveChrome/{model_name}")`

### Predicting using pipeline
`prediction = model.pipeline.predict({"input_file": "path to input file", "bin_size": {some integer}})`

This returns a numpy array containing the output from the final softmax function.

e.g. `model.pipeline.predict({"input_file": "data/input_file", "bin_size": 100})`

### Predicting for a single batch
First, we need to set up our dataloader `dl`.

`dl = model.default_dataloader(input_file="path to input file", bin_size={some integer})`

Next, we can use the iterator functionality of the dataloader.

`it = dl.batch_iter(batch_size=32)`

`single_batch = next(it)`

First line gets us an iterator named `it` with each batch containing 32 items. We can use `next(it)` to get a batch.

Then, we can perform prediction on this single batch.

`prediction = model.predict_on_batch(single_batch['inputs'])`

This also returns a numpy array containing the output from the final softmax function.

# robustness-genomics
