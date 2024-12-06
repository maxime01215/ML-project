# Miniproject BIO-322

## Project by : 
Maxime Lehmann SCIPER : 342169   
Mariem Maazoun SCiPER : 330223  

## Data Instructions : 
### Files

train.csv - the training set  
test.csv - the test set; contains only the features and not the purity level; use it to make predictions that you then upload in the same format as the sample_submission.csv  
sample_submission.csv - a sample submission file in the correct format  
substances.csv - infrared spectra of substances that may be mixed with heroin; the dataset contains multiple spectra for a given substance, each one obtained from a different sample.  

### Columns

sample_name - name of the sample  
device_serial - serial number of the device used to measure the sample  
substance_form_display - preparation of the sample  
measure_type_display - how the sample was measured  
prod_substance - name of the substance for which the purity should be measured  
PURITY - purity level in percentage  
901.1 … 1676.2 - channel intensities of the infrared spectrum  
