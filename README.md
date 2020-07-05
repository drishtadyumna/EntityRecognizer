# How to Use?

Clone the repository to your local computer.

Change your current working directory to the cloned repository directory path on your computer.

In your terminal or Python bash, write the following command:

$ python entityRecognizer.py -i home/test_data.xlsx -o home/

Make sure that model.hdf5, tokenizer.pickle are in the same directory as the entityRecognizer.py file.

The second argument('home/test_data.xlsx') is the full relative path to the excel file containing the Narrations. 

If the third argument is 'home/' for example, then the resulting file's path will be 'home/results.xlsx'

# Additional Requirements

Tensorflow 2.0
Scikit Learn
Pandas
Numpy

# Other files

 Training.ipynb is the colab jupyter notebook which was used to train the Tensorflow model on the data contained in raw_data.xlsx.


