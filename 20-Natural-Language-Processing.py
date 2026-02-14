import nltk
nltk.download_shell()

messages = [line.rstrip() for line in open('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\20-Natural-Language-Processing\\smsspamcollection\\SMSSpamCollection')]
print(len(messages))

messages[1]