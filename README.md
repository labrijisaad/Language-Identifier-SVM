# 📙 `Language-Identifier with SVM in python` 🐍:
<p align="center">
  <img src="https://user-images.githubusercontent.com/74627083/181483400-7c02cb91-512f-48b4-b1fb-577148639791.png" />
</p>

- 🎯 In this notebook, I tried to write a script capable of idenetifying the language used in a text. 
- 🛠️ The supported languages are:  **`Swahili`**, **`Wolof`**, **`French`**, **`English`**, **`Arabic`** and **`Dyula`**.
> The code in this notebook can easily be modified to add or modify the languages ​​you want to detect: if you want to add other languages, just add a training dataset on the targeted languages, this dataset can be found by example on [HuggingFace Datesets](https://huggingface.co/datasets?sort=downloads).

<br>

- You can fin the **`model`** and the **`vectorizer`** in the **`/model`** directory. (you can also fin the **`python script`**)
- Here are TWO ways to use the trained model in notebook: (You must before installing the requirements)

      !pip install pickle sys pandas

##### meth 1
> via model and vectorizer import

    import pickle
    import pandas as pd

    SVM_model = pickle.load(open('model/SVM_model_language_identifier.pkl', 'rb'))
    SVM_vectorizer = pickle.load(open("model/SVM_vectorizer.pk","rb"))


    def predict_language(text):
        serie = pd.Series(text)
        vector = SVM_vectorizer.transform(serie)
        return str(SVM_model.predict(vector)[0])

##### meth 2
> by calling a python script that does all the work for us

    text = "I'm not really into the birthday thing honestly but I admit this was a really chill"
    var = !python model/language_identifier.py $text 
    print(var[-1])



- 📫 Feel free to contact me if anything is wrong or if anything needs to be changed 😎!  **labrijisaad@gmail.com**

<a href="https://colab.research.google.com/github/labrijisaad/Language-Identifier-SVM" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
