# 📙 `Language-Identifier with SVM in python` 🐍
<p align="center">
  <img src="https://user-images.githubusercontent.com/74627083/181483400-7c02cb91-512f-48b4-b1fb-577148639791.png" />
</p>

- 🎯 In this project, I developed a script that can identify the language used in a given text. 
- 🛠️ The script currently supports the following languages: **`Swahili`**, **`Wolof`**, **`French`**, **`English`**, **`Arabic`** and **`Dyula`**.
- ⚠️ To obtain accurate results, the input text should be relatively long (at least 4-5 words). The script can be easily modified to add or modify the supported languages by adding a training dataset for the desired language, this dataset can be found by example on [HuggingFace Datesets](https://huggingface.co/datasets?sort=downloads).

<br>

- You can find the **`model`** and the **`vectorizer`** in the **`/model`** directory. (you can also find the **`python script`**: will be used in **`meth2`**)
- Here are **TWO** ways to use the trained model in notebook: (You must before install the requirements)

```py
    !pip install pickle sys pandas
```

##### meth 1
> via model and vectorizer import

```py
    import pickle
    import pandas as pd

    SVM_model = pickle.load(open('model/SVM_model_language_identifier.pkl', 'rb'))
    SVM_vectorizer = pickle.load(open("model/SVM_vectorizer.pk","rb"))


    def predict_language(text):
        serie = pd.Series(text)
        vector = SVM_vectorizer.transform(serie)
        return str(SVM_model.predict(vector)[0])
    
    text = "Na nga def ?" 
    print(predict_language(text))
    
    >>> wolof
 ```

##### meth 2
> by calling a script that does all the work for us

```py
    text = "I'm not really into the birthday thing honestly but I admit this was a really chill"
    var = !python model/language_identifier.py $text 
    print(var[-1])
    
    >>> english
```

- 💪 Model performance: Here are the results obtained after training the model

      wolof:  {'precision': 0.9956011730205279, 'recall': 0.9883551673944687, 'f1-score': 0.9919649379108838, 'support': 687}
      french:  {'precision': 0.9971264367816092, 'recall': 0.9788434414668548, 'f1-score': 0.9879003558718862, 'support': 709}
      swahili:  {'precision': 1.0, 'recall': 0.9849108367626886, 'f1-score': 0.9923980649619903, 'support': 729}
      english:  {'precision': 0.9683195592286501, 'recall': 0.9736842105263158, 'f1-score': 0.9709944751381215, 'support': 722}
      arabic:  {'precision': 0.9363354037267081, 'recall': 0.9741518578352181, 'f1-score': 0.9548693586698337, 'support': 619}
      dyula:  {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 691}
      
 > - Overall, it the SVM model is performing very well for all the languages, with high values for all performance metrics.

- 📫 Feel free to contact me if anything is wrong or if anything needs to be changed 😎!  **labrijisaad@gmail.com**

<a href="https://colab.research.google.com/github/labrijisaad/Language-Identifier-SVM" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

> - 🙌 Notebook made by [@labriji_saad](https://github.com/labrijisaad)
> - 🔗 Linledin [@labriji_saad](https://www.linkedin.com/in/labrijisaad/)
