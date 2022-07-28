import pickle
import sys
import pandas as pd


text = " ".join(sys.argv) # getting the argument

# Load the model
SVM_model = pickle.load(open('model/SVM_model_language_identifier.pkl', 'rb'))
SVM_vectorizer = pickle.load(open("model/SVM_vectorizer.pk", "rb"))


def predict_language(txt):
    serie_txt = pd.Series(txt)
    vector = SVM_vectorizer.transform(serie_txt)
    return str(SVM_model.predict(vector)[0])


def main():
    output = predict_language(text)
    print(output)


if __name__ == "__main__":
    try:
        main()
    except:
        print("Something went wrong :(")