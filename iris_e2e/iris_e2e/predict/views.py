from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .models import PredResults

def predict(request):
    return render(request, 'predict.html')

def predict_chances(request):
    print('1')
    # to check whether we are using post method
    if request.POST.get('action') == 'post':
        
        print('2')
        # Receive data from client (html page)
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))
        print('3')
        # Unpickle model
        model = pd.read_pickle(r"G:\DjangoProject\iris_e2e\new_model.pickle")
        print('4')
        # Make prediction
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        print('5')
        classification = result[0]
        print('6')
        PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                                   petal_width=petal_width, classification=classification)
        print('7')
        # now we will return the result back to predict page through json format
        return JsonResponse({'result': classification, 'sepal_length': sepal_length,
                             'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width},
                            safe=False)

def view_results(request):
    # Submit prediction and show all

    #extract data from database
    data = {"dataset": PredResults.objects.all()}
    #send data to database
    return render(request, "results.html", data)