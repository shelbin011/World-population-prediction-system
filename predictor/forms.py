from django import forms


class PredictionForm(forms.Form):
    country = forms.CharField(max_length=100, initial='India')
    degree = forms.IntegerField(min_value=1, max_value=10, initial=3)
    future_years = forms.CharField(max_length=200, required=False, 
                                  help_text="Comma-separated years to predict")
