from django.shortcuts import render
from .forms import PredictionForm
import pandas as pd
import base64
from io import BytesIO
from ml_model import prepare_data, train_polynomial_model, predict_years, plot_fit
import os


# Create your views here.

def index(request):
    context = {}
    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            country = form.cleaned_data['country']
            degree = form.cleaned_data['degree']
            future_years_str = form.cleaned_data['future_years']
            # Use default CSV
            csv_path = os.path.join(os.path.dirname(__file__), '..', 'world_population.csv')
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                context['error'] = f"Could not read default CSV: {e}"
                context['form'] = form
                return render(request, 'predictor/index.html', context)
            
            try:
                new_df = prepare_data(df, country=country)
                poly, model = train_polynomial_model(new_df, degree=degree)

                years = []
                for part in future_years_str.split(','):
                    p = part.strip()
                    if p:
                        try:
                            years.append(int(p))
                        except ValueError:
                            continue
                preds = predict_years(poly, model, years) if years else []

                context.update({
                    'training_data': new_df.to_html(classes='table table-bordered', index=False),
                    'predictions': list(zip(years, map(int, preds))) if years else [],
                })

                # build plot image
                fig = plot_fit(new_df, poly, model)
                buf = BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
                context['plot'] = plot_data
            except Exception as e:
                context['error'] = str(e)
    else:
        form = PredictionForm(initial={'country': 'India', 'degree': 3, 'future_years': '2030,2040,2050'})
    context['form'] = form
    return render(request, 'predictor/index.html', context)
