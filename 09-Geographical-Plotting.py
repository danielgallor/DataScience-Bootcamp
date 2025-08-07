import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas as pd

init_notebook_mode(connected=True) 

data = dict(type ='choropleth', 
            locations = ['AZ', 'CA', 'NY'],
            locationmode ='USA-states',
            colorscale = 'Portland', 
            text =['text1', 'text2', 'text3'],
            z = [1.0,2.0,3.0],
            colorbar = {'title':'Colorbar Title'})

layout = dict (geo = {'scope': 'usa'})
choromap = go.Figure(data =[data], layout = layout)

choromap.write_html("USA choropleth map.html", auto_open=True)


iplot(choromap)

# ----

df = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\09-Geographical-Plotting\\2011_US_AGRI_Exports.csv")
df.head()

data = dict (
    type  ='Choropleth',
    colorscale = 'YIOrRd',
    locations = df['code'],
    z = df['total exports'],
    locationmode = 'USA-states',
    text = df['text'],
    marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
    colorbar = {'title':"Millions USD"}
    )

layout = dict( 
    title = '2011 USA Agriculture Exports by State',
    geo = dict( scope = 'usa',
               showlakes = True,
               lakecolor = 'rgb(85,173,240)'
    )
)

choromap2 = go.Figure( data = [data], layout = layout)
choromap2.write_html("USA Agricultiral Exports Map.html", auto_open=True)