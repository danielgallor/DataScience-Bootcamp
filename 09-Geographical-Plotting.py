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

choromap.write_html("USA Choropleth Map.html", auto_open=True)


iplot(choromap)

# ---- USA

df = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\09-Geographical-Plotting\\2011_US_AGRI_Exports.csv")
df.head()

data = dict (
    type  = 'choropleth',
    colorscale = 'ylorrd',
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
iplot(choromap2)  ## creates interactive plot - pop up a new chrome window
choromap2.write_html("USA Agricultural Exports Map.html")

# ---- International

df = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\09-Geographical-Plotting\\2014_World_GDP.csv")
df.head()

data =  dict(
    type = 'choropleth',
    locations = df['CODE'],
    z = df['GDP (BILLIONS)'],
    text = df['COUNTRY'],
    colorbar = {'title': 'GDP Billions US'}
)

layout = dict (
    title = '2014 Global GDP',
    geo = dict(
        showframe = False,
        projection = {'type':'mercator'}
    )
)

choromap3 = go.Figure( data = [data], layout = layout)
iplot(choromap3)
choromap3.write_html("International GDP Map.html")