import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Create a random DataFrame
df = pd.DataFrame(np.random.randn(100, 4), columns='A B C D'.split())

# Create a categorical DataFrame
df2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [32, 43, 50]})

# Plot line chart of random data
fig1 = px.line(df, title='Line Plot of Random Data')
fig1.write_html("line_plot.html", auto_open=True)

# Plot bar chart of categorical data
fig2 = px.bar(df2, x='Category', y='Values', title='Bar Chart of Categories')
fig2.write_html("bar_chart.html", auto_open=True)

