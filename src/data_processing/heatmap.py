import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff


def get_us_heatmap(df):
    # original_counts = df['state'].value_counts()
    # print(original_counts)
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df = df.astype(float, errors='ignore')
    df = df.reset_index(drop=True)
    new_counts = df['state'].value_counts()
    # print(new_counts)

    states = df['state'].astype(str).unique()
    cts = []
    system_counts = []
    normalized_counts = []
    for s in states:
        c = new_counts[s]
        cts.append(c)
        sc = df['solar_system_count'].where(df['state'] == s).sum(axis=0)
        system_counts.append(sc)
        normalized_counts.append(float(sc)/float(c))

    new_df = pd.DataFrame({'state': states, 'count': cts, 'systems': system_counts, 'norm': normalized_counts})
    new_df = new_df.applymap(lambda s: s.upper() if type(s) == str else s)
    print(new_df)

    color_scale = [
        [0, "#171c42"],
        [200, "#223f78"],
        [400, "#1267b2"],
        [600, "#4590c4"],
        [800, "#8cb5c9"],
        [1000, "#b6bed5"],
        [1200, "#dab2be"],
        [1400, "#d79d8b"],
        [1600, "#c46852"],
        [1800, "#a63329"],
        [2000, "#701b20"],
        [2200, "#3c0911"]
    ]

    # bounds = list(np.linspace(-100, 2600, len(color_scale) - 1))

    plt_data = [dict(
        type="choropleth",
        autocolorscale=True,
        locations=new_df['state'],
        z=new_df['norm'].astype(float),
        locationmode='USA-states',
        text=new_df['state'],
        marker=dict(
            line=dict(
                color='rgb(255,255,255)',
                width=2
            )),
        colorbar=dict(title='Average Number of Systems')
    )]

    layout = dict(title='Solar System Density per State Normalized by Number of Census Tracts',
                  geo=dict(scope="usa", showlakes=True, lakecolor='rgb(255, 255, 255)'))

    py.plot(go.Figure(data=plt_data, layout=layout), validate=False)
