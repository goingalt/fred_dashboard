# Dash-based FRED dashboard with 4 quadrants and dynamic keyword search
import warnings
import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime

import os
FRED_API_KEY = os.environ.get('FRED_API_KEY', None)
# Suppress the specific deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, message='.*parsing.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*parsing.*')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server



FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Default series
DEFAULT_GDP = ['GDP', 'CANGDPRQDSMEI', 'DEUGDPRQDSMEI', 'FRAGDPRQDSMEI', 'ITAGDPRQDSMEI', 'GBRGDPRQDSMEI', 'JPNGDPRQDSMEI']
DEFAULT_CPI = ['CPIAUCSL', 'CANCPIALLMINMEI', 'DEUCPIALLMINMEI', 'FRACPIALLMINMEI', 'ITACPIALLMINMEI', 'GBRCPIALLMINMEI', 'JPNCPIALLMINMEI']
DEFAULT_RATE = ['FEDFUNDS', 'CANINTDSR', 'IR3TIB01DEM156N', 'IR3TIB01FRM156N', 'IR3TIB01ITM156N', 'IR3TIB01GBM156N', 'IR3TIB01JPM156N']
DEFAULT_OIL = ['DCOILWTICO']

# Layout
app.layout = html.Div([
    dbc.Container([
        html.H2("FRED Quadrant Dashboard with Keyword Search"),

        # 2x2 Graph Grid
        dbc.Row([
            dbc.Col(dcc.Graph(id='graph-1'), width=6),
            dbc.Col(dcc.Graph(id='graph-2'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='graph-3'), width=6),
            dbc.Col(dcc.Graph(id='graph-4'), width=6)
        ]),

        dcc.Store(id='active-graph'),

        # Modal
        dbc.Modal([
            dbc.ModalHeader("Select FRED Series"),
            dbc.ModalBody([
                dbc.Label("Keyword Search"),
                dcc.Input(id='keyword-input', type='text', debounce=True, placeholder="e.g. inflation"),
                html.Br(), html.Br(),
                dbc.Label("Matching Series"),
                dcc.Dropdown(id='series-dropdown'),
                html.Br(),
                dbc.Label("Date Range"),
                dcc.DatePickerRange(id='date-picker', display_format='Y-MM-DD')
            ]),
            dbc.ModalFooter([
                dbc.Button("Update", id='update-button', color='primary'),
                dbc.Button("Close", id='close-button', color='secondary')
            ])
        ], id='config-modal', is_open=False, size="lg", style={"maxWidth": "800px"})
    ])
])

# Series ID to Country mapping
series_id_to_country = {
    'GDP': 'United States',
    'CANGDPRQDSMEI': 'Canada',
    'DEUGDPRQDSMEI': 'Germany',
    'FRAGDPRQDSMEI': 'France',
    'ITAGDPRQDSMEI': 'Italy',
    'GBRGDPRQDSMEI': 'United Kingdom',
    'JPNGDPRQDSMEI': 'Japan',
    'CPIAUCSL': 'United States',
    'CANCPIALLMINMEI': 'Canada',
    'DEUCPIALLMINMEI': 'Germany',
    'FRACPIALLMINMEI': 'France',
    'ITACPIALLMINMEI': 'Italy',
    'GBRCPIALLMINMEI': 'United Kingdom',
    'JPNCPIALLMINMEI': 'Japan',
    'FEDFUNDS': 'United States',
    'CANINTDSR': 'Canada',
    'IR3TIB01DEM156N': 'Germany',
    'IR3TIB01FRM156N': 'France',
    'IR3TIB01ITM156N': 'Italy',
    'IR3TIB01GBM156N': 'United Kingdom',
    'IR3TIB01JPM156N': 'Japan',
    'DCOILWTICO': 'Oil (WTI)'
}

def fetch_series(series_id, start='2000-01-01', end=None):
    if end is None:
        meta_url = f"{FRED_BASE_URL}/series"
        meta_params = {
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'series_id': series_id
        }
        try:
            meta_resp = requests.get(meta_url, params=meta_params)
            if meta_resp.status_code == 200:
                series_info = meta_resp.json().get('seriess', [{}])[0]
                end = series_info.get('observation_end', pd.to_datetime('today').strftime('%Y-%m-%d'))
            else:
                end = pd.to_datetime('today').strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Failed to fetch metadata for {series_id}: {e}")
            end = pd.to_datetime('today').strftime('%Y-%m-%d')

    url = f"{FRED_BASE_URL}/series/observations"
    params = {
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'series_id': series_id,
        'observation_start': start,
        'observation_end': end
    }
    try:
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print(f"Error: HTTP {r.status_code} for {series_id}")
            return pd.DataFrame()
        data = r.json().get('observations', [])
        if not data:
            print(f"No observations found for {series_id}")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'value' not in df.columns:
            print(f"'value' column missing in series {series_id}")
            return pd.DataFrame()
        # More robust date parsing to avoid deprecation warnings
        if 'date' in df.columns:
            # FRED API returns dates in YYYY-MM-DD format, so we can be explicit
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            
            # If explicit format fails, try without format specification
            if df['date'].isna().all():
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.dropna(subset=['value'])
    except Exception as e:
        print(f"Exception fetching {series_id}: {e}")
        return pd.DataFrame()

def build_multi_series_chart(series_ids, title):
    fig = go.Figure()
    for sid in series_ids:
        df = fetch_series(sid)
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df['date'], 
                y=df['value'], 
                mode='lines', 
                name=series_id_to_country.get(sid, sid)
            ))
    fig.update_layout(title=title, yaxis_title='Value')
    return fig

# Click Graph -> Open Modal
for i in range(1, 5):
    @app.callback(
        Output('config-modal', 'is_open', allow_duplicate=True),
        Output('active-graph', 'data', allow_duplicate=True),
        Output('series-dropdown', 'value', allow_duplicate=True),
        Output('date-picker', 'start_date', allow_duplicate=True),
        Output('date-picker', 'end_date', allow_duplicate=True),
        Output('date-picker', 'max_date_allowed', allow_duplicate=True),
        Input(f'graph-{i}', 'clickData'),
        State('config-modal', 'is_open'),
        prevent_initial_call=True
    )
    def show_modal(clickData, is_open):
        if clickData and not is_open:
            active_graph_id = ctx.triggered_id
            default_start = '2000-01-01'
            default_end = pd.to_datetime('today').strftime('%Y-%m-%d')
            max_date = default_end
            return True, active_graph_id, None, default_start, default_end, max_date
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Close Modal
@app.callback(
    Output('config-modal', 'is_open', allow_duplicate=True),
    Input('close-button', 'n_clicks'),
    prevent_initial_call=True
)
def close_modal(n):
    if n:
        return False
    return dash.no_update

# Search FRED Series
@app.callback(
    Output('series-dropdown', 'options'),
    Input('keyword-input', 'value')
)
def search_fred_series(keyword):
    if not keyword:
        return []
    
    url = f"{FRED_BASE_URL}/series/search"
    params = {
        'api_key': FRED_API_KEY,
        'search_text': keyword,
        'file_type': 'json',
        'limit': 20
    }
    
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200:
            results = r.json().get('seriess', [])
            return [{'label': f"{s['title']} ({s['id']})", 'value': s['id']} for s in results]
        else:
            return []
    except Exception as e:
        print(f"Error searching series: {e}")
        return []

# Update Graph
@app.callback(
    Output('graph-1', 'figure'),
    Output('graph-2', 'figure'),
    Output('graph-3', 'figure'),
    Output('graph-4', 'figure'),
    Input('update-button', 'n_clicks'),
    State('active-graph', 'data'),
    State('series-dropdown', 'value'),
    State('date-picker', 'start_date'),
    State('date-picker', 'end_date'),
    prevent_initial_call=True
)
def update_graph(n_clicks, active, series_id, start, end):
    if not n_clicks or not series_id or not active:
        return [dash.no_update] * 4
    
    df = fetch_series(series_id, start, end)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {series_id}", yaxis_title='Value')
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines'))
        fig.update_layout(title=f"{series_id}", yaxis_title='Value')

    # Return the updated figure for the active graph, no_update for others
    outputs = [dash.no_update] * 4
    graph_index = int(active.split('-')[1]) - 1
    outputs[graph_index] = fig
    return outputs

# Load Default Charts
@app.callback(
    Output('graph-1', 'figure', allow_duplicate=True),
    Output('graph-2', 'figure', allow_duplicate=True),
    Output('graph-3', 'figure', allow_duplicate=True),
    Output('graph-4', 'figure', allow_duplicate=True),
    Input(app.layout, 'children'),
    prevent_initial_call='initial_duplicate'
)
def load_defaults(_):
    fig1 = build_multi_series_chart(DEFAULT_GDP, "G7 GDP Growth")
    fig2 = build_multi_series_chart(DEFAULT_CPI, "G7 Inflation Rates")
    fig3 = build_multi_series_chart(DEFAULT_RATE, "G7 Policy Rates")
    fig4 = build_multi_series_chart(DEFAULT_OIL, "Oil Price per Barrel")
    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run(debug=True)
