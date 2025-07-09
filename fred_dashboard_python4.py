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

# Get FRED API key from environment
FRED_API_KEY = os.environ.get('FRED_API_KEY', None)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*parsing.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*parsing.*')

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# FRED API base URL
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Default series for each quadrant
DEFAULT_GDP = ['GDP', 'CANGDPRQDSMEI', 'DEUGDPRQDSMEI', 'FRAGDPRQDSMEI', 'ITAGDPRQDSMEI', 'GBRGDPRQDSMEI', 'JPNGDPRQDSMEI']
DEFAULT_CPI = ['CPIAUCSL', 'CANCPIALLMINMEI', 'DEUCPIALLMINMEI', 'FRACPIALLMINMEI', 'ITACPIALLMINMEI', 'GBRCPIALLMINMEI', 'JPNCPIALLMINMEI']
DEFAULT_RATE = ['FEDFUNDS', 'CANINTDSR', 'IR3TIB01DEM156N', 'IR3TIB01FRM156N', 'IR3TIB01ITM156N', 'IR3TIB01GBM156N', 'IR3TIB01JPM156N']
DEFAULT_OIL = ['DCOILWTICO']

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
    """Fetch series data from FRED API"""
    if not FRED_API_KEY:
        print("Warning: FRED_API_KEY not set. Using sample data.")
        # Return sample data for demo purposes
        dates = pd.date_range(start=start, end=end or '2024-01-01', freq='M')
        values = [100 + i + (i % 10) * 5 for i in range(len(dates))]
        return pd.DataFrame({'date': dates, 'value': values})
    
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
        
        # Parse dates with explicit format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            if df['date'].isna().all():
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.dropna(subset=['value'])
    except Exception as e:
        print(f"Exception fetching {series_id}: {e}")
        return pd.DataFrame()

def build_multi_series_chart(series_ids, title):
    """Build a chart with multiple series"""
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
    fig.update_layout(
        title=title, 
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

def search_fred_series(keyword):
    """Search for FRED series by keyword"""
    if not keyword or not FRED_API_KEY:
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

# App Layout
app.layout = html.Div([
    dbc.Container([
        html.H2("FRED Quadrant Dashboard with Keyword Search", className="text-center mb-4"),
        
        # API Key warning
        html.Div(id="api-warning", style={'display': 'none' if FRED_API_KEY else 'block'}),
        
        # 2x2 Graph Grid
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='graph-1', style={'height': '400px'})
            ], width=6),
            dbc.Col([
                dcc.Graph(id='graph-2', style={'height': '400px'})
            ], width=6)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='graph-3', style={'height': '400px'})
            ], width=6),
            dbc.Col([
                dcc.Graph(id='graph-4', style={'height': '400px'})
            ], width=6)
        ]),

        # Store for active graph
        dcc.Store(id='active-graph'),

        # Configuration Modal
        dbc.Modal([
            dbc.ModalHeader("Select FRED Series"),
            dbc.ModalBody([
                dbc.Label("Keyword Search"),
                dcc.Input(
                    id='keyword-input', 
                    type='text', 
                    debounce=True, 
                    placeholder="e.g. inflation, unemployment, GDP"
                ),
                html.Br(), html.Br(),
                dbc.Label("Matching Series"),
                dcc.Dropdown(id='series-dropdown', placeholder="Select a series"),
                html.Br(),
                dbc.Label("Date Range"),
                dcc.DatePickerRange(
                    id='date-picker', 
                    display_format='YYYY-MM-DD',
                    start_date='2000-01-01',
                    end_date=pd.to_datetime('today').strftime('%Y-%m-%d')
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Update Graph", id='update-button', color='primary'),
                dbc.Button("Close", id='close-button', color='secondary')
            ])
        ], id='config-modal', is_open=False, size="lg")
    ], fluid=True)
])

# Show API warning if no key
@app.callback(
    Output('api-warning', 'children'),
    Input('api-warning', 'id')
)
def show_api_warning(_):
    if not FRED_API_KEY:
        return dbc.Alert([
            html.H4("FRED API Key Required", className="alert-heading"),
            html.P("To use live data, set the FRED_API_KEY environment variable. "),
            html.P("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"),
            html.P("Currently showing sample data for demonstration.")
        ], color="warning", dismissable=True)
    return ""

# Search FRED Series
@app.callback(
    Output('series-dropdown', 'options'),
    Input('keyword-input', 'value')
)
def update_series_dropdown(keyword):
    return search_fred_series(keyword)

# Show modal when graph is clicked
@app.callback(
    Output('config-modal', 'is_open'),
    Output('active-graph', 'data'),
    [Input(f'graph-{i}', 'clickData') for i in range(1, 5)] + [Input('close-button', 'n_clicks')],
    State('config-modal', 'is_open'),
    prevent_initial_call=True
)
def toggle_modal(*args):
    # Get the context to see which input triggered the callback
    trigger_id = ctx.triggered_id
    is_open = args[-1]  # Last argument is the state
    
    # If close button was clicked, close modal
    if trigger_id == 'close-button':
        return False, dash.no_update
    
    # If a graph was clicked and modal is not open, open it
    if trigger_id and trigger_id.startswith('graph-') and not is_open:
        return True, trigger_id
    
    return is_open, dash.no_update

# Update specific graph
@app.callback(
    [Output(f'graph-{i}', 'figure') for i in range(1, 5)],
    Input('update-button', 'n_clicks'),
    State('active-graph', 'data'),
    State('series-dropdown', 'value'),
    State('date-picker', 'start_date'),
    State('date-picker', 'end_date'),
    prevent_initial_call=True
)
def update_selected_graph(n_clicks, active_graph, series_id, start_date, end_date):
    if not n_clicks or not series_id or not active_graph:
        return [dash.no_update] * 4
    
    # Fetch data for selected series
    df = fetch_series(series_id, start_date, end_date)
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {series_id}",
            template='plotly_white'
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['value'], 
            mode='lines',
            name=series_id
        ))
        fig.update_layout(
            title=f"{series_id}",
            yaxis_title='Value',
            template='plotly_white'
        )

    # Update only the active graph
    outputs = [dash.no_update] * 4
    if active_graph:
        graph_index = int(active_graph.split('-')[1]) - 1
        outputs[graph_index] = fig
    
    return outputs

# Load default charts on startup
@app.callback(
    [Output(f'graph-{i}', 'figure', allow_duplicate=True) for i in range(1, 5)],
    Input('api-warning', 'id'),
    prevent_initial_call='initial_duplicate'
)
def load_default_charts(_):
    fig1 = build_multi_series_chart(DEFAULT_GDP, "G7 GDP Growth")
    fig2 = build_multi_series_chart(DEFAULT_CPI, "G7 Inflation Rates")
    fig3 = build_multi_series_chart(DEFAULT_RATE, "G7 Policy Rates")
    fig4 = build_multi_series_chart(DEFAULT_OIL, "Oil Price (WTI)")
    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run(debug=True)
