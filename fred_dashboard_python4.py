# Dash-based FRED dashboard with 4 quadrants, dynamic keyword search, and data tables
import warnings
import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context, dash_table
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
DEFAULT_GDP = ['GDPC1', 'NGDPRSAXDCCAQ', 'CLVMNACSCAB1GQDE', 'CLVMNACSCAB1GQFR', 'CLVMNACSCAB1GQIT', 'GBRGDPRQDSMEI', 'JPNRGDPEXP']
DEFAULT_CPI = ['CPIAUCSL', 'CANCPIALLMINMEI', 'DEUCPIALLMINMEI', 'FRACPIALLMINMEI', 'ITACPIALLMINMEI', 'GBRCPIALLMINMEI', 'JPNCPIALLMINMEI']
DEFAULT_RATE = ['FEDFUNDS', 'CANINTDSR', 'IR3TIB01DEM156N', 'IR3TIB01FRM156N', 'IR3TIB01ITM156N', 'IR3TIB01GBM156N', 'IR3TIB01JPM156N']
DEFAULT_OIL = ['DCOILWTICO']

# Series ID to Country mapping
series_id_to_country = {
    'GDPC1': 'United States',
    'NGDPRSAXDCCAQ': 'Canada',
    'CLVMNACSCAB1GQDE': 'Germany',
    'CLVMNACSCAB1GQFR': 'France',
    'CLVMNACSCAB1GQIT': 'Italy',
    'GBRGDPRQDSMEI': 'United Kingdom',
    'JPNRGDPEXP': 'Japan',
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

# Store for graph data
graph_data_store = {}

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

def build_multi_series_chart(series_ids, title, calculate_pct_change=False):
    """Build a chart with multiple series and return both figure and data"""
    fig = go.Figure()
    combined_data = []
    
    for sid in series_ids:
        df = fetch_series(sid)
        if not df.empty:
            country_name = series_id_to_country.get(sid, sid)
                        # Calculate percentage change if requested
            if calculate_pct_change:
                df = df.sort_values('date')
                df['pct_change'] = df['value'].pct_change(periods=4) * 100
                df = df.dropna(subset=['pct_change'])
                
                fig.add_trace(go.Scatter(
                    x=df['date'], 
                    y=df['pct_change'], 
                    mode='lines', 
                    name=country_name
                ))
                
                # Prepare data for table
                df_table = df.copy()
                df_table['series'] = country_name
                df_table['date'] = df_table['date'].dt.strftime('%Y-%m-%d')
                df_table['value'] = df_table['pct_change']
                combined_data.append(df_table[['date', 'series', 'value']])
            else:
                fig.add_trace(go.Scatter(
                    x=df['date'], 
                    y=df['value'], 
                    mode='lines', 
                    name=country_name
                ))
                
                # Prepare data for table
                df_table = df.copy()
                df_table['series'] = country_name
                df_table['date'] = df_table['date'].dt.strftime('%Y-%m-%d')
                combined_data.append(df_table[['date', 'series', 'value']])
    
   # Update y-axis title based on whether we're showing percentage change
    y_title = '% Change (4 Quarters Ago)' if calculate_pct_change else 'Value'
    
    fig.update_layout(
        title=title, 
        yaxis_title=y_title,
        hovermode='x unified',
        template='plotly_white'
    )
    # Combine all data for table
    if combined_data:
        table_data = pd.concat(combined_data, ignore_index=True)
        table_data = table_data.sort_values(['date', 'series'])
    else:
        table_data = pd.DataFrame(columns=['date', 'series', 'value'])
    
    return fig, table_data

def create_data_table(data, table_id):
    """Create a data table component"""
    if data.empty:
        return html.Div("No data available", className="text-center text-muted")
    
    return dash_table.DataTable(
        id=table_id,
        data=data.to_dict('records'),
        columns=[
            {'name': 'Date', 'id': 'date'},
            {'name': 'Series', 'id': 'series'},
            {'name': 'Value', 'id': 'value', 'type': 'numeric', 'format': {'specifier': '.2f'}}
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': '12px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        page_size=10,
        sort_action="native",
        filter_action="native",
        export_format="csv",
        style_table={'overflowX': 'auto'}
    )

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
        html.H2("FRED Quadrant Dashboard with Data Tables", className="text-center mb-4"),
        
        # API Key warning
        html.Div(id="api-warning", style={'display': 'none' if FRED_API_KEY else 'block'}),
        
        # 2x2 Graph Grid with Data Tables
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='graph-1', style={'height': '400px'}),
                html.H5("Data Table", className="mt-3 mb-2"),
                html.Div(id='table-1')
            ], width=6),
            dbc.Col([
                dcc.Graph(id='graph-2', style={'height': '400px'}),
                html.H5("Data Table", className="mt-3 mb-2"),
                html.Div(id='table-2')
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='graph-3', style={'height': '400px'}),
                html.H5("Data Table", className="mt-3 mb-2"),
                html.Div(id='table-3')
            ], width=6),
            dbc.Col([
                dcc.Graph(id='graph-4', style={'height': '400px'}),
                html.H5("Data Table", className="mt-3 mb-2"),
                html.Div(id='table-4')
            ], width=6)
        ]),

        # Store for active graph and graph data
        dcc.Store(id='active-graph'),
        dcc.Store(id='graph-data-store'),

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

# Update specific graph and table
@app.callback(
    [Output(f'graph-{i}', 'figure') for i in range(1, 5)] +
    [Output(f'table-{i}', 'children') for i in range(1, 5)] +
    [Output('graph-data-store', 'data')],
    Input('update-button', 'n_clicks'),
    State('active-graph', 'data'),
    State('series-dropdown', 'value'),
    State('date-picker', 'start_date'),
    State('date-picker', 'end_date'),
    State('graph-data-store', 'data'),
    prevent_initial_call=True
)
def update_selected_graph(n_clicks, active_graph, series_id, start_date, end_date, current_data):
    if not n_clicks or not series_id or not active_graph:
        return [dash.no_update] * 9
    
    # Initialize current_data if None
    if current_data is None:
        current_data = {}
    
    # Fetch data for selected series
    df = fetch_series(series_id, start_date, end_date)
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {series_id}",
            template='plotly_white'
        )
        table_data = pd.DataFrame(columns=['date', 'series', 'value'])
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
        
        # Prepare table data
        table_data = df.copy()
        table_data['series'] = series_id
        table_data['date'] = table_data['date'].dt.strftime('%Y-%m-%d')
        table_data = table_data[['date', 'series', 'value']]

    # Update only the active graph and table
    graph_outputs = [dash.no_update] * 4
    table_outputs = [dash.no_update] * 4
    
    if active_graph:
        graph_index = int(active_graph.split('-')[1]) - 1
        graph_outputs[graph_index] = fig
        table_outputs[graph_index] = create_data_table(table_data, f'table-{graph_index + 1}')
        
        # Store the data for this graph
        current_data[f'graph-{graph_index + 1}'] = table_data.to_dict('records')
    
    return graph_outputs + table_outputs + [current_data]

# Load default charts and tables on startup
@app.callback(
    [Output(f'graph-{i}', 'figure', allow_duplicate=True) for i in range(1, 5)] +
    [Output(f'table-{i}', 'children', allow_duplicate=True) for i in range(1, 5)] +
    [Output('graph-data-store', 'data', allow_duplicate=True)],
    Input('api-warning', 'id'),
    prevent_initial_call='initial_duplicate'
)
def load_default_charts(_):
    # Build charts and get data
    fig1, data1 = build_multi_series_chart(DEFAULT_GDP, "G7 GDP Growth (% Change vs 4 Quarters Ago)", calculate_pct_change=True)
    fig2, data2 = build_multi_series_chart(DEFAULT_CPI, "G7 Inflation Rates", calculate_pct_change=True)
    fig3, data3 = build_multi_series_chart(DEFAULT_RATE, "G7 Policy Rates")
    fig4, data4 = build_multi_series_chart(DEFAULT_OIL, "Oil Price (WTI)")
    
    # Create tables
    table1 = create_data_table(data1, 'table-1')
    table2 = create_data_table(data2, 'table-2')
    table3 = create_data_table(data3, 'table-3')
    table4 = create_data_table(data4, 'table-4')
    
    # Store data
    stored_data = {
        'graph-1': data1.to_dict('records'),
        'graph-2': data2.to_dict('records'),
        'graph-3': data3.to_dict('records'),
        'graph-4': data4.to_dict('records')
    }
    
    return [fig1, fig2, fig3, fig4] + [table1, table2, table3, table4] + [stored_data]

if __name__ == '__main__':
    app.run(debug=True)
