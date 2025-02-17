from flask import Flask, render_template, request
import math, statistics, json
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
def d1(S, K, t, r, v):
    return (math.log(S / K) + ((r + (v**2 / 2)) * t)) / (v * math.sqrt(t))

def d2(S, K, t, r, v):
    return d1(S, K, t, r, v) - v * math.sqrt(t)

def black_scholes_call(S, K, t, r, v):
    d1_val = d1(S, K, t, r, v)
    d2_val = d2(S, K, t, r, v)
    return S * statistics.NormalDist().cdf(d1_val) - K * math.exp(-r * t) * statistics.NormalDist().cdf(d2_val)

def black_scholes_put(S, K, t, r, v):
    d1_val = d1(S, K, t, r, v)
    d2_val = d2(S, K, t, r, v)
    return K * math.exp(-r * t) * statistics.NormalDist().cdf(-d2_val) - S * statistics.NormalDist().cdf(-d1_val)

def calculate_greeks(S, K, t, r, v, option_type='call'):
    d1_val = d1(S, K, t, r, v)
    d2_val = d2(S, K, t, r, v)
    
    # Common calculations
    pdf_d1 = norm.pdf(d1_val)
    pdf_d2 = norm.pdf(d2_val)
    cdf_d1 = norm.cdf(d1_val)
    cdf_d2 = norm.cdf(d2_val)
    
    # First-order Greeks
    if option_type == 'call':
        delta = cdf_d1
        theta = (-S * pdf_d1 * v / (2 * math.sqrt(t)) - 
                r * K * math.exp(-r * t) * cdf_d2)
    else:
        delta = cdf_d1 - 1
        theta = (-S * pdf_d1 * v / (2 * math.sqrt(t)) + 
                r * K * math.exp(-r * t) * (1 - cdf_d2))
    
    # Second-order Greeks
    gamma = pdf_d1 / (S * v * math.sqrt(t))
    vega = S * math.sqrt(t) * pdf_d1
    
    # Third-order Greeks
    vanna = -pdf_d1 * d2_val / v  # Vanna (dDelta/dVol)
    charm = -pdf_d1 * (r / (v * math.sqrt(t)) - d2_val / (2 * t))  # Charm (dDelta/dTime)
    speed = -gamma * (d1_val / (v * math.sqrt(t)) + 1) / S  # Speed (d²Gamma/dS²)
    color = -gamma * (r + v**2 * (2 * d1_val * d2_val - d1_val**2) / (4 * t)) / (2 * t)  # Color (dGamma/dTime)
    volga = vega * d1_val * d2_val / v  # Volga (d²Vega/dVol²)
    zomma = gamma * (d1_val * d2_val - 1) / v  # Zomma (d²Delta/dVol²)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'vanna': vanna,
        'charm': charm,
        'speed': speed,
        'color': color,
        'volga': volga,
        'zomma': zomma
    }

def calculate_strategy_prices(strategy, spot_range, K1, K2=None, K3=None, K4=None, t=1.0, r=0.05, v=0.2):
    prices = []
    for s in spot_range:
        if strategy == "Long Butterfly":
            # Buy 1 call at K1, sell 2 calls at K2, buy 1 call at K3
            p1 = black_scholes_call(s, K1, t, r, v)
            p2 = 2 * black_scholes_call(s, K2, t, r, v)
            p3 = black_scholes_call(s, K3, t, r, v)
            prices.append(p1 - p2 + p3)
        elif strategy == "Iron Condor":
            # Buy put at K1, sell put at K2, sell call at K3, buy call at K4
            p1 = black_scholes_put(s, K1, t, r, v)
            p2 = black_scholes_put(s, K2, t, r, v)
            p3 = black_scholes_call(s, K3, t, r, v)
            p4 = black_scholes_call(s, K4, t, r, v)
            prices.append(p1 - p2 - p3 + p4)
        elif strategy == "Box Spread":
            # Buy call at K1, sell put at K1, sell call at K2, buy put at K2
            c1 = black_scholes_call(s, K1, t, r, v)
            p1 = black_scholes_put(s, K1, t, r, v)
            c2 = black_scholes_call(s, K2, t, r, v)
            p2 = black_scholes_put(s, K2, t, r, v)
            prices.append(c1 - p1 - c2 + p2)
    return prices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about_me.html')

@app.route('/options', methods=['GET', 'POST'])
def options():
    # Default parameters
    spot = 100.0
    strike = 100.0
    time = 1.0
    interest = 0.05
    volatility = 0.2
    min_spot = spot * 0.5
    max_spot = spot * 1.5
    min_vol = volatility * 0.5
    max_vol = round(volatility * 1.5, 2)
    strategy = "Long Straddle"
    num_simulations = 10000
    num_steps = 252

    if request.method == 'POST':
        try:
            spot = float(request.form.get('spot', spot))
            strike = float(request.form.get('strike', strike))
            time = float(request.form.get('time', time))
            interest = float(request.form.get('interest', interest))
            volatility = float(request.form.get('volatility', volatility))
            min_spot = float(request.form.get('min_spot', min_spot))
            max_spot = float(request.form.get('max_spot', max_spot))
            min_vol = float(request.form.get('min_vol', min_vol))
            max_vol = float(request.form.get('max_vol', max_vol))
            strategy = request.form.get('strategy', strategy)
            num_simulations = int(request.form.get('num_simulations', num_simulations))
            num_steps = int(request.form.get('num_steps', num_steps))
        except Exception as e:
            print("Error parsing form values:", e)

    # Option pricing
    call = black_scholes_call(spot, strike, time, interest, volatility)
    put = black_scholes_put(spot, strike, time, interest, volatility)

    # Calculate all Greeks for both call and put options
    call_greeks = calculate_greeks(spot, strike, time, interest, volatility, 'call')
    put_greeks = calculate_greeks(spot, strike, time, interest, volatility, 'put')

    # Create Greek visualization charts
    time_range = np.linspace(0.1, 2, 100)
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 100)
    vol_range = np.linspace(volatility * 0.5, volatility * 1.5, 100)

    # Delta vs Spot Price
    delta_vs_spot = go.Figure()
    delta_vs_spot.add_trace(go.Scatter(x=spot_range, 
                                      y=[calculate_greeks(s, strike, time, interest, volatility, 'call')['delta'] for s in spot_range],
                                      name='Call Delta'))
    delta_vs_spot.add_trace(go.Scatter(x=spot_range, 
                                      y=[calculate_greeks(s, strike, time, interest, volatility, 'put')['delta'] for s in spot_range],
                                      name='Put Delta'))
    delta_vs_spot.update_layout(title='Delta vs Spot Price',
                               xaxis_title='Spot Price',
                               yaxis_title='Delta')

    # Theta vs Time
    theta_vs_time = go.Figure()
    theta_vs_time.add_trace(go.Scatter(x=time_range,
                                      y=[calculate_greeks(spot, strike, t, interest, volatility, 'call')['theta'] for t in time_range],
                                      name='Call Theta'))
    theta_vs_time.add_trace(go.Scatter(x=time_range,
                                      y=[calculate_greeks(spot, strike, t, interest, volatility, 'put')['theta'] for t in time_range],
                                      name='Put Theta'))
    theta_vs_time.update_layout(title='Theta vs Time',
                               xaxis_title='Time to Expiry',
                               yaxis_title='Theta')

    # Vega vs Volatility
    vega_vs_vol = go.Figure()
    vega_vs_vol.add_trace(go.Scatter(x=vol_range,
                                    y=[calculate_greeks(spot, strike, time, interest, v, 'call')['vega'] for v in vol_range],
                                    name='Vega'))
    vega_vs_vol.update_layout(title='Vega vs Volatility',
                             xaxis_title='Volatility',
                             yaxis_title='Vega')

    # Higher-order Greeks visualization
    vanna_vs_spot = go.Figure()
    vanna_vs_spot.add_trace(go.Scatter(x=spot_range,
                                      y=[calculate_greeks(s, strike, time, interest, volatility, 'call')['vanna'] for s in spot_range],
                                      name='Vanna'))
    vanna_vs_spot.update_layout(title='Vanna vs Spot Price',
                               xaxis_title='Spot Price',
                               yaxis_title='Vanna')
    

    # 3D Surface Plots for Call and Put options
    call_surface = None
    put_surface = None

    # --- Option Price Surfaces ---
    if min_spot < max_spot and min_vol < max_vol:
        spot_range = np.linspace(min_spot, max_spot, 50)
        vol_range = np.linspace(min_vol, max_vol, 50)
        spot_mesh, vol_mesh = np.meshgrid(spot_range, vol_range)
        vectorized_call = np.vectorize(lambda s, v: black_scholes_call(s, strike, time, interest, v))
        vectorized_put = np.vectorize(lambda s, v: black_scholes_put(s, strike, time, interest, v))
        # Convert to lists so Plotly can serialize them properly.
        call_prices = vectorized_call(spot_mesh, vol_mesh).tolist()
        put_prices = vectorized_put(spot_mesh, vol_mesh).tolist()
        
        fig_call = go.Figure(data=[go.Surface(z=call_prices, 
                                               x=spot_mesh.tolist(), 
                                               y=vol_mesh.tolist())])
        fig_call.update_layout(autosize=False,
                               width=600,
                               height=500,
                               margin=dict(l=0, r=0, b=0, t=40),
                               scene=dict(aspectmode='auto',
                                          xaxis_title='Spot Price',
                                          yaxis_title='Volatility',
                                          zaxis_title='Option Price'))
        call_surface = fig_call.to_dict()
        
        fig_put = go.Figure(data=[go.Surface(z=put_prices, 
                                              x=spot_mesh.tolist(), 
                                              y=vol_mesh.tolist())])
        fig_put.update_layout(autosize=False,
                              width=600,
                              height=500,
                              margin=dict(l=0, r=0, b=0, t=40),
                              scene=dict(aspectmode='auto',
                                         xaxis_title='Spot Price',
                                         yaxis_title='Volatility',
                                         zaxis_title='Option Price'))
        put_surface = fig_put.to_dict()

    # Profit/Loss Diagram for Option Strategies
    spot_range_line = np.linspace(spot * 0.5, spot * 1.5, 100)
    if strategy == "Long Straddle":
        pnl = [max(s - strike, 0) + max(strike - s, 0) - call - put for s in spot_range_line]
    elif strategy == "Bull Call Spread":
        short_strike = strike * 1.1
        long_call = black_scholes_call(spot, strike, time, interest, volatility)
        short_call = black_scholes_call(spot, short_strike, time, interest, volatility)
        strategy_cost = long_call - short_call
        pnl = [max(0, min(s - strike, short_strike - strike)) - strategy_cost for s in spot_range_line]
    elif strategy == "Bear Put Spread":
        long_strike = strike * 0.9
        long_put = black_scholes_put(spot, long_strike, time, interest, volatility)
        short_put = black_scholes_put(spot, strike, time, interest, volatility)
        strategy_cost = long_put - short_put
        pnl = [max(0, min(strike - s, strike - long_strike)) + strategy_cost for s in spot_range_line]
    elif strategy == "Long Strangle":
        low_strike = strike * 0.9
        high_strike = low_strike * 1.25
        long_call = black_scholes_call(spot, high_strike, time, interest, volatility)
        long_put = black_scholes_put(spot, low_strike, time, interest, volatility)
        strategy_cost = long_call + long_put
        pnl = [max(0, s - high_strike) + max(0, low_strike - s) - strategy_cost for s in spot_range_line]
    elif strategy == "Long Butterfly":
        k1 = strike * 0.9
        k2 = strike
        k3 = strike * 1.1
        pnl = calculate_strategy_prices("Long Butterfly", spot_range_line, k1, k2, k3, t=time, r=interest, v=volatility)
    elif strategy == "Iron Condor":
        k1 = strike * 0.8
        k2 = strike * 0.9
        k3 = strike * 1.1
        k4 = strike * 1.2
        pnl = calculate_strategy_prices("Iron Condor", spot_range_line, k1, k2, k3, k4, t=time, r=interest, v=volatility)
    elif strategy == "Box Spread":
        k1 = strike
        k2 = strike * 1.1
        pnl = calculate_strategy_prices("Box Spread", spot_range_line, k1, k2, t=time, r=interest, v=volatility)
    else:
        pnl = [0 for s in spot_range_line]

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=spot_range_line, y=pnl, mode='lines', name='Profit/Loss'))
    fig_pnl.update_layout(xaxis_title='Spot Price at Expiration',
                          yaxis_title='Profit/Loss')
    pnl_chart = fig_pnl.to_dict()

    # --- Monte Carlo Simulation and Histogram ---
    dt = time / num_steps
    simulation_data = np.exp((interest - 0.5 * volatility**2) * dt +
                             volatility * np.random.normal(0, np.sqrt(dt), size=(num_simulations, num_steps)).T)
    simulation_data = np.vstack([np.ones(num_simulations), simulation_data])
    simulation_data = spot * simulation_data.cumprod(axis=0)
    
    final_prices = simulation_data[-1]

    call_payoffs = np.maximum(final_prices - strike, 0)
    put_payoffs = np.maximum(strike - final_prices, 0)
    simulated_call_price = np.mean(call_payoffs) * np.exp(-interest * time)
    simulated_put_price = np.mean(put_payoffs) * np.exp(-interest * time)

    
    fig_mc = go.Figure()
    for i in range(min(100, num_simulations)):
        # Convert each trace to a list
        fig_mc.add_trace(go.Scatter(y=simulation_data[:, i].tolist(),
                                    mode='lines',
                                    line=dict(width=1),
                                    showlegend=False))
    fig_mc.update_layout(autosize=True,
                         xaxis_title='Time Steps',
                         yaxis_title='Asset Price')
    mc_chart = fig_mc.to_dict()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=final_prices.tolist(),
                                    nbinsx=50,
                                    name='Final Prices'))
    fig_hist.update_layout(autosize=True,
                           xaxis_title='Asset Price',
                           yaxis_title='Frequency')
    hist_chart = fig_hist.to_dict()

    return render_template('options_visualizer.html',
                           spot=spot,
                           strike=strike,
                           time=time,
                           interest=interest,
                           volatility=volatility,
                           call=call,
                           put=put,
                           call_greeks=call_greeks,
                           put_greeks=put_greeks,
                           call_surface=call_surface,
                           put_surface=put_surface,
                           strategy=strategy,
                           pnl_chart=pnl_chart,
                           min_spot=min_spot,
                           max_spot=max_spot,
                           min_vol=min_vol,
                           max_vol=max_vol,
                           num_simulations=num_simulations,
                           num_steps=num_steps,
                           simulated_call_price=simulated_call_price,
                           simulated_put_price=simulated_put_price,
                           mc_chart=mc_chart,
                           hist_chart=hist_chart,
                           delta_vs_spot=delta_vs_spot.to_dict(),
                           theta_vs_time=theta_vs_time.to_dict(),
                           vega_vs_vol=vega_vs_vol.to_dict(),
                           vanna_vs_spot=vanna_vs_spot.to_dict())

@app.route('/projects')
def projects():
    projects = [
        {
            "name": "Project 1: Options Pricer",
            "link": "https://global-options-pricing.streamlit.app",
            "description": "Options Pricer is a Flask app for pricing financial options using a variety of models. It demonstrates calculations with Black-Scholes and more."
        },
        {
            "name": "Project 2: Teach Me About",
            "link": "https://github.com/ramankc6/teach-me-about/tree/main",
            "description": "Teach Me About provides educational resources on various topics with interactive content."
        },
        {
            "name": "Project 3: IMC Prosperity",
            "link": "https://github.com/imc-prosperity",
            "description": "IMC Prosperity is a platform for managing personal finances with budgeting, expense tracking, and goal setting tools."
        }
    ]
    return render_template('projects.html', projects=projects)

@app.route('/resume')
def resume():
    return render_template('resume.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50708, debug=True)
