from flask import Flask, render_template, request
import math, statistics, json
import numpy as np
import plotly
import plotly.graph_objects as go

app = Flask(__name__)

def black_scholes_call(S, K, t, r, v):
    d1 = (math.log(S / K) + ((r + (v**2 / 2)) * t)) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return S * statistics.NormalDist().cdf(d1) - K * math.exp(-r * t) * statistics.NormalDist().cdf(d2)

def black_scholes_put(S, K, t, r, v):
    d1 = (math.log(S / K) + ((r + (v**2 / 2)) * t)) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return K * math.exp(-r * t) * statistics.NormalDist().cdf(-d2) - S * statistics.NormalDist().cdf(-d1)

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

    # Greeks
    delta_call = (black_scholes_call(spot * 1.01, strike, time, interest, volatility) - call) / (spot * 0.01)
    delta_put = (black_scholes_put(spot * 1.01, strike, time, interest, volatility) - put) / (spot * 0.01)
    gamma = ((black_scholes_call(spot * 1.01, strike, time, interest, volatility) -
              2 * call +
              black_scholes_call(spot * 0.99, strike, time, interest, volatility))
             / ((0.01 * spot) ** 2)) / 100
    theta_call = (black_scholes_call(spot, strike, time - 1/365, interest, volatility) - call) / (1/365)
    theta_put = (black_scholes_put(spot, strike, time - 1/365, interest, volatility) - put) / (1/365)
    vega = (black_scholes_call(spot, strike, time, interest, volatility + 0.01) - call) / 0.01

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
                           delta_call=delta_call,
                           delta_put=delta_put,
                           gamma=gamma,
                           theta_call=theta_call,
                           theta_put=theta_put,
                           vega=vega,
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
                           hist_chart=hist_chart)

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
    app.run(debug=True)
