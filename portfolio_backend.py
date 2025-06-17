from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Portfolio Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PortfolioRequest(BaseModel):
    symbols: List[str]
    method: str
    risk_tolerance: Optional[float] = 0.5
    investment_amount: Optional[float] = 100000
    views: Optional[Dict] = None
    monte_carlo_runs: Optional[int] = 1000

class OptimizationResult(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    method_used: str
    allocation_dollars: Dict[str, float]

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
    def fetch_data(self, symbols: List[str], period: str = "2y"):
        """Fetch historical price data for given symbols"""
        try:
            data = yf.download(symbols, period=period)['Adj Close']
            if len(symbols) == 1:
                data = data.to_frame(symbols[0])
            return data.dropna()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")
    
    def calculate_returns(self, prices):
        """Calculate daily returns"""
        return prices.pct_change().dropna()
    
    def calculate_portfolio_metrics(self, weights, returns):
        """Calculate portfolio metrics"""
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # VaR and CVaR calculation
        portfolio_returns = (returns * weights).sum(axis=1)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        
        return portfolio_return, portfolio_vol, sharpe_ratio, var_95, cvar_95
    
    def mean_variance_optimization(self, returns, risk_tolerance=0.5):
        """Traditional Mean-Variance Optimization"""
        n_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        def objective(weights):
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Risk-adjusted return based on risk tolerance
            return -(portfolio_return - risk_tolerance * portfolio_vol**2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def risk_parity_optimization(self, returns):
        """Risk Parity Optimization"""
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib)**2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(risk_parity_objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def black_litterman_optimization(self, returns, views=None):
        """Black-Litterman Model Implementation"""
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        
        # Market capitalization weights (simplified as equal weights)
        market_weights = np.array([1/n_assets] * n_assets)
        
        # Risk aversion parameter
        risk_aversion = 3.0
        
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_weights)
        
        # If no views provided, use equilibrium
        if not views:
            mu_bl = pi
        else:
            # Simplified views implementation
            # In practice, this would be more sophisticated
            tau = 0.025  # Scaling factor
            
            # Create P matrix (picking matrix) and Q vector (view returns)
            P = np.eye(n_assets)  # Simplified: views on all assets
            Q = np.array([views.get(col, 0) for col in returns.columns])
            
            # Uncertainty matrix for views
            omega = tau * np.dot(P, np.dot(cov_matrix, P.T))
            
            # Black-Litterman formula
            M1 = np.linalg.inv(tau * cov_matrix)
            M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
            M3 = np.dot(np.linalg.inv(tau * cov_matrix), pi)
            M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
            
            mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        
        # Optimize portfolio with Black-Litterman returns
        def objective(weights):
            portfolio_return = np.sum(weights * mu_bl)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_vol**2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = market_weights
        
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def monte_carlo_simulation(self, returns, weights, n_simulations=1000, time_horizon=252):
        """Monte Carlo simulation for portfolio performance"""
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Cholesky decomposition for correlated random returns
        L = np.linalg.cholesky(cov_matrix)
        
        portfolio_values = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(0, 1, (time_horizon, len(returns.columns)))
            correlated_returns = np.dot(random_returns, L.T) + mean_returns.values
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(correlated_returns, weights)
            
            # Calculate cumulative value
            portfolio_value = np.prod(1 + portfolio_returns)
            portfolio_values.append(portfolio_value)
        
        return np.array(portfolio_values)

optimizer = PortfolioOptimizer()

@app.post("/optimize", response_model=OptimizationResult)
async def optimize_portfolio(request: PortfolioRequest):
    try:
        # Fetch data
        prices = optimizer.fetch_data(request.symbols)
        returns = optimizer.calculate_returns(prices)
        
        # Choose optimization method
        if request.method.lower() == "mean_variance":
            weights = optimizer.mean_variance_optimization(returns, request.risk_tolerance)
            method_name = "Mean-Variance Optimization"
        elif request.method.lower() == "risk_parity":
            weights = optimizer.risk_parity_optimization(returns)
            method_name = "Risk Parity"
        elif request.method.lower() == "black_litterman":
            weights = optimizer.black_litterman_optimization(returns, request.views)
            method_name = "Black-Litterman"
        else:
            weights = optimizer.mean_variance_optimization(returns, request.risk_tolerance)
            method_name = "Mean-Variance Optimization (Default)"
        
        # Calculate portfolio metrics
        portfolio_return, portfolio_vol, sharpe_ratio, var_95, cvar_95 = \
            optimizer.calculate_portfolio_metrics(weights, returns)
        
        # Calculate dollar allocations
        allocation_dollars = {
            symbol: float(weight * request.investment_amount)
            for symbol, weight in zip(request.symbols, weights)
        }
        
        # Create weight dictionary
        weight_dict = {
            symbol: float(weight)
            for symbol, weight in zip(request.symbols, weights)
        }
        
        return OptimizationResult(
            weights=weight_dict,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe_ratio),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            method_used=method_name,
            allocation_dollars=allocation_dollars
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monte_carlo")
async def run_monte_carlo(request: PortfolioRequest):
    try:
        # Fetch data and optimize portfolio first
        prices = optimizer.fetch_data(request.symbols)
        returns = optimizer.calculate_returns(prices)
        
        # Get optimal weights
        if request.method.lower() == "risk_parity":
            weights = optimizer.risk_parity_optimization(returns)
        elif request.method.lower() == "black_litterman":
            weights = optimizer.black_litterman_optimization(returns, request.views)
        else:
            weights = optimizer.mean_variance_optimization(returns, request.risk_tolerance)
        
        # Run Monte Carlo simulation
        simulation_results = optimizer.monte_carlo_simulation(
            returns, weights, request.monte_carlo_runs
        )
        
        # Calculate statistics
        mean_return = np.mean(simulation_results)
        std_return = np.std(simulation_results)
        percentiles = np.percentile(simulation_results, [5, 25, 50, 75, 95])
        
        return {
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "percentile_5": float(percentiles[0]),
            "percentile_25": float(percentiles[1]),
            "median": float(percentiles[2]),
            "percentile_75": float(percentiles[3]),
            "percentile_95": float(percentiles[4]),
            "simulation_results": simulation_results.tolist()[:100],  # Return first 100 for visualization
            "probability_of_loss": float(np.mean(simulation_results < 1.0)),
            "expected_value": float(mean_return * request.investment_amount)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/efficient_frontier")
async def get_efficient_frontier(symbols: str, points: int = 20):
    try:
        symbol_list = symbols.split(",")
        prices = optimizer.fetch_data(symbol_list)
        returns = optimizer.calculate_returns(prices)
        
        n_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Generate efficient frontier
        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), points)
        frontier_volatility = []
        frontier_weights = []
        
        for target in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, target=target: np.sum(x * mean_returns) - target}
            ]
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_guess = np.array([1/n_assets] * n_assets)
            
            def minimize_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            try:
                result = minimize(minimize_volatility, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
                if result.success:
                    frontier_volatility.append(minimize_volatility(result.x))
                    frontier_weights.append(result.x.tolist())
                else:
                    frontier_volatility.append(None)
                    frontier_weights.append(None)
            except:
                frontier_volatility.append(None)
                frontier_weights.append(None)
        
        return {
            "returns": target_returns.tolist(),
            "volatilities": frontier_volatility,
            "weights": frontier_weights,
            "symbols": symbol_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Portfolio Optimization API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)