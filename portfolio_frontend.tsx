import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { Calculator, TrendingUp, Shield, BarChart3, DollarSign, Target, AlertTriangle, Activity } from 'lucide-react';

const PortfolioOptimizer = () => {
  const [symbols, setSymbols] = useState(['AAPL', 'GOOGL', 'MSFT', 'TSLA']);
  const [method, setMethod] = useState('mean_variance');
  const [riskTolerance, setRiskTolerance] = useState(0.5);
  const [investmentAmount, setInvestmentAmount] = useState(100000);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [monteCarloResults, setMonteCarloResults] = useState(null);
  const [efficientFrontier, setEfficientFrontier] = useState(null);
  const [views, setViews] = useState({});

  const API_BASE = 'http://localhost:8000';

  const optimizationMethods = [
    { value: 'mean_variance', label: 'Mean-Variance Optimization', description: 'Classic MPT approach balancing risk and return' },
    { value: 'risk_parity', label: 'Risk Parity', description: 'Equal risk contribution from all assets' },
    { value: 'black_litterman', label: 'Black-Litterman', description: 'Market equilibrium with investor views' }
  ];

  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00ff00', '#ff00ff', '#00ffff', '#ffff00'];

  const optimizePortfolio = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbols,
          method,
          risk_tolerance: riskTolerance,
          investment_amount: investmentAmount,
          views: Object.keys(views).length > 0 ? views : null
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to optimize portfolio');
      }

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error optimizing portfolio:', error);
      alert('Error optimizing portfolio. Please check your inputs and try again.');
    } finally {
      setLoading(false);
    }
  };

  const runMonteCarloSimulation = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/monte_carlo`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbols,
          method,
          risk_tolerance: riskTolerance,
          investment_amount: investmentAmount,
          monte_carlo_runs: 1000,
          views: Object.keys(views).length > 0 ? views : null
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to run Monte Carlo simulation');
      }

      const data = await response.json();
      setMonteCarloResults(data);
    } catch (error) {
      console.error('Error running Monte Carlo simulation:', error);
      alert('Error running Monte Carlo simulation.');
    } finally {
      setLoading(false);
    }
  };

  const fetchEfficientFrontier = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/efficient_frontier?symbols=${symbols.join(',')}&points=20`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch efficient frontier');
      }

      const data = await response.json();
      setEfficientFrontier(data);
    } catch (error) {
      console.error('Error fetching efficient frontier:', error);
      alert('Error fetching efficient frontier.');
    } finally {
      setLoading(false);
    }
  };

  const addSymbol = () => {
    setSymbols([...symbols, '']);
  };

  const removeSymbol = (index) => {
    const newSymbols = symbols.filter((_, i) => i !== index);
    setSymbols(newSymbols);
  };

  const updateSymbol = (index, value) => {
    const newSymbols = [...symbols];
    newSymbols[index] = value.toUpperCase();
    setSymbols(newSymbols);
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const pieChartData = results ? 
    Object.entries(results.weights).map(([symbol, weight]) => ({
      name: symbol,
      value: weight * 100,
      amount: results.allocation_dollars[symbol]
    })) : [];

  const frontierData = efficientFrontier ? 
    efficientFrontier.returns.map((ret, index) => ({
      return: ret * 100,
      volatility: efficientFrontier.volatilities[index] * 100
    })).filter(point => point.volatility !== null) : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4 flex items-center justify-center gap-3">
            <TrendingUp className="text-emerald-400" />
            Portfolio Optimizer
          </h1>
          <p className="text-xl text-slate-300 max-w-3xl mx-auto">
            Advanced portfolio optimization using Modern Portfolio Theory, Risk Parity, Black-Litterman, and Monte Carlo simulation
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 shadow-2xl">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <Calculator className="text-blue-400" />
                Configuration
              </h2>

              {/* Stock Symbols */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-300 mb-3">Stock Symbols</label>
                {symbols.map((symbol, index) => (
                  <div key={index} className="flex gap-2 mb-2">
                    <input
                      type="text"
                      value={symbol}
                      onChange={(e) => updateSymbol(index, e.target.value)}
                      className="flex-1 px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-400"
                      placeholder="AAPL"
                    />
                    {symbols.length > 2 && (
                      <button
                        onClick={() => removeSymbol(index)}
                        className="px-3 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}
                <button
                  onClick={addSymbol}
                  className="w-full mt-2 px-3 py-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors"
                >
                  + Add Symbol
                </button>
              </div>

              {/* Optimization Method */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-300 mb-3">Optimization Method</label>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                >
                  {optimizationMethods.map((opt) => (
                    <option key={opt.value} value={opt.value} className="bg-slate-800">
                      {opt.label}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-slate-400 mt-1">
                  {optimizationMethods.find(opt => opt.value === method)?.description}
                </p>
              </div>

              {/* Risk Tolerance */}
              {method === 'mean_variance' && (
                <div className="mb-6">
                  <label className="block text-sm font-medium text-slate-300 mb-3">
                    Risk Tolerance: {riskTolerance.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="2.0"
                    step="0.1"
                    value={riskTolerance}
                    onChange={(e) => setRiskTolerance(parseFloat(e.target.value))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-xs text-slate-400 mt-1">
                    <span>Conservative</span>
                    <span>Aggressive</span>
                  </div>
                </div>
              )}

              {/* Investment Amount */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-300 mb-3">Investment Amount</label>
                <input
                  type="number"
                  value={investmentAmount}
                  onChange={(e) => setInvestmentAmount(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              {/* Black-Litterman Views */}
              {method === 'black_litterman' && (
                <div className="mb-6">
                  <label className="block text-sm font-medium text-slate-300 mb-3">Market Views (Optional)</label>
                  {symbols.map((symbol) => (
                    <div key={symbol} className="flex gap-2 mb-2">
                      <span className="text-slate-300 text-sm w-16">{symbol}:</span>
                      <input
                        type="number"
                        step="0.01"
                        placeholder="0.00"
                        onChange={(e) => setViews({...views, [symbol]: parseFloat(e.target.value) || 0})}
                        className="flex-1 px-2 py-1 bg-white/5 border border-white/20 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                      />
                    </div>
                  ))}
                  <p className="text-xs text-slate-400">Expected annual return adjustments</p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="space-y-3">
                <button
                  onClick={optimizePortfolio}
                  disabled={loading}
                  className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {loading ? 'Optimizing...' : 'Optimize Portfolio'}
                </button>
                
                <button
                  onClick={runMonteCarloSimulation}
                  disabled={loading}
                  className="w-full py-3 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-lg font-semibold hover:from-emerald-600 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {loading ? 'Running...' : 'Monte Carlo Simulation'}
                </button>
                
                <button
                  onClick={fetchEfficientFrontier}
                  disabled={loading}
                  className="w-full py-3 bg-gradient-to-r from-orange-500 to-red-600 text-white rounded-lg font-semibold hover:from-orange-600 hover:to-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {loading ? 'Calculating...' : 'Efficient Frontier'}
                </button>
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-8">
            {/* Portfolio Metrics */}
            {results && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 shadow-2xl">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                  <Target className="text-emerald-400" />
                  Optimization Results - {results.method_used}
                </h3>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp className="w-5 h-5 text-green-400" />
                      <span className="text-sm text-slate-300">Expected Return</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{formatPercentage(results.expected_return)}</p>
                  </div>
                  
                  <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-5 h-5 text-blue-400" />
                      <span className="text-sm text-slate-300">Volatility</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{formatPercentage(results.volatility)}</p>
                  </div>
                  
                  <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      <Shield className="w-5 h-5 text-purple-400" />
                      <span className="text-sm text-slate-300">Sharpe Ratio</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{results.sharpe_ratio.toFixed(3)}</p>
                  </div>
                  
                  <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                      <span className="text-sm text-slate-300">VaR (95%)</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{formatPercentage(Math.abs(results.var_95))}</p>
                  </div>
                </div>

                {/* Allocation Chart */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Portfolio Allocation</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={pieChartData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {pieChartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Dollar Allocation</h4>
                    <div className="space-y-3">
                      {Object.entries(results.allocation_dollars).map(([symbol, amount]) => (
                        <div key={symbol} className="flex justify-between items-center bg-white/5 rounded-lg p-3">
                          <span className="text-white font-medium">{symbol}</span>
                          <span className="text-emerald-400 font-bold">{formatCurrency(amount)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Monte Carlo Results */}
            {monteCarloResults && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 shadow-2xl">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                  <BarChart3 className="text-blue-400" />
                  Monte Carlo Simulation Results
                </h3>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      <DollarSign className="w-5 h-5 text-green-400" />
                      <span className="text-sm text-slate-300">Expected Value</span>
                    </div>
                    <p className="text-xl font-bold text-white">{formatCurrency(monteCarloResults.expected_value)}</p>
                  </div>
                  
                  <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                      <span className="text-sm text-slate-300">Probability of Loss</span>
                    </div>
                    <p className="text-xl font-bold text-white">{formatPercentage(monteCarloResults.probability_of_loss)}</p>
                  </div>
                  
                  <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp className="w-5 h-5 text-blue-400" />
                      <span className="text-sm text-slate-300">95th Percentile</span>
                    </div>
                    <p className="text-xl font-bold text-white">{formatCurrency(monteCarloResults.percentile_95 * investmentAmount)}</p>
                  </div>
                  
                  <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp className="w-5 h-5 text-orange-400" />
                      <span className="text-sm text-slate-300">5th Percentile</span>
                    </div>
                    <p className="text-xl font-bold text-white">{formatCurrency(monteCarloResults.percentile_5 * investmentAmount)}</p>
                  </div>
                </div>

                <div className="bg-white/5 rounded-lg p-4">
                  <h4 className="text-lg font-semibold text-white mb-4">Return Distribution</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={monteCarloResults.simulation_results.map((result, index) => ({ 
                      index: index + 1, 
                      return: (result - 1) * 100 
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="index" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip 
                        formatter={(value) => [`${value.toFixed(2)}%`, 'Return']}
                        labelFormatter={(value) => `Simulation ${value}`}
                        contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      />
                      <Bar dataKey="return" fill="#3B82F6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Efficient Frontier */}
            {efficientFrontier && frontierData.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 shadow-2xl">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                  <Activity className="text-orange-400" />
                  Efficient Frontier
                </h3>
                
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={frontierData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="volatility" 
                      stroke="#9CA3AF"
                      label={{ value: 'Volatility (%)', position: 'insideBottom', offset: -10, style: { textAnchor: 'middle', fill: '#9CA3AF' } }}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      label={{ value: 'Expected Return (%)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#9CA3AF' } }}
                    />
                    <Tooltip 
                      formatter={(value, name) => [`${value.toFixed(2)}%`, name === 'return' ? 'Expected Return' : 'Volatility']}
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="return" 
                      stroke="#F59E0B" 
                      strokeWidth={3}
                      dot={{ fill: '#F59E0B', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, fill: '#F59E0B' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
                
                <p className="text-sm text-slate-400 mt-4">
                  The efficient frontier shows the optimal risk-return combinations. Each point represents the maximum expected return for a given level of risk.
                </p>
              </div>
            )}

            {/* Help Section */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 shadow-2xl">
              <h3 className="text-xl font-bold text-white mb-4">Optimization Methods</h3>
              <div className="space-y-4">
                {optimizationMethods.map((method) => (
                  <div key={method.value} className="bg-white/5 rounded-lg p-4 border border-white/10">
                    <h4 className="text-lg font-semibold text-white mb-2">{method.label}</h4>
                    <p className="text-slate-300 text-sm">{method.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3B82F6;
          cursor: pointer;
        }
        
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3B82F6;
          cursor: pointer;
          border: none;
        }
      `}</style>
    </div>
  );
};

export default PortfolioOptimizer