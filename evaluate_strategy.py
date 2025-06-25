#!/usr/bin/env python3
"""
PairWise Alpha Strategy Performance Evaluator
============================================

This tool evaluates your trading strategy using the official PairWise Alpha metrics:
- Profitability (45 points max)
- Sharpe Ratio (35 points max) 
- Max Drawdown (20 points max)
- Stability Score (bonus points)

Usage:
    python evaluate_strategy.py

Requirements:
    - strategy.py (your strategy file)
    - data_download_manager.py (provided data manager)
    - Internet connection (for data download)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import the required modules
try:
    from data_download_manager import CryptoDataManager
    from strategy import get_coin_metadata, generate_signals
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure 'strategy.py' and 'data_download_manager.py' are in the same directory")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingSimulator:
    """
    Simulates trading based on signals with realistic position sizing and fee calculation.
    """
    
    def __init__(self, starting_capital: float = 1000.0, fee_pct: float = 0.001):
        self.starting_capital = starting_capital
        self.fee_pct = fee_pct  # 0.1% fee each way
        
    def simulate_trading(self, signals_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate trading based on signals and return portfolio performance.
        
        Args:
            signals_df: DataFrame with timestamp, symbol, signal, position_size
            price_data: DataFrame with timestamp and price columns
            
        Returns:
            DataFrame with timestamp, portfolio_value, cash, positions, trades
        """
        # Merge signals with price data
        df = pd.merge(signals_df, price_data, on='timestamp', how='left')
        
        # Initialize tracking variables
        cash = self.starting_capital
        positions = {}  # {symbol: {shares: float, avg_cost: float}}
        portfolio_history = []
        trade_history = []
        
        for idx, row in df.iterrows():
            timestamp = row['timestamp']
            symbol = row['symbol']
            signal = row['signal']
            position_size = row['position_size']
            
            # Get current price (try multiple column formats)
            price = None
            possible_price_cols = [f'close_{symbol}_1H', f'{symbol}_price', 'price', 'close']
            for col in possible_price_cols:
                if col in row and pd.notna(row[col]):
                    price = row[col]
                    break
                    
            if price is None or price <= 0:
                # No valid price, skip this timestamp
                current_portfolio_value = cash + sum(
                    pos['shares'] * price if price and price > 0 else 0 
                    for pos in positions.values()
                )
                portfolio_history.append({
                    'timestamp': timestamp,
                    'portfolio_value': current_portfolio_value,
                    'cash': cash,
                    'positions': positions.copy()
                })
                continue
            
            # Initialize position if not exists
            if symbol not in positions:
                positions[symbol] = {'shares': 0.0, 'avg_cost': 0.0}
            
            # Execute trading logic
            if signal == 'BUY' and position_size > 0:
                # BUY: Use position_size % of available cash
                allocated_cash = cash * position_size
                if allocated_cash > 1.0:  # Minimum $1 trade
                    fee = allocated_cash * self.fee_pct
                    investment_after_fee = allocated_cash - fee
                    shares_bought = investment_after_fee / price
                    
                    # Update cash
                    cash -= allocated_cash
                    
                    # Update position with weighted average cost
                    old_shares = positions[symbol]['shares']
                    old_cost = positions[symbol]['avg_cost']
                    new_total_shares = old_shares + shares_bought
                    
                    if new_total_shares > 0:
                        positions[symbol]['avg_cost'] = (
                            (old_shares * old_cost + shares_bought * price) / new_total_shares
                        )
                        positions[symbol]['shares'] = new_total_shares
                    
                    # Record trade
                    trade_history.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares_bought,
                        'price': price,
                        'gross_amount': allocated_cash,
                        'fee': fee,
                        'net_amount': investment_after_fee
                    })
            
            elif signal == 'SELL' and position_size > 0 and positions[symbol]['shares'] > 0:
                # SELL: Sell position_size % of current holdings
                shares_to_sell = positions[symbol]['shares'] * position_size
                if shares_to_sell > 0.0001:  # Minimum meaningful trade
                    gross_proceeds = shares_to_sell * price
                    fee = gross_proceeds * self.fee_pct
                    net_proceeds = gross_proceeds - fee
                    
                    # Update cash
                    cash += net_proceeds
                    
                    # Update position
                    positions[symbol]['shares'] -= shares_to_sell
                    if positions[symbol]['shares'] < 0.0001:
                        positions[symbol]['shares'] = 0.0
                    
                    # Record trade
                    trade_history.append({
                        'timestamp': timestamp,
                        'symbol': symbol,  
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': price,
                        'gross_amount': gross_proceeds,
                        'fee': fee,
                        'net_amount': net_proceeds
                    })
            
            # Calculate current portfolio value
            position_values = sum(
                pos['shares'] * price for pos in positions.values()
                if pos['shares'] > 0
            )
            current_portfolio_value = cash + position_values
            
            # Record portfolio state
            portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': current_portfolio_value,
                'cash': cash,
                'positions': positions.copy()
            })
        
        # Final liquidation
        final_timestamp = df['timestamp'].iloc[-1] if len(df) > 0 else pd.Timestamp.now()
        final_price_row = df.iloc[-1] if len(df) > 0 else None
        
        if final_price_row is not None:
            for symbol, position in positions.items():
                if position['shares'] > 0:
                    # Get final price for liquidation
                    final_price = None
                    for col in possible_price_cols:
                        if col in final_price_row and pd.notna(final_price_row[col]):
                            final_price = final_price_row[col]
                            break
                    
                    if final_price and final_price > 0:
                        gross_proceeds = position['shares'] * final_price
                        fee = gross_proceeds * self.fee_pct
                        net_proceeds = gross_proceeds - fee
                        cash += net_proceeds
                        
                        trade_history.append({
                            'timestamp': final_timestamp,
                            'symbol': symbol,
                            'action': 'LIQUIDATE',
                            'shares': position['shares'],
                            'price': final_price,
                            'gross_amount': gross_proceeds,
                            'fee': fee,
                            'net_amount': net_proceeds
                        })
                        
                        position['shares'] = 0.0
        
        # Create final portfolio history
        portfolio_df = pd.DataFrame(portfolio_history)
        trades_df = pd.DataFrame(trade_history)
        
        return portfolio_df, trades_df


class PerformanceCalculator:
    """
    Calculates PairWise Alpha performance metrics.
    """
    
    @staticmethod
    def calculate_profitability_score(portfolio_df: pd.DataFrame, starting_capital: float) -> Tuple[float, float]:
        """Calculate profitability percentage and score (max 45 points)."""
        if len(portfolio_df) == 0:
            return 0.0, 0.0
            
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        pnl_percent = ((final_value - starting_capital) / starting_capital) * 100
        
        # Scoring formula: score = min(45.0, max(0.0, (pnl_percent / 300.0) * 45.0))
        score = min(45.0, max(0.0, (pnl_percent / 300.0) * 45.0))
        
        return pnl_percent, score
    
    @staticmethod
    def calculate_sharpe_ratio_score(trades_df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate Sharpe ratio and score (max 35 points)."""
        if len(trades_df) < 2:
            return 0.0, 0.0
        
        # Calculate trade-level returns
        trade_returns = []
        buy_trades = trades_df[trades_df['action'] == 'BUY'].copy()
        sell_trades = trades_df[trades_df['action'].isin(['SELL', 'LIQUIDATE'])].copy()
        
        # Match buy/sell pairs by symbol and timestamp order
        for symbol in trades_df['symbol'].unique():
            symbol_buys = buy_trades[buy_trades['symbol'] == symbol].sort_values('timestamp')
            symbol_sells = sell_trades[sell_trades['symbol'] == symbol].sort_values('timestamp')
            
            buy_idx = 0
            for _, sell in symbol_sells.iterrows():
                if buy_idx < len(symbol_buys):
                    buy = symbol_buys.iloc[buy_idx]
                    
                    # Calculate return for this trade pair
                    buy_cost = buy['gross_amount']  # Including fees
                    sell_proceeds = sell['net_amount']  # After fees
                    
                    if buy_cost > 0:
                        trade_return = (sell_proceeds - buy_cost) / buy_cost
                        trade_returns.append(trade_return)
                    
                    buy_idx += 1
        
        if len(trade_returns) < 2:
            return 0.0, 0.0
        
        # Calculate Sharpe ratio
        mean_return = np.mean(trade_returns)
        std_return = np.std(trade_returns, ddof=1)
        
        if std_return == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = mean_return / std_return  # Risk-free rate = 0
        
        # Scoring formula: score = min(35.0, max(0.0, (sharpe / 5.0) * 35.0))
        score = min(35.0, max(0.0, (sharpe_ratio / 5.0) * 35.0))
        
        return sharpe_ratio, score
    
    @staticmethod
    def calculate_max_drawdown_score(portfolio_df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate maximum drawdown percentage and score (max 20 points)."""
        if len(portfolio_df) == 0:
            return 0.0, 0.0
        
        portfolio_values = portfolio_df['portfolio_value'].values
        
        # Calculate running maximum (peak)
        peaks = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdowns
        drawdowns = (peaks - portfolio_values) / peaks
        max_drawdown_pct = np.max(drawdowns) * 100
        
        # Scoring formula: score = max(0.0, (1.0 - (drawdown / 50.0)) * 20.0)
        score = max(0.0, (1.0 - (max_drawdown_pct / 50.0)) * 20.0)
        
        return max_drawdown_pct, score
    
    @staticmethod
    def calculate_stability_score(portfolio_df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate R-squared stability score (bonus points)."""
        if len(portfolio_df) < 10:
            return 0.0, 0.0
        
        # Create time index for regression
        time_index = np.arange(len(portfolio_df))
        portfolio_values = portfolio_df['portfolio_value'].values
        
        # Linear regression: portfolio_value = β₀ + β₁ × time + ε
        try:
            correlation_matrix = np.corrcoef(time_index, portfolio_values)
            r_squared = correlation_matrix[0, 1] ** 2
            
            if np.isnan(r_squared):
                r_squared = 0.0
                
        except:
            r_squared = 0.0
        
        # Scoring formula: score = min(5.0, max(0.0, r_squared * 5.0))
        score = min(5.0, max(0.0, r_squared * 5.0))
        
        return r_squared, score


def evaluate_strategy():
    """
    Main evaluation function that runs the complete strategy evaluation.
    """
    print("🚀 Starting PairWise Alpha Strategy Evaluation")
    print("=" * 60)
    
    # Step 1: Get strategy metadata
    try:
        metadata = get_coin_metadata()
        print(f"📊 Strategy Metadata:")
        print(f"   Targets: {metadata['targets']}")
        print(f"   Anchors: {metadata['anchors']}")
    except Exception as e:
        print(f"❌ Error getting metadata: {e}")
        return
    
    # Step 2: Download market data
    print(f"\n📥 Downloading market data...")
    data_manager = CryptoDataManager()
    
    try:
        # Combine targets and anchors for data download
        all_symbols = metadata['targets'] + metadata['anchors']
        market_data = data_manager.get_market_data(all_symbols)
        print(f"✅ Downloaded data shape: {market_data.shape}")
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return
    
    # Step 3: Split data into target and anchor DataFrames
    print(f"\n🔄 Preparing data for strategy...")
    try:
        # Get target symbols
        target_symbols = [t['symbol'] for t in metadata['targets']]
        anchor_symbols = [a['symbol'] for a in metadata['anchors']]
        
        # Create target DataFrame
        target_cols = ['timestamp']
        for target in metadata['targets']:
            symbol = target['symbol']
            timeframe = target['timeframe']
            target_cols.extend([
                f'open_{symbol}_{timeframe}',
                f'high_{symbol}_{timeframe}',
                f'low_{symbol}_{timeframe}',
                f'close_{symbol}_{timeframe}',
                f'volume_{symbol}_{timeframe}'
            ])
        
        target_df = market_data[target_cols].copy()
        
        # Create anchor DataFrame  
        anchor_cols = ['timestamp']
        for anchor in metadata['anchors']:
            symbol = anchor['symbol']
            timeframe = anchor['timeframe']
            anchor_cols.extend([
                f'open_{symbol}_{timeframe}',
                f'high_{symbol}_{timeframe}',
                f'low_{symbol}_{timeframe}',
                f'close_{symbol}_{timeframe}',
                f'volume_{symbol}_{timeframe}'
            ])
        
        anchor_df = market_data[anchor_cols].copy()
        
        print(f"✅ Target data shape: {target_df.shape}")
        print(f"✅ Anchor data shape: {anchor_df.shape}")
        
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return
    
    # Step 4: Generate signals
    print(f"\n🧠 Generating trading signals...")
    try:
        signals_df = generate_signals(anchor_df, target_df)
        print(f"✅ Generated {len(signals_df)} signal rows")
        
        # Validate signals
        buy_count = (signals_df['signal'] == 'BUY').sum()
        sell_count = (signals_df['signal'] == 'SELL').sum()
        hold_count = (signals_df['signal'] == 'HOLD').sum()
        
        print(f"   📈 BUY signals: {buy_count}")
        print(f"   📉 SELL signals: {sell_count}")
        print(f"   ⏸️  HOLD signals: {hold_count}")
        print(f"   🔄 Complete pairs: {min(buy_count, sell_count)}")
        
    except Exception as e:
        print(f"❌ Error generating signals: {e}")
        return
    
    # Step 5: Run trading simulation
    print(f"\n💹 Running trading simulation...")
    try:
        simulator = TradingSimulator(starting_capital=1000.0)
        portfolio_df, trades_df = simulator.simulate_trading(signals_df, market_data)
        
        print(f"✅ Simulation completed")
        print(f"   📊 Portfolio history: {len(portfolio_df)} records")
        print(f"   💰 Trades executed: {len(trades_df)} trades")
        
        if len(portfolio_df) > 0:
            final_value = portfolio_df['portfolio_value'].iloc[-1]
            print(f"   🎯 Final portfolio value: ${final_value:.2f}")
        
    except Exception as e:
        print(f"❌ Error in trading simulation: {e}")
        return
    
    # Step 6: Calculate performance metrics
    print(f"\n📊 Calculating Performance Metrics")
    print("=" * 60)
    
    calc = PerformanceCalculator()
    
    # Profitability
    pnl_pct, prof_score = calc.calculate_profitability_score(portfolio_df, 1000.0)
    print(f"📈 PROFITABILITY")
    print(f"   Return: {pnl_pct:.2f}%")
    print(f"   Score: {prof_score:.1f} / 45.0")
    print(f"   Status: {'✅ PASS' if prof_score >= 15 else '❌ FAIL'} (need ≥15)")
    
    # Sharpe Ratio
    sharpe, sharpe_score = calc.calculate_sharpe_ratio_score(trades_df)
    print(f"\n📊 SHARPE RATIO")
    print(f"   Ratio: {sharpe:.3f}")
    print(f"   Score: {sharpe_score:.1f} / 35.0")
    print(f"   Status: {'✅ PASS' if sharpe_score >= 10 else '❌ FAIL'} (need ≥10)")
    
    # Max Drawdown
    drawdown_pct, drawdown_score = calc.calculate_max_drawdown_score(portfolio_df)
    print(f"\n📉 MAX DRAWDOWN")
    print(f"   Drawdown: {drawdown_pct:.2f}%")
    print(f"   Score: {drawdown_score:.1f} / 20.0")
    print(f"   Status: {'✅ PASS' if drawdown_score >= 5 else '❌ FAIL'} (need ≥5)")
    
    # Stability Score
    r_squared, stability_score = calc.calculate_stability_score(portfolio_df)
    print(f"\n🔄 STABILITY SCORE (Bonus)")  
    print(f"   R-squared: {r_squared:.3f}")
    print(f"   Score: {stability_score:.1f} / 5.0")
    
    # Total Score
    total_score = prof_score + sharpe_score + drawdown_score + stability_score
    print(f"\n🏆 TOTAL SCORE")
    print(f"   Score: {total_score:.1f} / 100.0")
    print(f"   Status: {'✅ PASS' if total_score >= 60 else '❌ FAIL'} (need ≥60)")
    
    # Final qualification status
    print(f"\n🎯 QUALIFICATION STATUS")
    print("=" * 60)
    
    individual_pass = prof_score >= 15 and sharpe_score >= 10 and drawdown_score >= 5
    total_pass = total_score >= 60
    qualified = individual_pass and total_pass
    
    if qualified:
        print("🎉 QUALIFIED! Your strategy meets all requirements.")
        print("✅ All individual metrics pass their thresholds")
        print("✅ Total score exceeds 60 points")
    else:
        print("❌ NOT QUALIFIED. Issues found:")
        if not individual_pass:
            if prof_score < 15:
                print(f"   • Profitability too low: {prof_score:.1f} < 15")
            if sharpe_score < 10:
                print(f"   • Sharpe ratio too low: {sharpe_score:.1f} < 10")
            if drawdown_score < 5:
                print(f"   • Max drawdown too high: {drawdown_score:.1f} < 5")
        if not total_pass:
            print(f"   • Total score too low: {total_score:.1f} < 60")
    
    print(f"\n📋 Summary for optimization:")
    print(f"   Need to improve: Profitability by {max(0, 15-prof_score):.1f} pts, "
          f"Sharpe by {max(0, 10-sharpe_score):.1f} pts, "
          f"Drawdown by {max(0, 5-drawdown_score):.1f} pts")
    
    return {
        'qualified': qualified,
        'profitability': {'pct': pnl_pct, 'score': prof_score},
        'sharpe': {'ratio': sharpe, 'score': sharpe_score},
        'drawdown': {'pct': drawdown_pct, 'score': drawdown_score},
        'stability': {'r_squared': r_squared, 'score': stability_score},
        'total_score': total_score
    }


if __name__ == "__main__":
    results = evaluate_strategy()