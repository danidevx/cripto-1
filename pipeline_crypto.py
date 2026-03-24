import os
import logging
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import vectorbt as vbt
import ccxt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineCryptoBT:
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.SYMBOL = 'BTC/USDT'
        self.TIMEFRAME = '5m'
        self.N_STRATEGIES = 500
        self.MONTHS_DATA = 15
        self.MIN_SHARPE = 0.2
        
        self.df = None
        self.all_results = []

    def descargar_datos_binance(self):
        logger.info(f"Descargando datos de {self.SYMBOL} desde Binance...")
        exchange = ccxt.binance()
        
        # Calcular fecha de inicio (15 meses atrás)
        since = exchange.parse8601((datetime.now() - timedelta(days=15 * 30)).isoformat())
        
        all_ohlcv = []
        while since < exchange.milliseconds():
            ohlcv = exchange.fetch_ohlcv(self.SYMBOL, self.TIMEFRAME, since=since)
            if not ohlcv:
                break
            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            logger.info(f"Descargados hasta {exchange.iso8601(since)}")
            time.sleep(0.1)

        df = pd.DataFrame(all_ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df = df.set_index('Date')
        
        # Asegurar que tenemos exactamente 15 meses
        cutoff = datetime.now() - timedelta(days=15 * 30)
        df = df[df.index >= cutoff]
        
        self.df = df
        df.to_csv(self.data_dir / "btc_5m_data.csv")
        logger.info(f"Datos guardados. Total velas: {len(df)}")
        return df

    def generar_parametros(self, n: int) -> List[Dict]:
        estrategias = []
        tipos = ['ma_cross', 'rsi_only', 'rsi_vol', 'vol_only', 'ma_vol', 'rsi_ma']
        
        for i in range(n):
            # Distribución equitativa: asigna el tipo basándose en el índice i
            tipo = tipos[i % len(tipos)]
            
            sl_pct = round(random.uniform(0.002, 0.05), 4)
            tp_pct = round(random.uniform(0.005, 0.1), 4)
            
            params = {
                'name': f'STRAT_{tipo}_{i}',
                'tipo': tipo,
                'stop_loss_pct': sl_pct,
                'take_profit_pct': tp_pct,
            }
            
            if tipo == 'ma_cross':
                params.update({'fast_period': random.randint(5, 100), 'slow_period': random.randint(50, 200)})
            elif tipo == 'rsi_only':
                params.update({'rsi_period': random.randint(7, 30), 'rsi_oversold': random.randint(20, 40)})
            elif tipo == 'rsi_vol':
                params.update({'rsi_period': random.randint(7, 30), 'rsi_oversold': random.randint(20, 40), 'vol_period': random.randint(10, 50), 'vol_mult': round(random.uniform(1.0, 3.0), 2)})
            elif tipo == 'vol_only':
                params.update({'vol_period': random.randint(10, 50), 'vol_mult': round(random.uniform(1.0, 3.0), 2)})
            elif tipo == 'ma_vol':
                params.update({'ma_period': random.randint(20, 200), 'vol_period': random.randint(10, 50), 'vol_mult': round(random.uniform(1.0, 3.0), 2)})
            elif tipo == 'rsi_ma':
                params.update({'rsi_period': random.randint(7, 30), 'rsi_oversold': random.randint(20, 40), 'ma_period': random.randint(20, 200)})
            
            estrategias.append(params)
        return estrategias

    def run_estrategia(self, df: pd.DataFrame, params: Dict) -> Dict:
        try:
            close = df['Close']
            entries = pd.Series(False, index=close.index)
            
            if params['tipo'] == 'ma_cross':
                fast = vbt.MA.run(close, window=params['fast_period'])
                slow = vbt.MA.run(close, window=params['slow_period'])
                entries = fast.ma > slow.ma
            elif params['tipo'] == 'rsi_only':
                rsi = vbt.RSI.run(close, window=params['rsi_period'])
                entries = rsi.rsi < params['rsi_oversold']
            elif params['tipo'] == 'rsi_vol':
                rsi = vbt.RSI.run(close, window=params['rsi_period'])
                vol_ma = vbt.MA.run(df['Volume'], window=params['vol_period'])
                entries = (rsi.rsi < params['rsi_oversold']) & (df['Volume'] > vol_ma.ma * params['vol_mult'])
            elif params['tipo'] == 'vol_only':
                vol_ma = vbt.MA.run(df['Volume'], window=params['vol_period'])
                entries = df['Volume'] > vol_ma.ma * params['vol_mult']
            elif params['tipo'] == 'ma_vol':
                ma = vbt.MA.run(close, window=params['ma_period'])
                vol_ma = vbt.MA.run(df['Volume'], window=params['vol_period'])
                entries = (close > ma.ma) & (df['Volume'] > vol_ma.ma * params['vol_mult'])
            elif params['tipo'] == 'rsi_ma':
                rsi = vbt.RSI.run(close, window=params['rsi_period'])
                ma = vbt.MA.run(close, window=params['ma_period'])
                entries = (rsi.rsi < params['rsi_oversold']) & (close > ma.ma)
            
            entries = entries.vbt.signals.fshift(1).fillna(False)
            
            pf = vbt.Portfolio.from_signals(
                close, entries=entries, exits=None,
                init_cash=10000, fees=0.001, freq='5m',
                sl_stop=params['stop_loss_pct'], tp_stop=params['take_profit_pct']
            )
            
            returns = pf.returns()
            return {
                'params': params,
                'sharpe': pf.sharpe_ratio(),
                'total_return': pf.total_return() * 100,
                'max_dd': pf.max_drawdown() * 100,
                'profit_factor': pf.profit_factor(),
                'trades': pf.trades.count(),
                'pf_obj': pf
            }
        except Exception as e:
            return None

    def calcular_periodos(self, pf, total_months=15, split_months=1.5):
        # Dividir los 15 meses en 10 periodos de 1.5 meses
        returns = pf.returns()
        period_len = int((len(returns) / total_months) * split_months)
        
        period_returns = []
        for i in range(0, len(returns), period_len):
            chunk = returns.iloc[i : i + period_len]
            if len(chunk) == 0: break
            period_returns.append(chunk.sum() * 100)
            
        return period_returns[:10] # Asegurar 10 periodos

    def run(self):
        self.descargar_datos_binance()
        estrategias_params = self.generar_parametros(self.N_STRATEGIES)
        
        resultados = []
        for i, params in enumerate(estrategias_params):
            if i % 50 == 0: logger.info(f"Evaluando {i}/{self.N_STRATEGIES}...")
            res = self.run_estrategia(self.df, params)
            if res and res['trades'] > 0:
                resultados.append(res)
        
        if len(resultados) == 0:
            logger.error("Ninguna estrategia produjo trades")
            return
        
        # Top 10 por Sharpe
        top10 = sorted(resultados, key=lambda x: x['sharpe'] if not np.isnan(x['sharpe']) else -1, reverse=True)[:10]
        
        # Generar informe.md
        with open(self.results_dir / "informe.md", "w") as f:
            f.write("# INFORME DE ESTRATEGIAS BITCOIN (5m)\n\n")
            f.write(f"Activo: {self.SYMBOL} | Temporalidad: {self.TIMEFRAME} | Periodo: {self.MONTHS_DATA} meses\n\n")
            
            f.write("## TOP 10 ESTRATEGIAS Y DESGLOSE TEMPORAL (Periodos de 1.5 meses)\n\n")
            
            for i, res in enumerate(top10):
                p = res['params']
                per_rets = self.calcular_periodos(res['pf_obj'])
                
                f.write(f"### {i+1}. {p['name']}\n")
                f.write(f"- **Tipo**: {p['tipo']}\n")
                f.write(f"- **Sharpe**: {res['sharpe']:.2f} | **Retorno Total**: {res['total_return']:.2f}% | **MaxDD**: {res['max_dd']:.2f}%\n")
                f.write(f"- **Profit Factor**: {res['profit_factor']:.2f} | **Trades**: {res['trades']}\n\n")
                
                f.write("| Periodo | Rendimiento |\n")
                f.write("|---|---|\n")
                for j, r in enumerate(per_rets):
                    f.write(f"| {j+1} (1.5m) | {r:.2f}% |\n")
                f.write("\n---\n\n")
        
        logger.info("Pipeline completado. Informe generado en results/informe.md")


if __name__ == "__main__":
    pipeline = PipelineCryptoBT()
    pipeline.run()
