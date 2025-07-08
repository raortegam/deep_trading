import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingDataDownloader:
    def __init__(self, symbol, period='2y'):
        """
        Inicializa el descargador de datos para trading algorítmico
        
        Args:
            symbol (str): Símbolo del activo (ej: 'AAPL', 'MSFT')
            period (str): Período de datos ('1y', '2y', '5y', 'max')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.features = None
        
    def download_data(self):
        """Descarga datos básicos OHLCV"""
        try:
            # Método alternativo más robusto
            self.data = yf.download(self.symbol, period=self.period, progress=False)
            
            # Si falla el primer método, intentar con Ticker
            if self.data.empty:
                ticker = yf.Ticker(self.symbol)
                self.data = ticker.history(period=self.period, auto_adjust=True, prepost=True)
            
            # Limpiar datos
            self.data = self.data.dropna()
            
            # Asegurar que las columnas están en el formato correcto
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.get_level_values(0)
            
            print(f"✓ Datos descargados para {self.symbol}: {len(self.data)} registros")
            print(f"✓ Período: {self.data.index[0]} a {self.data.index[-1]}")
            return True
            
        except Exception as e:
            print(f"✗ Error descargando datos: {e}")
            print("Intentando método alternativo...")
            
            # Método alternativo con fechas específicas
            try:
                end_date = datetime.now()
                if self.period == '1y':
                    start_date = end_date - timedelta(days=365)
                elif self.period == '2y':
                    start_date = end_date - timedelta(days=730)
                elif self.period == '5y':
                    start_date = end_date - timedelta(days=1825)
                else:
                    start_date = end_date - timedelta(days=730)  # Default 2 años
                
                self.data = yf.download(self.symbol, 
                                      start=start_date, 
                                      end=end_date, 
                                      progress=False)
                
                if not self.data.empty:
                    self.data = self.data.dropna()
                    print(f"✓ Datos descargados (método alternativo): {len(self.data)} registros")
                    return True
                else:
                    print("✗ No se pudieron descargar datos")
                    return False
                    
            except Exception as e2:
                print(f"✗ Error en método alternativo: {e2}")
                return False
    
    def calculate_technical_indicators(self):
        """Calcula indicadores técnicos para predicción de dirección"""
        df = self.data.copy()
        
        # === PRECIO Y VOLUMEN ===
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['OC_Ratio'] = (df['Open'] - df['Close']) / df['Close']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # === MEDIAS MÓVILES ===
        df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
        df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # Cruces de medias móviles
        df['SMA_5_20_Cross'] = np.where(df['SMA_5'] > df['SMA_20'], 1, 0)
        df['SMA_10_50_Cross'] = np.where(df['SMA_10'] > df['SMA_50'], 1, 0)
        df['Price_Above_SMA20'] = np.where(df['Close'] > df['SMA_20'], 1, 0)
        
        # === OSCILADORES ===
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['RSI_Oversold'] = np.where(df['RSI'] < 30, 1, 0)
        df['RSI_Overbought'] = np.where(df['RSI'] > 70, 1, 0)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
        df['MACD_Cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['Stoch_Cross'] = np.where(df['Stoch_K'] > df['Stoch_D'], 1, 0)
        
        # Williams %R
        df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'])
        
        # === VOLATILIDAD ===
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatilidad realizada
        df['Volatility_5'] = df['Returns'].rolling(5).std()
        df['Volatility_20'] = df['Returns'].rolling(20).std()
        
        # === MOMENTUM ===
        df['ROC_5'] = talib.ROC(df['Close'], timeperiod=5)
        df['ROC_10'] = talib.ROC(df['Close'], timeperiod=10)
        df['CMO'] = talib.CMO(df['Close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # === VOLUMEN ===
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['Volume_SMA'] = talib.SMA(df['Volume'], timeperiod=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # === PATRONES DE VELAS ===
        df['Doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['Hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['Shooting_Star'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['Engulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        
        # === SOPORTES Y RESISTENCIAS ===
        df['High_5'] = df['High'].rolling(5).max()
        df['Low_5'] = df['Low'].rolling(5).min()
        df['Near_High'] = np.where(df['Close'] >= df['High_5'] * 0.98, 1, 0)
        df['Near_Low'] = np.where(df['Close'] <= df['Low_5'] * 1.02, 1, 0)
        
        # === FEATURES TEMPORALES ===
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Is_Monday'] = np.where(df['Day_of_Week'] == 0, 1, 0)
        df['Is_Friday'] = np.where(df['Day_of_Week'] == 4, 1, 0)
        
        # === TARGET VARIABLE (24H AHEAD) ===
        df['Future_Close'] = df['Close'].shift(-1)  # Precio de cierre siguiente
        df['Future_Return'] = (df['Future_Close'] / df['Close']) - 1
        df['Target_Direction'] = np.where(df['Future_Return'] > 0, 1, 0)  # 1=Sube, 0=Baja
        
        # Target para clasificación multi-clase
        df['Target_Multi'] = pd.cut(df['Future_Return'], 
                                   bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf], 
                                   labels=['Strong_Down', 'Down', 'Neutral', 'Up', 'Strong_Up'])
        
        self.features = df
        print(f"✓ Indicadores técnicos calculados: {len(df.columns)} variables")
        return df
    
    def get_feature_matrix(self, drop_na=True):
        """Obtiene matriz de features lista para ML"""
        if self.features is None:
            print("✗ Primero calcula los indicadores técnicos")
            return None
        
        # Seleccionar features relevantes (excluir precios raw y target)
        feature_cols = [col for col in self.features.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 
                                    'Future_Close', 'Future_Return', 'Target_Direction', 'Target_Multi']]
        
        X = self.features[feature_cols]
        y_binary = self.features['Target_Direction']
        y_multi = self.features['Target_Multi']
        
        if drop_na:
            # Eliminar filas con NaN
            valid_idx = ~(X.isna().any(axis=1) | y_binary.isna())
            X = X.loc[valid_idx]
            y_binary = y_binary.loc[valid_idx]
            y_multi = y_multi.loc[valid_idx]
        
        print(f"✓ Matriz de features: {X.shape}")
        print(f"✓ Distribución target binario: {y_binary.value_counts().to_dict()}")
        
        return X, y_binary, y_multi
    
    def get_latest_features(self):
        """Obtiene las features más recientes para predicción en tiempo real"""
        if self.features is None:
            print("✗ Primero calcula los indicadores técnicos")
            return None
        
        latest = self.features.iloc[-1]
        feature_cols = [col for col in self.features.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 
                                    'Future_Close', 'Future_Return', 'Target_Direction', 'Target_Multi']]
        
        return latest[feature_cols]
    
    def save_data(self, filename=None):
        """Guarda los datos procesados"""
        if filename is None:
            filename = f"{self.symbol}_trading_data.csv"
        
        self.features.to_csv(filename)
        print(f"✓ Datos guardados en {filename}")
    
    def get_data_summary(self):
        """Resumen de los datos descargados"""
        if self.features is None:
            return "No hay datos procesados"
        
        summary = {
            'Symbol': self.symbol,
            'Period': f"{self.features.index[0]} to {self.features.index[-1]}",
            'Total Records': len(self.features),
            'Features': len([col for col in self.features.columns if col not in ['Target_Direction', 'Target_Multi']]),
            'Missing Values': self.features.isnull().sum().sum(),
            'Target Distribution': self.features['Target_Direction'].value_counts().to_dict()
        }
        
        return summary

# === EJEMPLO DE USO ===
if __name__ == "__main__":
    # Configurar el símbolo a descargar
    symbol = "AAPL"  # Cambiar por el símbolo deseado
    
    # Crear instancia del descargador
    downloader = TradingDataDownloader(symbol, period='2y')
    
    # Descargar y procesar datos
    if downloader.download_data():
        df = downloader.calculate_technical_indicators()
        
        # Obtener matriz de features
        X, y_binary, y_multi = downloader.get_feature_matrix()
        
        # Mostrar resumen
        print("\n=== RESUMEN DE DATOS ===")
        summary = downloader.get_data_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Mostrar últimas features
        print("\n=== ÚLTIMAS FEATURES (para predicción) ===")
        latest_features = downloader.get_latest_features()
        print(latest_features.head(10))
        
        # Guardar datos
        downloader.save_data()
        
        print(f"\n✓ Proceso completado. {len(X)} registros listos para entrenamiento")
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target balance: {(y_binary.sum() / len(y_binary) * 100):.1f}% positivo")