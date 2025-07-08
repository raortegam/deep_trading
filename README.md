# Deep Trading

Herramientas para análisis y trading algorítmico.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/raortegam/deep_trading.git
   cd deep_trading
   ```

2. Instala las dependencias con Poetry:
   ```bash
   poetry install
   ```

## Uso

```python
from functions.download_data import TradingDataDownloader

downloader = TradingDataDownloader("AAPL")
downloader.download_data()
```

## Estructura del Proyecto

```
deep_trading/
├── functions/
│   └── download_data.py
├── tests/
├── .gitignore
└── pyproject.toml
```
