# Polymarket Trading Bot

A production-grade Python bot for Polymarket prediction markets. Combines the best implementations from 25+ open-source repos into a single, well-tested system.

## Architecture

| Module | Primary Source Repos | Functionality |
|--------|---------------------|---------------|
| `clob_client.py` | Polymarket/py-clob-client, hbr-l/polypy | CLOB API wrapper, EIP-712 signing, HMAC auth, order execution |
| `market_scanner.py` | realfishsam/prediction-market-arbitrage-bot, Polymarket/poly-market-maker, demone456/kalshi-polymarket-bot | Gamma API market discovery, keyword classification |
| `kelly.py` | Polymarket/poly-market-maker, realfishsam/prediction-market-arbitrage-bot | Fractional Kelly criterion, position limits |
| `ssvi.py` | Polymarket/poly-market-maker, ThinkEnigmatic/polymarket-bot-arena | SSVI volatility surface fitting, probability extraction |
| `persistence.py` | perpetual-s/polymarket-python-infrastructure, Jonmaa/btc-polymarket-bot | SQLite state management, trade logging |
| `redemption.py` | Polymarket/py-clob-client, Polymarket/go-order-utils | On-chain CTF redemption via web3.py |
| `bot.py` | Polymarket/poly-market-maker, Jonmaa/btc-polymarket-bot | Async scan loop, orchestration |
| `config.py` | perpetual-s/polymarket-python-infrastructure | Environment-based configuration |

## Features

- **Paper mode**: Full simulation without real orders
- **GTC limit orders only**: No market orders, no IOC
- **Fractional Kelly sizing**: Configurable fraction (default 1/4 Kelly)
- **SQLite persistence**: Orders, positions, scan history, redemptions
- **Async scanning loop**: Non-blocking market discovery
- **Market classification**: Crypto, politics, sports, finance, etc.
- **SSVI calibration**: Implied vol surface fitting for probability estimation
- **On-chain redemption**: Redeem winning positions via web3.py
- **Position limits**: Configurable max exposure

## Setup

```bash
cp .env.example .env
# Edit .env with your credentials
pip install -r requirements.txt
```

## Usage

```bash
# Paper mode, single scan
python bot.py --once --paper

# Paper mode, continuous loop
python bot.py --paper

# Live mode (requires real credentials)
python bot.py --live

# Custom log level
python bot.py --once --paper --log-level DEBUG
```

## Testing

```bash
pytest tests/ -v
```

## Comparison Matrix

| Repo | CLOB Client | Order Exec | Market Scan | Kelly | SSVI | Proxy | EIP-712 | Redemption | Arbitrage |
|------|:-----------:|:----------:|:-----------:|:-----:|:----:|:-----:|:-------:|:----------:|:---------:|
| Polymarket/py-clob-client | **Best** | Good | - | - | - | Yes | **Best** | - | - |
| Polymarket/poly-market-maker | Good | **Best** | Good | Good | Good | Yes | Good | - | - |
| realfishsam/prediction-market-arbitrage-bot | - | - | **Best** | Good | - | - | - | - | **Best** |
| Jonmaa/btc-polymarket-bot | Good | Good | Good | - | - | - | Good | - | - |
| demone456/kalshi-polymarket-bot | Good | Good | Good | Good | - | - | - | - | Good |
| ThinkEnigmatic/polymarket-bot-arena | - | Good | Good | - | Good | - | - | - | - |
| perpetual-s/polymarket-python-infrastructure | Good | Good | - | - | - | **Best** | Good | - | - |
| hbr-l/polypy | Good | - | - | - | - | - | - | - | - |
| Polymarket/go-order-utils | - | - | - | - | - | - | Good | **Best** | - |
| JonathanPetersonn/oracle-lag-sniper | - | Good | - | - | - | - | - | - | Good |
