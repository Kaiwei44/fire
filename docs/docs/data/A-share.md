---
title: A-Share Data
permalink: /data/a-share/
parent: Data Management
nav_order: 4.1
---

# A-Share Data

Fire provides comprehensive data for the Chinese A-share market, including historical prices, financial 
statements, and other relevant information. This section outlines the available datasets and how to access them.

## Available Datasets

### Historical Prices

- **Daily Prices**: Contains daily open, high, low, close, money, vwap and volume data.
- **Daily Valuations**: Provides daily valuation metrics such as P/E ratio, P/B ratio, etc


数据范围：
2015.01.01-2025.05.01

数据内容分类：

1. quote：量价数据
字段数量：28 个
数据类型：DataFrame
数据规模：(2509 行，5363 列)

|字段|中文名|
|---|---|
|open|后复权开盘价——日级|
|close|后复权收盘价——日级|
|high|后复权最高价——日级|
|low|后复权最低价——日级|
|volume|成交量(股/份)——日级|
|money|成交额(元)——日级|
|return_adj|涨跌幅——日级|
|vwap|成交量加权均价——日级|
|adj_factor|复权因子|
|open_dr|原始开盘价——日级|
|high_dr|原始最高价——日级|
|low_dr|原始最低价——日级|
|close_dr|原始收盘价——日级|
|volume_dr|成交量(股/份)——日级|
|vwap_dr|成交量加权均价——日级|
|FinanceValue|融资余额(元)|
|FinanceBuyValue|融资买入额(元)|
|FinanceRefundValue|融资偿还额(元)|
|SecurityVolume|融券余量(股)|
|SecuritySellVolume|融券卖出量(股)|
|SecurityRefundVolume|融券偿还量(股)|
|SecurityValue|融券余额(元)|
|TradingValue|融资融券余额(元)|
|FinaInTotalRatio|融资占交易所融资余额比(%)|
|SecuInTotalRatio|融券占交易所融券余额比(%)|
|shares_holding|持股数量(股)|
|hold_ratio|持股占比(%)|
|adjusted_hold_ratio|调整后的持股占比(%)|


2. valuation：估值数据
字段数量：14 个
数据类型：DataFrame
数据规模：(2509 行，5363 列)

|字段|中文名|
|---|---|
|circulating_market_cap|流通市值(亿元)(含港股)|
|pcf_ratio|市现率(PCF, 现金净流量TTM)|
|market_cap|总市值(亿元)(含港股)|
|pe_ratio_lyr|静态市盈率(PE)|
|circulating_cap|流通股本(万股)(含港股)|
|capitalization|总股本(万股)(含港股)|
|pb_ratio|市净率(PB)|
|pe_ratio|市盈率(PE, TTM)|
|ps_ratio|市销率(PS, TTM)|
|turnover_ratio|换手率(%)|
|circulating_market_cap_ashare|A股流通市值（亿元）|
|market_cap_ashare|A股总市值（亿元）|
|circulating_cap_ashare|A股流通股本（万股）|
|capitalization_ashare|A股总股本（万股）|


3. financial：财务数据
字段数量：11 个
数据类型：DataFrame
数据规模：(2509 行，5363 列)

|字段|中文名|
|---|---|
|inventories|存货(元)|
|total_current_assets|流动资产合计(元)|
|fixed_assets|固定资产(元)|
|good_will|商誉(元)|
|total_assets|资产总计(元)|
|total_liability|负债合计(元)|
|operating_revenue|营业收入(元)|
|operating_profit|营业利润(元)|
|total_profit|利润总额(元)|
|net_profit|净利润(元)|
|basic_eps|基本每股收益(元)|

