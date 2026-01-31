package com.trader.trading_backend.dto;

import lombok.Builder;
import lombok.Data;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

@Data
@Builder
public class InferenceRequestDTO {
    // 1. Portfolio Context (The "State")
    private BigDecimal currentCash;
    private Map<String, Double> currentHoldings; // e.g. {"AAPL": 0.1, "GOOG": 0.2}

    // 2. Market Context (The "Features")
    // Map<Ticker, List<DayData>>
    private Map<String, List<StockFeatureDTO>> marketData;
}
