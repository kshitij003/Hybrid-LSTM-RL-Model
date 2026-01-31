package com.trader.trading_backend.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class StockFeatureDTO {
    private String date;
    private Double close;
    private Double volume;
    private Double sentimentScore; // Merged from News or Summary
}
