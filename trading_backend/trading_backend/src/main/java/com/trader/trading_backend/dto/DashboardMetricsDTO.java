package com.trader.trading_backend.dto;

import lombok.Builder;
import lombok.Data;
import java.math.BigDecimal;
import java.util.List;

@Data
@Builder
public class DashboardMetricsDTO {
    private BigDecimal portfolioValue;
    private Double dayChange;
    private Double aiConfidence;
    private BigDecimal cashWeight;
    private List<ActiveStockDTO> activeStocks;

    @Data
    @Builder
    public static class ActiveStockDTO {
        private String ticker;
        private int shares;
        private Double weight;
    }
}
