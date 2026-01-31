package com.trader.trading_backend.dto;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Builder
public class StockPriceDTO {
    private LocalDateTime timestamp;
    private Double open;
    private Double high;
    private Double low;
    private Double close;
    private Long volume;
}
