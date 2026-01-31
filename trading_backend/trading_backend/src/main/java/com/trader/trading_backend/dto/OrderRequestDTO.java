package com.trader.trading_backend.dto;

import com.trader.trading_backend.Enum.TransactionType;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class OrderRequestDTO {
    private Long portfolioId;
    private String ticker;
    private TransactionType type; // BUY or SELL
    private Integer quantity;
    private String reason; // "AI_REBALANCE" or "USER_MANUAL"
}
