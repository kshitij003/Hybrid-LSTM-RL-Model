package com.trader.trading_backend.service;

import com.trader.trading_backend.Enum.TransactionType;
import com.trader.trading_backend.Repository.Market_Data_Management.StockPriceRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.Repository.Portfolio_Trading_Engine.PortfolioRepository;
import com.trader.trading_backend.dto.OrderRequestDTO;
import com.trader.trading_backend.dto.StockPriceDTO;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import com.trader.trading_backend.entity.Market_Data_Management.StockPrice;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Holding;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class PortfolioService {

    private final PortfolioRepository portfolioRepo;
    private final StockRepository stockRepo;
    private final StockPriceRepository priceRepo;

    // Minimum trade value to avoid spamming tiny orders ($10)
    private static final BigDecimal MIN_TRADE_THRESHOLD = BigDecimal.valueOf(10.0);

    /**
     * TASK 1: Calculate Net Asset Value (NAV)
     * Returns: Cash + (Shares * Current Market Price)
     */
    @Transactional(readOnly = true)
    public BigDecimal calculateNetAssetValue(Long portfolioId) {
        Portfolio portfolio = portfolioRepo.findById(portfolioId)
                .orElseThrow(() -> new RuntimeException("Portfolio not found"));

        BigDecimal totalHoldingsValue = BigDecimal.ZERO;

        for (Holding holding : portfolio.getHoldings()) {
            BigDecimal currentPrice = getCurrentPrice(holding.getStock());

            // Value = Qty * Price
            BigDecimal holdingValue = currentPrice.multiply(BigDecimal.valueOf(holding.getQuantity()));
            totalHoldingsValue = totalHoldingsValue.add(holdingValue);
        }

        return portfolio.getCurrentCashBalance().add(totalHoldingsValue);
    }

    /**
     * TASK 2: Generate Orders (The "Diff Engine")
     * Converts AI Target Weights (e.g. 20%) -> Actual Buy/Sell Orders
     */
    @Transactional
    public List<OrderRequestDTO> generateRebalancingOrders(Long portfolioId, Map<String, Double> targetWeights) {
        Portfolio portfolio = portfolioRepo.findById(portfolioId)
                .orElseThrow(() -> new RuntimeException("Portfolio not found"));

        BigDecimal totalPortfolioValue = calculateNetAssetValue(portfolioId);
        List<OrderRequestDTO> orders = new ArrayList<>();

        // 1. Create a map of current holdings for easy lookup
        Map<String, Holding> currentHoldingsMap = portfolio.getHoldings().stream()
                .collect(Collectors.toMap(h -> h.getStock().getTicker(), h -> h));

        // 2. Iterate through AI Targets
        for (Map.Entry<String, Double> entry : targetWeights.entrySet()) {
            String ticker = entry.getKey();
            Double targetWeight = entry.getValue();

            // Skip "CASH" placeholder (AI might output CASH weight)
            if (ticker.equalsIgnoreCase("CASH")) continue;

            Stock stock = stockRepo.findByTickerIgnoreCase(ticker)
                    .orElseThrow(() -> new RuntimeException("Stock not found: " + ticker));

            BigDecimal currentPrice = getCurrentPrice(stock);

            // A. Calculate Target Value in $$$
            // Target $ = TotalValue * Target%
            BigDecimal targetValue = totalPortfolioValue.multiply(BigDecimal.valueOf(targetWeight));

            // B. Calculate Current Value in $$$
            BigDecimal currentValue = BigDecimal.ZERO;
            if (currentHoldingsMap.containsKey(ticker)) {
                currentValue = currentPrice.multiply(BigDecimal.valueOf(currentHoldingsMap.get(ticker).getQuantity()));
            }

            // C. Calculate the Difference
            BigDecimal difference = targetValue.subtract(currentValue);

            // D. Generate Order if diff is significant (> $10)
            if (difference.abs().compareTo(MIN_TRADE_THRESHOLD) > 0) {
                int quantity = difference.divide(currentPrice, 0, RoundingMode.DOWN).intValue();

                if (quantity != 0) {
                    orders.add(OrderRequestDTO.builder()
                            .portfolioId(portfolioId)
                            .ticker(ticker)
                            .type(quantity > 0 ? TransactionType.BUY : TransactionType.SELL)
                            .quantity(Math.abs(quantity))
                            .reason("AI_REBALANCE")
                            .build());
                }
            }
        }

        // 3. Risk Validation (Optional Circuit Breaker)
        validateRisk(portfolio, orders);

        return orders;
    }

    /**
     * TASK 3: Risk Validation
     * Prevents the AI from doing something illegal (like spending money we don't have).
     * Note: Detailed execution checks happen in ExecutionService, this is a high-level sanity check.
     */
    private void validateRisk(Portfolio portfolio, List<OrderRequestDTO> orders) {
        BigDecimal projectedCash = portfolio.getCurrentCashBalance();

        for (OrderRequestDTO order : orders) {
            Stock stock = stockRepo.findByTickerIgnoreCase(order.getTicker()).orElseThrow();
            BigDecimal price = getCurrentPrice(stock);
            BigDecimal cost = price.multiply(BigDecimal.valueOf(order.getQuantity()));

            if (order.getType() == TransactionType.BUY) {
                projectedCash = projectedCash.subtract(cost);
            } else {
                projectedCash = projectedCash.add(cost);
            }
        }

        if (projectedCash.compareTo(BigDecimal.ZERO) < 0) {
            System.out.println("⚠️ RISK ALERT: AI attempted to overspend! Order batch rejected for Portfolio {}"+portfolio.getId());
            orders.clear(); // Reject ALL orders to be safe
        }
    }

    // Helper: Get latest price efficiently (Top 1)
    private BigDecimal getCurrentPrice(Stock stock) {
        List<StockPrice> latest = priceRepo.findByStockIdOrderByTimestampDesc(stock.getId(), PageRequest.of(0, 1));
        if (latest.isEmpty()) {
            throw new RuntimeException("No price data found for " + stock.getTicker());
        }
        return latest.get(0).getClosePrice(); // Ensure StockPrice entity has getClosePrice()
    }

    public Portfolio getPortfolioById(Long id) {
        return portfolioRepo.findById(id).orElseThrow(() -> new RuntimeException("Portfolio not found"));
    }
}
