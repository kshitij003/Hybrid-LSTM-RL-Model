package com.trader.trading_backend.controller;

import com.trader.trading_backend.dto.OrderRequestDTO;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import com.trader.trading_backend.service.MarketDataService;
import com.trader.trading_backend.service.ModelIntegrationService;
import com.trader.trading_backend.service.OrderExecutionService;
import com.trader.trading_backend.service.PortfolioService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/portfolio")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:3000") // Allow React Frontend
public class PortfolioController {

    private final PortfolioService portfolioService;
    private final ModelIntegrationService modelService;
    private final OrderExecutionService executionService;
    private final MarketDataService marketDataService; // For manual sync demo

    // 1. Get Portfolio Summary (For the Dashboard Card)
    @GetMapping("/{id}")
    public ResponseEntity<Portfolio> getPortfolio(@PathVariable Long id) {
        // In real app, verify User ID matches Token
        return ResponseEntity.ok(portfolioService.getPortfolioById(id));
    }

    // 2. Get Real-Time Net Worth (For the Big Green Number)
    @GetMapping("/{id}/nav")
    public ResponseEntity<BigDecimal> getNetAssetValue(@PathVariable Long id) {
        return ResponseEntity.ok(portfolioService.calculateNetAssetValue(id));
    }

    // 3. DEMO BUTTON: "Run AI Analysis Now"
    // Allows the professor/user to force a rebalance click
    @PostMapping("/{id}/rebalance")
    public ResponseEntity<String> forceRebalance(@PathVariable Long id) {
        Portfolio portfolio = portfolioService.getPortfolioById(id);

        // A. Ask AI for weights
        Map<String, Double> targets = modelService.getRebalancingSignals(portfolio);

        if (targets.isEmpty()) {
            return ResponseEntity.ok("AI suggests HOLDING current position.");
        }

        // B. Generate Orders
        List<OrderRequestDTO> orders = portfolioService.generateRebalancingOrders(id, targets);

        // C. Execute Orders
        int successCount = 0;
        for (OrderRequestDTO order : orders) {
            boolean done = executionService.executeOrder(order);
            if (done) successCount++;
        }

        return ResponseEntity.ok("Rebalancing Complete. Executed " + successCount + " trades.");
    }

    // 4. DEMO BUTTON: "Sync Market Data"
    // Useful if data looks stale during a presentation
    @PostMapping("/sync-data")
    public ResponseEntity<String> forceDataSync() {
        // Hardcoded list or fetch from DB
        marketDataService.syncstockprices(List.of("AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"));
        return ResponseEntity.ok("Market Data Synced Successfully.");
    }
}
