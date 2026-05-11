package com.trader.trading_backend.controller;

import com.trader.trading_backend.dto.DashboardMetricsDTO;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Holding;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import com.trader.trading_backend.service.PortfolioService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/dashboard")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
public class DashboardController {

    private final PortfolioService portfolioService;
    
    // Global System State
    public static boolean isTradingEnabled = true;
    public static boolean isLiveTradingMode = false;

    @GetMapping("/metrics")
    public ResponseEntity<DashboardMetricsDTO> getMetrics() {
        // Default to Portfolio ID 1 for demonstration
        Long portfolioId = 1L;
        
        Portfolio portfolio = portfolioService.getPortfolioById(portfolioId);
        BigDecimal nav = portfolioService.calculateNetAssetValue(portfolioId);
        
        List<DashboardMetricsDTO.ActiveStockDTO> activeStocks = portfolio.getHoldings().stream()
                .map(holding -> {
                    // This is a simplification; in a real app, you'd fetch the current price here too
                    // For metrics, we estimate weight based on current holdings
                    return DashboardMetricsDTO.ActiveStockDTO.builder()
                            .ticker(holding.getStock().getTicker())
                            .shares(holding.getQuantity())
                            .weight(0.0) // We'll calculate weights below
                            .build();
                })
                .collect(Collectors.toList());

        // Calculate weights
        BigDecimal cashWeight = portfolio.getCurrentCashBalance().divide(nav.add(BigDecimal.valueOf(0.001)), 4, RoundingMode.HALF_UP);

        return ResponseEntity.ok(DashboardMetricsDTO.builder()
                .portfolioValue(nav)
                .dayChange(0.45) // Simulated day change
                .aiConfidence(0.88) // Simulated confidence
                .cashWeight(cashWeight)
                .activeStocks(activeStocks)
                .build());
    }

    @PostMapping("/control")
    public ResponseEntity<Map<String, String>> updateControl(@RequestBody Map<String, String> request) {
        if (request.containsKey("action")) {
            isTradingEnabled = request.get("action").equals("START");
            System.out.println("🤖 SYSTEM: Trading " + (isTradingEnabled ? "ENABLED" : "DISABLED"));
        }
        
        if (request.containsKey("mode")) {
            isLiveTradingMode = request.get("mode").equals("LIVE");
            System.out.println("⚠️ SYSTEM: Mode changed to " + (isLiveTradingMode ? "LIVE" : "SIMULATION"));
        }
        
        return ResponseEntity.ok(Map.of("status", "success"));
    }
}
