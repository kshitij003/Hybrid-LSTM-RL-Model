package com.trader.trading_backend.controller;

import com.trader.trading_backend.service.MarketDataService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/market")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:3000")
public class MarketController {

    private final MarketDataService marketDataService;

    // Get graph data for the UI
    @GetMapping("/history")
    public ResponseEntity<Map<String, Object>> getStockHistory(@RequestParam List<String> tickers) {
        // Re-use your existing service method!
        return ResponseEntity.ok(marketDataService.getLatestFeatures(tickers, 90));
    }
}
