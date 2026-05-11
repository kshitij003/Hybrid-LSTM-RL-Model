package com.trader.trading_backend.controller;

import com.trader.trading_backend.service.NewsService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/news")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
public class NewsController {

    private final NewsService newsService;

    @PostMapping("/sync")
    public ResponseEntity<String> syncNews(@RequestParam String ticker, @RequestParam(defaultValue = "7") int days) {
        newsService.syncNewsBatch(List.of(ticker), days);
        return ResponseEntity.ok("News sync triggered for " + ticker);
    }

    @PostMapping("/sync-batch")
    public ResponseEntity<String> syncNewsBatch(@RequestBody Map<String, Object> request) {
        List<String> stocks = (List<String>) request.get("stocks");
        Integer days = (Integer) request.getOrDefault("days", 7);
        
        newsService.syncNewsBatch(stocks, days);
        return ResponseEntity.ok("Batch news sync triggered for " + stocks.size() + " stocks.");
    }
}
