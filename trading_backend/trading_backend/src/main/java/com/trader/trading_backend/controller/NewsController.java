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
@CrossOrigin(origins = "http://localhost:3000")
public class NewsController {

    private final NewsService newsService;

    /**
     * Fetch and store news for a single stock
     * POST /api/news/sync?ticker=AAPL&days=7
     */
    @PostMapping("/sync")
    public ResponseEntity<Map<String, Object>> syncNews(
            @RequestParam String ticker,
            @RequestParam(defaultValue = "7") int days
    ) {
        int articlesStored = newsService.fetchAndStoreNews(ticker, days);

        return ResponseEntity.ok(Map.of(
                "ticker", ticker,
                "articlesStored", articlesStored,
                "days", days,
                "message", "News synced successfully"
        ));
    }

    /**
     * Fetch and store news for multiple stocks
     * POST /api/news/sync-batch
     * Body: {
     *   "stocks": ["AAPL", "MSFT", "GOOGL"],
     *   "days": 7
     * }
     */
    @PostMapping("/sync-batch")
    public ResponseEntity<Map<String, Object>> syncBatchNews(@RequestBody Map<String, Object> request) {
        List<String> stocks = (List<String>) request.get("stocks");
        Integer days = (Integer) request.getOrDefault("days", 7);

        if (stocks == null || stocks.isEmpty()) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "stocks list is required"));
        }

        int totalArticles = newsService.fetchNewsForMultipleStocks(stocks, days);

        return ResponseEntity.ok(Map.of(
                "stocks", stocks,
                "totalArticlesStored", totalArticles,
                "days", days,
                "message", "Batch news sync completed"
        ));
    }

    /**
     * Get latest sentiment for a stock
     * GET /api/news/sentiment?ticker=AAPL
     */
    @GetMapping("/sentiment")
    public ResponseEntity<Map<String, Object>> getLatestSentiment(@RequestParam String ticker) {
        Double sentiment = newsService.getLatestSentiment(ticker);

        return ResponseEntity.ok(Map.of(
                "ticker", ticker,
                "sentiment", sentiment,
                "timestamp", java.time.LocalDateTime.now()
        ));
    }
}
