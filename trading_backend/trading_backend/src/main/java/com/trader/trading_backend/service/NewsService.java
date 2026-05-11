package com.trader.trading_backend.service;

import com.trader.trading_backend.Repository.Market_Data_Management.MarketNewsRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.entity.Market_Data_Management.MarketNews;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class NewsService {

    private final StockRepository stockRepo;
    private final MarketNewsRepository newsRepo;
    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.service.url:http://localhost:8000}")
    private String mlServiceUrl;

    /**
     * Sync news for multiple stocks by calling the ML Service.
     * This ensures we get REAL sentiment scores from FinBERT.
     */
    @Transactional
    public void syncNewsBatch(List<String> tickers, int days) {
        for (String ticker : tickers) {
            try {
                Stock stock = stockRepo.findByTickerIgnoreCase(ticker)
                        .orElseGet(() -> stockRepo.save(Stock.builder().ticker(ticker.toUpperCase()).build()));

                // Call ML Service to fetch news with sentiment
                String url = mlServiceUrl + "/api/news/sentiment?ticker=" + ticker;
                Map<String, Object> response = restTemplate.getForObject(url, Map.class);

                if (response != null && response.containsKey("latest_headlines")) {
                    List<String> headlines = (List<String>) response.get("latest_headlines");
                    Double sentimentScore = (Double) response.get("sentiment_score");

                    for (String headline : headlines) {
                        // Check if headline already exists to avoid duplicates
                        // (Simplified for demo; real app would use a URL or unique hash)
                        
                        MarketNews news = MarketNews.builder()
                                .stock(stock)
                                .headline(headline)
                                .sentimentScore(sentimentScore)
                                .publishedAt(LocalDateTime.now())
                                .source("ML_SERVICE_SYNC")
                                .build();
                        
                        newsRepo.save(news);
                    }
                    System.out.println("✅ Synced " + headlines.size() + " news items for " + ticker + " | Sentiment: " + sentimentScore);
                }

            } catch (Exception e) {
                System.err.println("❌ Failed to sync news for " + ticker + ": " + e.getMessage());
            }
        }
    }
}
