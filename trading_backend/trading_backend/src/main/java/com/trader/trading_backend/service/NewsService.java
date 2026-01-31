package com.trader.trading_backend.service;

import com.trader.trading_backend.Enum.SentimentLabel;
import com.trader.trading_backend.Repository.Market_Data_Management.MarketNewsRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.entity.Market_Data_Management.MarketNews;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class NewsService {

    private final MarketNewsRepository newsRepository;
    private final StockRepository stockRepository;
    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.service.url:http://localhost:8000}")
    private String mlServiceUrl;

    /**
     * Fetch and store news for a stock ticker
     * 
     * @param ticker Stock symbol (e.g., "AAPL")
     * @param days Number of days to look back
     * @return Number of articles stored
     */
    public int fetchAndStoreNews(String ticker, int days) {
        System.out.println("📰 Fetching news for " + ticker + " (last " + days + " days)");

        try {
            // 1. Get stock entity
            Stock stock = stockRepository.findByTickerIgnoreCase(ticker)
                    .orElseThrow(() -> new RuntimeException("Stock not found: " + ticker));

            // 2. Call ML service to fetch news with sentiment
            String url = mlServiceUrl + "/api/news/fetch";
            Map<String, Object> request = Map.of(
                    "ticker", ticker,
                    "days", days
            );

            ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
            Map<String, Object> result = response.getBody();

            if (result == null) {
                System.err.println("⚠️  No response from news service");
                return 0;
            }

            // 3. Extract articles
            List<Map<String, Object>> articles = (List<Map<String, Object>>) result.get("articles");

            if (articles == null || articles.isEmpty()) {
                System.out.println("   No news found for " + ticker);
                return 0;
            }

            // 4. Store in database
            List<MarketNews> newsEntities = new ArrayList<>();

            for (Map<String, Object> article : articles) {
                try {
                    // Extract sentiment data
                    Map<String, Object> sentimentData = (Map<String, Object>) article.get("sentiment");
                    Double sentimentScore = ((Number) sentimentData.get("score")).doubleValue();
                    String sentimentLabelStr = (String) sentimentData.get("label");

                    // Convert string to enum
                    SentimentLabel label;
                    switch (sentimentLabelStr.toLowerCase()) {
                        case "positive":
                            label = SentimentLabel.POSITIVE;
                            break;
                        case "negative":
                            label = SentimentLabel.NEGATIVE;
                            break;
                        default:
                            label = SentimentLabel.NEUTRAL;
                    }

                    // Parse published date
                    String publishedAtStr = (String) article.get("publishedAt");
                    LocalDateTime publishedAt = LocalDateTime.parse(
                            publishedAtStr,
                            DateTimeFormatter.ISO_DATE_TIME
                    );

                    // Create entity
                    MarketNews news = MarketNews.builder()
                            .stock(stock)
                            .headline((String) article.get("title"))
                            .source((String) article.get("source"))
                            .publishedAt(publishedAt)
                            .sentimentScore(sentimentScore)
                            .sentimentLabel(label)
                            .build();

                    newsEntities.add(news);

                } catch (Exception e) {
                    System.err.println("   ⚠️  Skipping article due to error: " + e.getMessage());
                }
            }

            // 5. Save all news entities
            newsRepository.saveAll(newsEntities);

            System.out.println("   ✅ Stored " + newsEntities.size() + " articles for " + ticker);
            System.out.println("   Average sentiment: " + result.get("averageSentiment"));

            return newsEntities.size();

        } catch (Exception e) {
            System.err.println("❌ News fetch failed for " + ticker + ": " + e.getMessage());
            e.printStackTrace();
            return 0;
        }
    }

    /**
     * Fetch news for multiple stocks
     * 
     * @param tickers List of stock symbols
     * @param days Number of days to look back
     * @return Total number of articles stored
     */
    public int fetchNewsForMultipleStocks(List<String> tickers, int days) {
        System.out.println("📰 Fetching news for " + tickers.size() + " stocks...");

        int totalArticles = 0;

        for (String ticker : tickers) {
            int count = fetchAndStoreNews(ticker, days);
            totalArticles += count;
        }

        System.out.println("✅ Total articles stored: " + totalArticles);
        return totalArticles;
    }

    /**
     * Get latest sentiment score for a stock
     * (Used by ModelIntegrationService)
     */
    public Double getLatestSentiment(String ticker) {
        Stock stock = stockRepository.findByTickerIgnoreCase(ticker).orElse(null);

        if (stock == null) {
            return 0.0;
        }

        // Get average sentiment from recent news (last 24 hours)
        LocalDateTime yesterday = LocalDateTime.now().minusDays(1);

        Double avgSentiment = newsRepository
                .findAverageSentimentByStockIdAndPublishedAt(stock.getId(), yesterday.toLocalDate());

        return avgSentiment != null ? avgSentiment : 0.0;
    }
}
