package com.trader.trading_backend.scheduled;

import com.trader.trading_backend.service.DataMaintenanceService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class ScheduledTasks {

    private final DataMaintenanceService dataMaintenanceService;
    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.service.url:http://localhost:8000}")
    private String mlServiceUrl;

    // List of stocks to train on
    private static final List<String> TRAINING_STOCKS = List.of(
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"
    );

    /**
     * DATA MAINTENANCE: Every 30 days
     * Compresses old news data to save storage
     * 
     * Runs on the 1st of every month at 2:00 AM
     */
    @Scheduled(cron = "0 0 2 1 * *")  // Monthly: 1st day, 2 AM
    public void monthlyDataMaintenance() {
        System.out.println("======================================");
        System.out.println("🗄️  SCHEDULED: Monthly Data Maintenance");
        System.out.println("======================================");

        try {
            dataMaintenanceService.compressoldnews();
            System.out.println("✅ Data maintenance completed successfully");
        } catch (Exception e) {
            System.err.println("❌ Data maintenance failed: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("======================================\n");
    }

    /**
     * MODEL RETRAINING: Every 3 months
     * Trains LSTM+RL model with last 90 days of data from PostgreSQL
     * 
     * Runs on: Jan 1, Apr 1, Jul 1, Oct 1 at 3:00 AM
     */
    @Scheduled(cron = "0 0 3 1 1,4,7,10 *")  // Quarterly: Jan/Apr/Jul/Oct 1st, 3 AM
    public void quarterlyModelRetraining() {
        System.out.println("======================================");
        System.out.println("🤖 SCHEDULED: Quarterly Model Retraining");
        System.out.println("   Stocks: " + TRAINING_STOCKS);
        System.out.println("   Data: Last 90 days");
        System.out.println("======================================");

        try {
            // Trigger training via ML service
            String url = mlServiceUrl + "/api/ml/trigger-training";

            Map<String, Object> request = Map.of(
                    "stocks", TRAINING_STOCKS,
                    "daysOfData", 90,
                    "lstmEpochs", 20,
                    "ppoTimesteps", 500000
            );

            System.out.println("📤 Sending training request to: " + url);

            ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
            Map<String, Object> result = response.getBody();

            System.out.println("✅ Training initiated: " + result.get("trainingId"));
            System.out.println("   Monitor at: /api/ml/training-status/" + result.get("trainingId"));

        } catch (Exception e) {
            System.err.println("❌ Model retraining failed: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("======================================\n");
    }

    /**
     * OPTIONAL: Weekly data sync
     * Keeps market data up to date
     * 
     * Runs every Sunday at 1:00 AM
     */
    @Scheduled(cron = "0 0 1 * * SUN")  // Weekly: Sunday 1 AM
    public void weeklyDataSync() {
        System.out.println("======================================");
        System.out.println("📊 SCHEDULED: Weekly Data Sync");
        System.out.println("======================================");

        try {
            // This would call MarketDataService.syncstockprices()
            // For now, just logging
            System.out.println("⏭️  Skipped: Manual trigger preferred");
            // Uncomment when ready to automate:
            // marketDataService.syncstockprices(TRAINING_STOCKS);

        } catch (Exception e) {
            System.err.println("❌ Data sync failed: " + e.getMessage());
        }

        System.out.println("======================================\n");
    }
    
    /**
     * DAILY NEWS SYNC: Every day at midnight
     * Fetches latest news and sentiment for all stocks
     * 
     * Runs daily at 12:00 AM
     */
    @Scheduled(cron = "0 0 0 * * *")  // Daily: Midnight
    public void dailyNewsFetch() {
        System.out.println("======================================");
        System.out.println("📰 SCHEDULED: Daily News & Sentiment Sync");
        System.out.println("======================================");

        try {
            // This would call NewsService.fetchNewsForMultipleStocks()
            // For now, just logging
            System.out.println("⏭️  News fetching configured");
            System.out.println("   Stocks: " + TRAINING_STOCKS);
            System.out.println("   Last 1 day of news");
            
            // Uncomment when ready to activate:
            // newsService.fetchNewsForMultipleStocks(TRAINING_STOCKS, 1);

        } catch (Exception e) {
            System.err.println("❌ News fetch failed: " + e.getMessage());
        }

        System.out.println("======================================\n");
    }
}
