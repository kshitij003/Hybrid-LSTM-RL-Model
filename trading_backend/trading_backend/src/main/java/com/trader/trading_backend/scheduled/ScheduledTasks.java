package com.trader.trading_backend.scheduled;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

/**
 * ScheduledTasks — periodic automation jobs.
 *
 * Removed:
 *   ❌ weeklyDataSync()   — market data is fetched on-demand by yfinance in ML service
 *   ❌ dailyNewsFetch()   — news + sentiment fetched on-demand per predict call
 *
 * Kept:
 *   ✅ quarterlyModelRetraining() — triggers full LSTM+PPO retrain every 3 months
 *   ✅ monthlyQuickUpdate()       — fine-tunes PPO on fresh data monthly (fast)
 */
@Component
@RequiredArgsConstructor
public class ScheduledTasks {

    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.service.url:http://localhost:8000}")
    private String mlServiceUrl;

    // Default Indian Nifty 50 stock universe for scheduled retraining.
    // Override via application.properties: ai.training.stocks
    private static final List<String> TRAINING_STOCKS = List.of(
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"
    );

    // ─────────────────────────────────────────────────────────────────────────
    //  Quarterly: Full retrain (LSTM + PPO from scratch)
    //  Runs: Jan 1, Apr 1, Jul 1, Oct 1 at 03:00 AM
    // ─────────────────────────────────────────────────────────────────────────
    @Scheduled(cron = "0 0 3 1 1,4,7,10 *")
    public void quarterlyModelRetraining() {
        System.out.println("==============================================");
        System.out.println("🤖 SCHEDULED: Quarterly Full Retrain");
        System.out.println("   Stocks : " + TRAINING_STOCKS);
        System.out.println("   Data   : 2015-01-01 → today (via yfinance)");
        System.out.println("==============================================");

        try {
            String url = mlServiceUrl + "/api/train/multi-stock";

            Map<String, Object> request = Map.of(
                    "stocks",    TRAINING_STOCKS,
                    "startDate", "2015-01-01",
                    "endDate",   java.time.LocalDate.now().toString(),
                    "config",    Map.of(
                            "lstmEpochs",    20,
                            "ppoTimesteps",  500000,
                            "initialBalance", 100000
                    )
            );

            ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
            Map<?, ?> result = response.getBody();

            System.out.println("✅ Full retrain initiated.");
            System.out.println("   Training ID : " + (result != null ? result.get("trainingId") : "unknown"));
            System.out.println("   Poll at     : GET /api/train/status/<id>");

        } catch (Exception e) {
            System.err.println("❌ Quarterly retraining failed: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("==============================================\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Monthly: Quick PPO fine-tune (skips LSTM retraining, fast ~20 min)
    //  Runs: 1st of every month at 02:00 AM (skips quarterly months)
    // ─────────────────────────────────────────────────────────────────────────
    @Scheduled(cron = "0 0 2 1 2,3,5,6,8,9,11,12 *")
    public void monthlyQuickUpdate() {
        System.out.println("==============================================");
        System.out.println("⚡ SCHEDULED: Monthly Quick PPO Update");
        System.out.println("   Stocks : " + TRAINING_STOCKS);
        System.out.println("   Steps  : 50,000 fine-tune timesteps");
        System.out.println("==============================================");

        try {
            String url = mlServiceUrl + "/api/train/quick-update";

            // Use last 12 months of data for the environment during fine-tuning
            String startDate = java.time.LocalDate.now().minusMonths(12).toString();
            String endDate   = java.time.LocalDate.now().toString();

            Map<String, Object> request = Map.of(
                    "stocks",    TRAINING_STOCKS,
                    "startDate", startDate,
                    "endDate",   endDate,
                    "config",    Map.of(
                            "ppoTimesteps",   50000,
                            "initialBalance", 100000
                    )
            );

            ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
            Map<?, ?> result = response.getBody();

            System.out.println("✅ Quick update initiated.");
            System.out.println("   Training ID : " + (result != null ? result.get("trainingId") : "unknown"));

        } catch (Exception e) {
            System.err.println("❌ Monthly quick update failed: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("==============================================\n");
    }
}
