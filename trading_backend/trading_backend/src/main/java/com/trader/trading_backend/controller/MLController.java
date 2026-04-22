package com.trader.trading_backend.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

/**
 * MLController — thin proxy between the frontend and the Python ML service.
 *
 * All training is triggered directly on the ML service (which fetches its own
 * data from yfinance). Spring Boot no longer exports DB data for training.
 *
 * Endpoints:
 *   POST /api/ml/train/full          → trigger full LSTM + PPO retrain
 *   POST /api/ml/train/quick-update  → fine-tune existing PPO (fast, ~20 min)
 *   GET  /api/ml/training-status/{id}→ poll job progress
 *   GET  /api/ml/training-jobs       → list all training jobs
 *   GET  /api/ml/health              → check if ML service is reachable
 */
@RestController
@RequestMapping("/api/ml")
@RequiredArgsConstructor
@CrossOrigin(origins = "${allowed.origin:http://localhost:3000}")
public class MLController {

    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.service.url:http://localhost:8000}")
    private String mlServiceUrl;

    // Default Indian Nifty 50 training universe
    private static final List<String> DEFAULT_STOCKS = List.of(
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"
    );

    // ─────────────────────────────────────────────────────────────────────────
    //  POST /api/ml/train/full
    //  Trigger a full LSTM + PPO retrain from scratch via yfinance data.
    //
    //  Body (all optional — sensible defaults used if omitted):
    //  {
    //    "stocks":    ["RELIANCE.NS", "TCS.NS", ...],
    //    "startDate": "2015-01-01",
    //    "endDate":   "2025-01-01",
    //    "config": {
    //      "lstmEpochs":    20,
    //      "ppoTimesteps":  500000,
    //      "initialBalance": 100000
    //    }
    //  }
    // ─────────────────────────────────────────────────────────────────────────
    @PostMapping("/train/full")
    public ResponseEntity<Map> triggerFullTraining(@RequestBody(required = false) Map<String, Object> body) {
        try {
            if (body == null) body = Map.of();

            List<String> stocks = body.containsKey("stocks")
                    ? (List<String>) body.get("stocks")
                    : DEFAULT_STOCKS;

            String startDate = (String) body.getOrDefault("startDate", "2015-01-01");
            String endDate   = (String) body.getOrDefault("endDate",   LocalDate.now().toString());

            @SuppressWarnings("unchecked")
            Map<String, Object> config = body.containsKey("config")
                    ? (Map<String, Object>) body.get("config")
                    : Map.of("lstmEpochs", 20, "ppoTimesteps", 500000, "initialBalance", 100000);

            Map<String, Object> payload = Map.of(
                    "stocks",    stocks,
                    "startDate", startDate,
                    "endDate",   endDate,
                    "config",    config
            );

            System.out.println("🚀 Triggering full ML retrain for: " + stocks);
            ResponseEntity<Map> response = restTemplate.postForEntity(
                    mlServiceUrl + "/api/train/multi-stock", payload, Map.class);

            System.out.println("✅ Full retrain initiated.");
            return ResponseEntity.accepted().body(response.getBody());

        } catch (Exception e) {
            System.err.println("❌ Full retrain trigger failed: " + e.getMessage());
            return ResponseEntity.status(502)
                    .body(Map.of("error", e.getMessage(), "status", "FAILED"));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  POST /api/ml/train/quick-update
    //  Fine-tune the existing PPO on recent data (~20 min, no LSTM retraining).
    //
    //  Body (all optional):
    //  {
    //    "stocks":    ["RELIANCE.NS", "TCS.NS", ...],
    //    "startDate": "2024-01-01",
    //    "endDate":   "2025-01-01",
    //    "config": { "ppoTimesteps": 50000, "initialBalance": 100000 }
    //  }
    // ─────────────────────────────────────────────────────────────────────────
    @PostMapping("/train/quick-update")
    public ResponseEntity<Map> triggerQuickUpdate(@RequestBody(required = false) Map<String, Object> body) {
        try {
            if (body == null) body = Map.of();

            List<String> stocks = body.containsKey("stocks")
                    ? (List<String>) body.get("stocks")
                    : DEFAULT_STOCKS;

            String startDate = (String) body.getOrDefault("startDate",
                    LocalDate.now().minusMonths(12).toString());
            String endDate = (String) body.getOrDefault("endDate",
                    LocalDate.now().toString());

            @SuppressWarnings("unchecked")
            Map<String, Object> config = body.containsKey("config")
                    ? (Map<String, Object>) body.get("config")
                    : Map.of("ppoTimesteps", 50000, "initialBalance", 100000);

            Map<String, Object> payload = Map.of(
                    "stocks",    stocks,
                    "startDate", startDate,
                    "endDate",   endDate,
                    "config",    config
            );

            System.out.println("⚡ Triggering quick PPO update for: " + stocks);
            ResponseEntity<Map> response = restTemplate.postForEntity(
                    mlServiceUrl + "/api/train/quick-update", payload, Map.class);

            System.out.println("✅ Quick update initiated.");
            return ResponseEntity.accepted().body(response.getBody());

        } catch (Exception e) {
            System.err.println("❌ Quick update trigger failed: " + e.getMessage());
            return ResponseEntity.status(502)
                    .body(Map.of("error", e.getMessage(), "status", "FAILED"));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  GET /api/ml/training-status/{jobId}
    //  Poll the progress of any training job.
    // ─────────────────────────────────────────────────────────────────────────
    @GetMapping("/training-status/{jobId}")
    public ResponseEntity<Map> getTrainingStatus(@PathVariable String jobId) {
        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(
                    mlServiceUrl + "/api/train/status/" + jobId, Map.class);
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(502)
                    .body(Map.of("error", e.getMessage()));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  GET /api/ml/training-jobs
    //  List all training jobs (newest first).
    // ─────────────────────────────────────────────────────────────────────────
    @GetMapping("/training-jobs")
    public ResponseEntity<Map> listTrainingJobs() {
        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(
                    mlServiceUrl + "/api/train/list", Map.class);
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(502)
                    .body(Map.of("error", e.getMessage()));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  GET /api/ml/health
    //  Check if the Python ML service is reachable.
    // ─────────────────────────────────────────────────────────────────────────
    @GetMapping("/health")
    public ResponseEntity<Map> mlServiceHealth() {
        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(
                    mlServiceUrl + "/api/health", Map.class);
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(503).body(Map.of(
                    "status",  "ML_SERVICE_UNREACHABLE",
                    "url",     mlServiceUrl,
                    "error",   e.getMessage()
            ));
        }
    }
}
