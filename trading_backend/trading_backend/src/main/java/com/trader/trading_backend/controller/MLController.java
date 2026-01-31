package com.trader.trading_backend.controller;

import com.trader.trading_backend.dto.TrainingDataDTO;
import com.trader.trading_backend.service.MarketDataService;
import com.trader.trading_backend.service.ModelIntegrationService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/ml")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:3000")
public class MLController {

    private final MarketDataService marketDataService;
    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.service.url:http://localhost:8000}")
    private String mlServiceUrl;

    /**
     * Export training data from PostgreSQL
     * GET /api/ml/export-data?stocks=AAPL,MSFT&days=90
     */
    @GetMapping("/export-data")
    public ResponseEntity<TrainingDataDTO> exportData(
            @RequestParam List<String> stocks,
            @RequestParam(defaultValue = "90") int days
    ) {
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(days);

        TrainingDataDTO data = marketDataService.exportTrainingData(
                stocks,
                startDate,
                endDate
        );

        return ResponseEntity.ok(data);
    }

    /**
     * Trigger ML model training with DB data
     * POST /api/ml/trigger-training
     */
    @PostMapping("/trigger-training")
    public ResponseEntity<Map<String, Object>> triggerTraining(
            @RequestParam(required = false) List<String> stocks,
            @RequestParam(defaultValue = "90") int daysOfData,
            @RequestParam(defaultValue = "20") int lstmEpochs,
            @RequestParam(defaultValue = "500000") int ppoTimesteps
    ) {
        try {
            // Default stocks if not provided
            if (stocks == null || stocks.isEmpty()) {
                stocks = List.of("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA");
            }

            System.out.println("🚀 Triggering ML training with " + daysOfData + " days of data for: " + stocks);

            // 1. Export data from PostgreSQL
            LocalDate endDate = LocalDate.now();
            LocalDate startDate = endDate.minusDays(daysOfData);

            TrainingDataDTO trainingData = marketDataService.exportTrainingData(
                    stocks,
                    startDate,
                    endDate
            );

            // 2. Add training configuration
            trainingData.setConfig(TrainingDataDTO.TrainingConfig.builder()
                    .lstmEpochs(lstmEpochs)
                    .ppoTimesteps(ppoTimesteps)
                    .sequenceLength(60)
                    .initialBalance(10000.0)
                    .build());

            // 3. Send to Flask ML service
            String url = mlServiceUrl + "/api/train/from-db";
            System.out.println("📤 Sending data to: " + url);

            ResponseEntity<Map> response = restTemplate.postForEntity(
                    url,
                    trainingData,
                    Map.class
            );

            Map<String, Object> result = new HashMap<>(response.getBody());
            result.put("dataSource", "PostgreSQL");
            result.put("daysOfData", daysOfData);
            result.put("stocks", stocks);

            System.out.println("✅ Training initiated: " + result.get("trainingId"));

            return ResponseEntity.ok(result);

        } catch (Exception e) {
            System.err.println("❌ Training trigger failed: " + e.getMessage());
            e.printStackTrace();

            return ResponseEntity.status(500)
                    .body(Map.of(
                            "error", e.getMessage(),
                            "status", "FAILED"
                    ));
        }
    }

    /**
     * Check training status
     * GET /api/ml/training-status/{jobId}
     */
    @GetMapping("/training-status/{jobId}")
    public ResponseEntity<Map<String, Object>> getTrainingStatus(@PathVariable String jobId) {
        try {
            String url = mlServiceUrl + "/api/train/status/" + jobId;
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(500)
                    .body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * List all training jobs
     * GET /api/ml/training-jobs
     */
    @GetMapping("/training-jobs")
    public ResponseEntity<Map<String, Object>> listTrainingJobs() {
        try {
            String url = mlServiceUrl + "/api/train/list";
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(500)
                    .body(Map.of("error", e.getMessage()));
        }
    }
}
