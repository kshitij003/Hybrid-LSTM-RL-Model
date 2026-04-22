package com.trader.trading_backend.component;

import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

/**
 * StartupSyncRunner — lightweight startup check.
 *
 * Market data sync has been removed: the ML service (Python/Flask) fetches
 * OHLCV data directly from yfinance on every /api/predict call.
 * Spring Boot only needs to manage portfolio state, trade history, and P&L.
 */
@Component
public class StartupSyncRunner implements CommandLineRunner {

    @Override
    public void run(String... args) {
        System.out.println("==============================================");
        System.out.println("✅ Trading Backend started.");
        System.out.println("   Market data is fetched on-demand by the ML");
        System.out.println("   service directly from yfinance — no DB sync");
        System.out.println("   required at startup.");
        System.out.println("==============================================");
    }
}
