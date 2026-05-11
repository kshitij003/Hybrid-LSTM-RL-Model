package com.trader.trading_backend.component;

import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.Repository.Portfolio_Trading_Engine.HoldingRepository;
import com.trader.trading_backend.Repository.Portfolio_Trading_Engine.PortfolioRepository;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

@Component
@RequiredArgsConstructor
public class DBDebugger implements CommandLineRunner {
    private final PortfolioRepository portfolioRepo;
    private final StockRepository stockRepo;
    private final HoldingRepository holdingRepo;

    @Override
    @Transactional
    public void run(String... args) throws Exception {
        System.out.println("\n🔍 [DB DEBUGGER] Checking State...");
        portfolioRepo.findAll().forEach(p -> {
            System.out.println("Portfolio: " + p.getPortfolioName() + " (ID: " + p.getId() + ")");
            System.out.println("  Cash: " + p.getCurrentCashBalance());
            p.getHoldings().forEach(h -> {
                System.out.println("  Holding: " + h.getStock().getTicker() + " | Qty: " + h.getQuantity() + " | Price: " + h.getStock().getCurrentPrice());
            });
        });
        System.out.println("🔍 [DB DEBUGGER] Done.\n");
    }
}
